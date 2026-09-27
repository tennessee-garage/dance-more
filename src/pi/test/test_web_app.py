import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from df2_pi import cli
from df2_pi.animation import AnimationRegistry
from df2_pi.engine import FrameClock, Runner
from df2_pi.output import FanOut, NullSink, PreviewSink
from df2_pi.playlists import PlaylistStore
from df2_pi.web import AppContext, create_app
from df2_pi.web.app import STATIC_DIR

SOLID = '''
from df2_pi.animation import animation
from df2_pi.pixels import TileFrame

@animation(name="{name}")
def render(previous, ctx):
    frame = TileFrame.black(ctx.geometry)
    frame.data[:] = (10, min(255, ctx.frame), 0)
    return frame
'''


class FakeTime:
    """Fake clock time that advances by exactly what the clock asks to
    sleep, plus a real millisecond so the render thread does not spin a
    core flat out under the tests."""

    def __init__(self) -> None:
        self.t = 100.0

    def now(self) -> float:
        self.t += 1e-6
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        time.sleep(0.001)


class Recorder:
    """A hardware-like sink: records the calls the runner's shutdown makes."""

    name = "recorder"

    def __init__(self) -> None:
        self.events: list[str] = []
        self.frames = 0

    def submit(self, frame, info, effects=None):
        self.frames += 1

    def latch(self):
        self.events.append("latch")

    def blackout(self):
        self.events.append("blackout")

    def close(self):
        self.events.append("close")


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    d = tmp_path / "animations"
    d.mkdir()
    (d / "solid.py").write_text(textwrap.dedent(SOLID.format(name="Solid")))
    return AnimationRegistry.discover(d)


@pytest.fixture
def store(registry) -> PlaylistStore:
    s = PlaylistStore(":memory:", registry=registry)
    yield s
    s.close()


def make_context(registry, store, *sinks, clock: FrameClock | None = None) -> AppContext:
    if clock is None:
        fake = FakeTime()
        clock = FrameClock(fps=30.0, spin_margin=0.0, now=fake.now, sleep=fake.sleep)
    preview = PreviewSink()
    fanout = FanOut([*sinks, preview])
    runner = Runner(registry, fanout, store=store, clock=clock)
    return AppContext(registry, store, fanout, runner, preview)


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.005)


# ---- construction ---------------------------------------------------------------------------


def test_create_app_builds_without_starting_the_runner(registry, store):
    ctx = make_context(registry, store, NullSink())
    app = create_app(ctx)
    assert isinstance(app, FastAPI)
    assert not ctx.runner.alive  # nothing runs until the lifespan does


def test_create_app_imports_no_hardware_and_the_driver_imports_no_fastapi():
    """A subprocess, since other tests in this session import serial."""
    code = textwrap.dedent('''
        import sys
        import df2_pi.cli, df2_pi.engine, df2_pi.output, df2_pi.playlists, df2_pi.animation
        assert "fastapi" not in sys.modules, "the driver imported fastapi"

        from df2_pi.animation import AnimationRegistry
        from df2_pi.engine import Runner
        from df2_pi.output import FanOut, NullSink, PreviewSink
        from df2_pi.web import AppContext, create_app

        registry = AnimationRegistry([])
        preview = PreviewSink()
        fanout = FanOut([NullSink(), preview])
        create_app(AppContext(registry, None, fanout, Runner(registry, fanout), preview))
        hardware = sorted(m for m in ("serial", "gpiozero", "lgpio", "df2_pi.transport") if m in sys.modules)
        assert not hardware, hardware
    ''')
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---- lifecycle ------------------------------------------------------------------------------


def test_startup_starts_the_runner_on_the_startup_playlist(registry, store):
    ctx = make_context(registry, store, NullSink())
    with TestClient(create_app(ctx)) as client:
        assert ctx.runner.alive
        wait_until(lambda: ctx.runner.state.frame > 0 and ctx.runner.state.playlist is not None)
        # An empty database seeds a Default playlist, exactly as `df2-pi play` does.
        assert ctx.runner.state.playlist[1] == "Default"
        assert ctx.runner.state.animation == ("solid", "Solid")
        body = client.get("/healthz").json()
        assert body["ok"] is True and body["runner"] == "alive" and body["frame"] > 0


def test_startup_without_a_store_plays_idle(registry):
    ctx = make_context(registry, None, NullSink())
    with TestClient(create_app(ctx)):
        wait_until(lambda: ctx.runner.state.frame > 0)
        assert ctx.runner.state.animation == ("_idle", "Idle")


def test_shutdown_stops_the_runner_with_blackout_then_latch(registry, store):
    recorder = Recorder()
    ctx = make_context(registry, store, recorder)
    with TestClient(create_app(ctx)):
        wait_until(lambda: recorder.frames > 3)
    assert not ctx.runner.alive
    # The render loop latches every frame; shutdown ends blackout, latch, close.
    assert recorder.events[-3:] == ["blackout", "latch", "close"]


def test_healthz_is_503_once_the_runner_has_exited(registry, store):
    ctx = make_context(registry, store, NullSink())
    with TestClient(create_app(ctx)) as client:
        wait_until(lambda: ctx.runner.state.frame > 0)
        ctx.runner.stop()
        ctx.runner.join(5.0)
        response = client.get("/healthz")
        assert response.status_code == 503
        assert response.json()["ok"] is False
        assert response.json()["runner"] == "stopped"


# ---- the page shell -------------------------------------------------------------------------


def _importmap(html: str) -> dict[str, str]:
    match = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
    assert match, "no importmap in index.html"
    return json.loads(match.group(1))["imports"]


def _url(ref: str) -> str:
    """A relative reference in index.html, as a path from the site root."""
    return "/" + ref.removeprefix("./")


def test_index_serves_the_shell_and_every_referenced_file_resolves(registry, store):
    ctx = make_context(registry, store, NullSink())
    with TestClient(create_app(ctx)) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["content-type"].startswith("text/html")
        html = index.text

        refs = re.findall(r'(?:href|src)="([^"]+)"', html)
        refs += list(_importmap(html).values())
        assert any(r.endswith("app.css") for r in refs) and any(r.endswith("app.js") for r in refs)
        for ref in refs:
            response = client.get(_url(ref))
            assert response.status_code == 200, ref
            assert response.headers["cache-control"] == "no-cache", ref
            if ref.endswith(".js"):
                assert "javascript" in response.headers["content-type"], ref


def test_every_module_import_resolves():
    """Bare specifiers - in our modules and in the vendored ones - must be in
    the importmap; relative imports must exist. A browser would fail the
    whole page on either, and nothing else here runs the JS."""
    imports = _importmap((STATIC_DIR / "index.html").read_text())
    for path in sorted(STATIC_DIR.rglob("*.js")):
        source = path.read_text()
        for spec in re.findall(r'''(?:\bfrom|\bimport)\s*["']([^"']+)["']''', source):
            if spec.startswith("."):
                assert (path.parent / spec).resolve().is_file(), f"{path.name}: {spec}"
            else:
                assert spec in imports, f"{path.name} imports {spec!r}, which the importmap does not map"


def test_static_files_are_covered_by_package_data():
    """pyproject's package-data globs are `static/*` and `static/vendor/*`
    and do not recurse; a file deeper than that would not ship in a
    non-editable install."""
    for path in STATIC_DIR.rglob("*"):
        if path.is_file():
            rel = path.relative_to(STATIC_DIR)
            assert len(rel.parts) == 1 or (len(rel.parts) == 2 and rel.parts[0] == "vendor"), rel


# ---- df2-pi serve ---------------------------------------------------------------------------


@pytest.fixture
def serve_args(tmp_path: Path, registry) -> list[str]:
    anim_dir = tmp_path / "animations"
    return ["--db", str(tmp_path / "df2.sqlite3"), "--animations", str(anim_dir)]


@pytest.fixture(autouse=False)
def no_serial(monkeypatch):
    import serial

    def boom(*a, **k):
        raise AssertionError("serial.Serial was touched in a headless run")

    monkeypatch.setattr(serial, "Serial", boom)


def test_serve_no_hardware_builds_the_app_without_binding(serve_args, no_serial, monkeypatch):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("build_app must not start a server"))
    args = cli.build_parser().parse_args([*serve_args, "serve", "--no-hardware", "--fps", "25", "--brightness", "100"])
    app = cli.build_app(args)
    ctx: AppContext = app.state.ctx
    assert not ctx.runner.alive
    assert ctx.runner.fps == 25.0
    assert ctx.runner.state.brightness == 100
    assert ctx.fanout.get("preview") is ctx.preview
    assert ctx.fanout.get("hardware") is None


def test_serve_hands_the_app_to_uvicorn(serve_args, no_serial, monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    assert cli.main([*serve_args, "serve", "--no-hardware", "--host", "127.0.0.1", "--port", "9123"]) == 0
    (app, kw), = calls
    assert isinstance(app, FastAPI)
    assert kw["host"] == "127.0.0.1" and kw["port"] == 9123


# ---- the GIL check --------------------------------------------------------------------------

GIL_WINDOW_S = 3.0
GIL_JITTER_TOLERANCE_MS = 1.0
# One client per Pi 5 core, each flat out: far more than any real use (a
# browser polls at 2 Hz) while leaving the render thread a core. More
# clients than cores measures CPU oversubscription, not the GIL - the render
# thread then waits on the OS scheduler, which only realtime priority fixes.
GIL_CLIENTS = 4
GIL_CLIENT_START_S = 1.5  # interpreter + httpx2 import on the Pi

# A client is a separate process, as a browser is, so the load competes for
# the CPU but not for the server's GIL.
GIL_CLIENT = """
import sys, time, httpx2
base, seconds, offset, paths = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
deadline, count = time.monotonic() + seconds, 0
with httpx2.Client(base_url=base) as client:
    while time.monotonic() < deadline:
        client.get(paths[(offset + count) % len(paths)])
        count += 1
print(count)
"""


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("DF2_SLOW_TESTS"), reason="set DF2_SLOW_TESTS=1 to run")
def test_request_burst_does_not_disturb_the_render_clock(registry, store):
    """The runner shares an interpreter with uvicorn's event loop. Under
    `GIL_CLIENTS` clients requesting as fast as they can, the clock must
    drop no frames and its jitter p95 must stay within a millisecond of the
    idle baseline. Real time, a real server on a real socket: run before
    every web PR, and on the Pi for a verdict that means anything."""
    import httpx2
    import uvicorn

    window = int(GIL_WINDOW_S * 30)
    ctx = make_context(registry, store, NullSink(), clock=FrameClock(fps=30.0, window=window))
    server = uvicorn.Server(uvicorn.Config(create_app(ctx), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    clients: list[subprocess.Popen] = []
    try:
        wait_until(lambda: server.started, timeout=10.0)
        port = server.servers[0].sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        paths = ["/", "/healthz"] + [_url(r) for r in _importmap(httpx2.get(base + "/").text).values()]

        time.sleep(GIL_WINDOW_S)  # a full telemetry window at idle
        idle = ctx.runner.clock.telemetry()

        run_s = GIL_CLIENT_START_S + GIL_WINDOW_S + 0.5
        clients = [
            subprocess.Popen(
                [sys.executable, "-c", GIL_CLIENT, base, str(run_s), str(i), *paths],
                stdout=subprocess.PIPE,
                text=True,
            )
            for i in range(GIL_CLIENTS)
        ]
        # The telemetry window is a frame count, so once a full window has
        # passed under load it holds loaded frames only.
        time.sleep(GIL_CLIENT_START_S + GIL_WINDOW_S)
        loaded = ctx.runner.clock.telemetry()
        requests = sum(int(c.communicate(timeout=10.0)[0]) for c in clients)
    finally:
        for c in clients:
            if c.poll() is None:
                c.kill()
        server.should_exit = True
        thread.join(10.0)

    print(
        f"\n{requests} requests from {GIL_CLIENTS} clients; jitter p95 idle {idle.jitter_ms.p95:.3f} ms, "
        f"loaded {loaded.jitter_ms.p95:.3f} ms (max {loaded.jitter_ms.max:.3f}); "
        f"dropped {idle.dropped} -> {loaded.dropped}"
    )
    assert requests > 100, "the burst never got going"
    assert loaded.dropped == idle.dropped
    assert loaded.jitter_ms.p95 <= idle.jitter_ms.p95 + GIL_JITTER_TOLERANCE_MS
