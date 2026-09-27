import json
import os
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from df2_pi.animation import AnimationRegistry
from df2_pi.engine import FrameClock, Runner
from df2_pi.geometry import FloorGeometry
from df2_pi.output import FanOut, PreviewSink, decode_preview
from df2_pi.pixels import TileFrame
from df2_pi.web import AppContext, create_app
from df2_pi.web.preview import ACK_WINDOW

# The frame number in every tile's red channel and a constant in green, so a
# decoded record identifies the frame it came from.
COUNTER = '''
from df2_pi.animation import animation
from df2_pi.pixels import TileFrame

@animation(name="Counter")
def render(previous, ctx):
    frame = TileFrame.black(ctx.geometry)
    frame.data[:] = (ctx.frame % 256, 7, 0)
    return frame
'''

FPS = 30.0
FULL_BYTES = 7 + 64 * 60 * 3
TILES_BYTES = 7 + 64 * 3


class PacedFakeTime:
    """Fake time for the clock, with a small real sleep per frame so the
    render thread neither spins a core nor outruns a test by thousands of
    frames."""

    def __init__(self, real_s: float = 0.002) -> None:
        self.t = 100.0
        self.real_s = real_s

    def now(self) -> float:
        self.t += 1e-6
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        time.sleep(self.real_s)


class Recorder:
    """Keeps every frame the fan-out submits, by frame number. (Not called
    `frames`: FanOut.state() reports a sink attribute of that name.)"""

    name = "recorder"

    def __init__(self) -> None:
        self.submitted: dict[int, TileFrame] = {}

    def submit(self, frame, info, effects=None):
        self.submitted[info.n] = frame

    def close(self):
        pass


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    d = tmp_path / "animations"
    d.mkdir()
    (d / "counter.py").write_text(textwrap.dedent(COUNTER))
    return AnimationRegistry.discover(d)


def fake_clock(real_s: float = 0.002) -> FrameClock:
    fake = PacedFakeTime(real_s)
    return FrameClock(fps=FPS, spin_margin=0.0, now=fake.now, sleep=fake.sleep)


def make_context(registry, clock: FrameClock | None = None) -> tuple[AppContext, Recorder]:
    if clock is None:
        clock = fake_clock()
    recorder = Recorder()
    preview = PreviewSink()
    fanout = FanOut([recorder, preview])
    runner = Runner(registry, fanout, clock=clock)
    runner.play_animation("counter")
    return AppContext(registry, None, fanout, runner, preview), recorder


def subscribers(ctx: AppContext) -> int:
    return ctx.runner.state.sinks["preview"]["subscriber_count"]


def take(ws) -> bytes:
    """Receive one record over a TestClient socket and ack it, as the page does."""
    raw = ws.receive_bytes()
    ws.send_text(json.dumps({"ack": decode_preview(raw).frame_no}))
    return raw


@contextmanager
def real_server(ctx: AppContext):
    """The app on a real uvicorn socket; yields the port. For what TestClient
    cannot show: its in-memory transport has no flow control at all."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(create_app(ctx), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        wait_until(lambda: server.started, timeout=10.0)
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(10.0)


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.005)


# ---- the stream -----------------------------------------------------------------------------


def test_records_are_what_the_fan_out_submitted_with_rising_frame_numbers(registry):
    ctx, recorder = make_context(registry)
    with TestClient(create_app(ctx)) as client, client.websocket_connect("/ws/preview") as ws:
        records = [take(ws) for _ in range(10)]
    numbers = [decode_preview(r).frame_no for r in records]
    assert numbers == sorted(set(numbers)), numbers
    for raw in records:
        assert len(raw) == FULL_BYTES
        record = decode_preview(raw)
        assert record.fmt == "full" and record.tile_source
        submitted = recorder.submitted[record.frame_no].to_pixels()
        assert np.array_equal(record.frame.data, submitted.data)


def test_tiles_records_are_199_bytes_and_fps_10_is_every_third_frame(registry):
    # Paced slowly enough that the client reads every record it is sent.
    ctx, recorder = make_context(registry, clock=fake_clock(real_s=0.01))
    with TestClient(create_app(ctx)) as client, client.websocket_connect("/ws/preview?format=tiles&fps=10") as ws:
        records = [take(ws) for _ in range(8)]
    assert {len(r) for r in records} == {TILES_BYTES}
    numbers = [decode_preview(r).frame_no for r in records]
    steps = np.diff(numbers)
    assert all(step % 3 == 0 for step in steps), numbers  # rate-limited against frame time
    assert 3 in steps, numbers
    for raw in records:
        record = decode_preview(raw)
        assert np.array_equal(record.frame.data, recorder.submitted[record.frame_no].data)


def test_a_mid_stream_change_takes_effect(registry):
    ctx, _ = make_context(registry)
    with TestClient(create_app(ctx)) as client, client.websocket_connect("/ws/preview") as ws:
        assert len(take(ws)) == FULL_BYTES
        ws.send_text(json.dumps({"format": "tiles", "fps": 10}))
        tiles = []
        deadline = time.monotonic() + 5
        while len(tiles) < 5 and time.monotonic() < deadline:
            raw = take(ws)
            if len(raw) == TILES_BYTES:  # a full record already in flight may precede them
                tiles.append(decode_preview(raw).frame_no)
        assert len(tiles) == 5
        assert all(step % 3 == 0 for step in np.diff(tiles)), tiles
        assert subscribers(ctx) == 1  # re-subscribed, not added


@pytest.mark.parametrize(
    "message, fragment",
    [
        ("not json", "Expecting value"),
        ('{"fps": -1}', "fps must be a positive number"),
        ('{"format": "jpeg"}', "format must be one of"),
        ('{"speed": 2}', 'expected {"ack": n}'),
        ('{"ack": "7"}', 'an ack is {"ack": FRAME_NO}'),
        ('{"ack": -1}', 'an ack is {"ack": FRAME_NO}'),
        ('{"ack": 3, "fps": 10}', 'an ack is {"ack": FRAME_NO}'),
    ],
)
def test_a_bad_control_message_gets_an_error_and_the_stream_carries_on(registry, message, fragment):
    ctx, _ = make_context(registry)
    with TestClient(create_app(ctx)) as client, client.websocket_connect("/ws/preview?format=tiles") as ws:
        take(ws)
        ws.send_text(message)
        deadline = time.monotonic() + 5
        while True:
            reply = ws.receive()
            if reply.get("text") is not None:
                assert fragment in json.loads(reply["text"])["error"]
                break
            ws.send_text(json.dumps({"ack": decode_preview(reply["bytes"]).frame_no}))
            assert time.monotonic() < deadline
        assert len(take(ws)) == TILES_BYTES


@pytest.mark.parametrize("query", ["format=jpeg", "fps=0", "fps=-5"])
def test_a_bad_query_is_refused(registry, query):
    ctx, _ = make_context(registry)
    with TestClient(create_app(ctx)) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(f"/ws/preview?{query}") as ws:
                ws.receive_bytes()
        assert refused.value.code == 1008
        assert subscribers(ctx) == 0


# ---- subscription lifecycle -----------------------------------------------------------------


def test_disconnect_unsubscribes_and_sink_health_shows_it(registry):
    ctx, _ = make_context(registry)
    with TestClient(create_app(ctx)) as client:
        assert subscribers(ctx) == 0
        with client.websocket_connect("/ws/preview") as a, client.websocket_connect("/ws/preview?format=tiles") as b:
            a.receive_bytes()
            b.receive_bytes()
            wait_until(lambda: subscribers(ctx) == 2)
            assert client.get("/api/state").json()["sinks"]["preview"]["subscriber_count"] == 2
        wait_until(lambda: subscribers(ctx) == 0)


def test_the_socket_closes_when_the_sink_goes(registry):
    """Server shutdown closes the fan-out, and with it every subscription:
    the browser is told the server is going rather than left hanging."""
    ctx, _ = make_context(registry)
    with TestClient(create_app(ctx)) as client, client.websocket_connect("/ws/preview?format=tiles") as ws:
        ws.receive_bytes()
        ctx.runner.stop()
        ctx.runner.join(5.0)
        with pytest.raises(WebSocketDisconnect) as closed:
            while True:
                ws.receive_bytes()
        assert closed.value.code == 1001


# ---- geometry -------------------------------------------------------------------------------


def test_geometry_matches_the_default_floor(registry):
    ctx, _ = make_context(registry)
    geometry = FloorGeometry.default()
    body = TestClient(create_app(ctx)).get("/api/preview/geometry").json()
    assert {k: body[k] for k in ("tile_rows", "tile_cols", "tiles", "leds_per_side", "leds_per_tile", "led_count", "cell_size", "width", "height")} == {
        "tile_rows": geometry.tile_rows,
        "tile_cols": geometry.tile_cols,
        "tiles": geometry.tiles,
        "leds_per_side": geometry.leds_per_side,
        "leds_per_tile": geometry.leds_per_tile,
        "led_count": geometry.led_count,
        "cell_size": geometry.cell_size,
        "width": geometry.width,
        "height": geometry.height,
    }
    cells = np.array(body["led_to_cell"]).reshape(geometry.tiles, geometry.leds_per_tile, 2)
    assert np.array_equal(cells, geometry.led_to_cell)
    assert body["display"]["flip_y"] is True


# ---- flow control ---------------------------------------------------------------------------
# A real server and socket: TestClient's in-memory transport has no flow
# control, so it cannot show what a stalled browser does to the server.


def test_a_client_that_never_acks_gets_the_window_then_silence(registry):
    from websockets.sync.client import connect

    ctx, _ = make_context(registry)
    with real_server(ctx) as port, connect(f"ws://127.0.0.1:{port}/ws/preview", max_size=None) as ws:
        got = [decode_preview(ws.recv(timeout=5)).frame_no for _ in range(ACK_WINDOW)]
        before = ctx.runner.state.frame
        with pytest.raises(TimeoutError):
            ws.recv(timeout=0.5)
        assert ctx.runner.state.frame - before > 50  # the runner went on without it
    assert got == sorted(got)


def test_an_ack_releases_the_newest_frame_not_a_backlog(registry):
    from websockets.sync.client import connect

    ctx, _ = make_context(registry)
    with real_server(ctx) as port, connect(f"ws://127.0.0.1:{port}/ws/preview", max_size=None) as ws:
        last = [decode_preview(ws.recv(timeout=5)).frame_no for _ in range(ACK_WINDOW)][-1]
        wait_until(lambda: ctx.runner.state.frame > last + 100)  # stalled well behind
        ws.send(json.dumps({"ack": last}))
        resumed = decode_preview(ws.recv(timeout=5)).frame_no
        assert resumed > last + 100, (last, resumed)  # current, not the next in line


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("DF2_SLOW_TESTS"), reason="set DF2_SLOW_TESTS=1 to run")
def test_a_stalled_client_never_slows_the_runner(registry):
    """Real time: while a client stops acking for 1.5 s the render clock
    drops nothing, and when it acks again it gets the present."""
    from websockets.sync.client import connect

    ctx, _ = make_context(registry, clock=FrameClock(fps=60.0))
    with real_server(ctx) as port, connect(f"ws://127.0.0.1:{port}/ws/preview", max_size=None) as ws:
        last = [decode_preview(ws.recv(timeout=5)).frame_no for _ in range(ACK_WINDOW)][-1]
        before = ctx.runner.state.frame
        time.sleep(1.5)
        produced = ctx.runner.state.frame - before
        ws.send(json.dumps({"ack": last}))
        resumed = decode_preview(ws.recv(timeout=5)).frame_no
    assert produced >= 80, produced
    assert ctx.runner.clock.dropped == 0
    assert resumed >= before + produced - 2, (before, produced, resumed)
