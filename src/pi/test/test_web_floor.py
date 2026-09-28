import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from df2_pi.animation import AnimationRegistry
from df2_pi.engine import FrameClock, Runner
from df2_pi.output import FanOut, HardwareSink, NullSink, PreviewSink
from df2_pi.protocol.constants import Cmd, Resp
from df2_pi.protocol.firmware_version import FirmwareVersion
from df2_pi.protocol.frame import Frame
from df2_pi.transport.chain_map import RowChainMap
from df2_pi.transport.floor import RowNotResponding
from df2_pi.web import AppContext, create_app

GOOD = FirmwareVersion(9, 0x1C00158A, 0)
DIRTY = FirmwareVersion(9, 0x1C00158A, 0x01)


class PacedFakeTime:
    def __init__(self) -> None:
        self.t = 100.0

    def now(self) -> float:
        self.t += 1e-6
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        time.sleep(0.002)


def status_payload(state=0x02, tiles=8, uptime=3725) -> bytes:
    return bytes([state, tiles]) + bytes(8) + uptime.to_bytes(4, "big")


def version_payload(row: FirmwareVersion, tiles: list[FirmwareVersion | None]) -> bytes:
    valid = sum(1 << i for i, t in enumerate(tiles) if t is not None)
    body = b"".join((t or GOOD).encode() for t in tiles)
    return row.encode() + bytes([valid]) + body


class FakeFloor:
    """Records what reaches the Row Bus, in order, and on which thread."""

    def __init__(self, dead_rows=(), dirty_row=None) -> None:
        self.chain_map = RowChainMap.alternating(2)
        self.events: list[tuple[str, str]] = []
        self.dead_rows = set(dead_rows)
        self.dirty_row = dirty_row
        self._lock = threading.Lock()

    def _log(self, what: str) -> None:
        with self._lock:
            self.events.append((what, threading.current_thread().name))

    def send_rows(self, payloads, cmd=Cmd.SEND_DATA) -> None:
        self._log("send_rows")

    def latch(self) -> None:
        self._log("latch")

    def blackout(self) -> None:
        self._log("blackout")

    def request(self, row: int, cmd: int, payload: bytes = b"") -> Frame:
        self._log(f"request {row}")
        if row in self.dead_rows:
            raise RowNotResponding(f"row 0x{row:02X} did not answer cmd 0x{cmd:02X} after 3 attempts")
        if cmd == Cmd.STATUS:
            return Frame(row, Resp.STATUS_RESP, status_payload())
        if cmd == Cmd.VERSION:
            head = DIRTY if row == self.dirty_row else GOOD
            return Frame(row, Resp.VERSION_RESP, version_payload(head, [GOOD] * 7 + [None]))
        raise AssertionError(cmd)


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    d = tmp_path / "animations"
    d.mkdir()
    return AnimationRegistry.discover(d)  # nothing: the idle animation plays


def floor_app(registry, floor):
    fake = PacedFakeTime()
    clock = FrameClock(fps=30.0, spin_margin=0.0, now=fake.now, sleep=fake.sleep)
    preview = PreviewSink()
    fanout = FanOut([HardwareSink(floor, clock=clock), preview])
    runner = Runner(registry, fanout, clock=clock)
    return create_app(AppContext(registry, None, fanout, runner, preview)), runner


def requests_are_between_frames(events) -> None:
    """Every request on the render thread at a frame boundary - never
    between a frame's send_rows and its latch, when the frame is on the
    wire - and never two in one frame."""
    names = [e for e, _ in events]
    for i, (event, thread) in enumerate(events):
        if not event.startswith("request") and event != "blackout":
            continue
        assert thread == "render", events[i]
        before = [n for n in names[:i] if n in ("latch", "send_rows")]
        assert not before or before[-1] == "latch", f"{event} while a frame was on the wire"
        last_send = max((j for j, n in enumerate(names[:i]) if n == "send_rows"), default=-1)
        earlier_requests = [j for j, n in enumerate(names[:i]) if n.startswith("request") and j > last_send]
        assert not earlier_requests, f"two requests in one frame at {i}"


# ---- the admin requests ---------------------------------------------------------------------


def test_status_asks_every_row_between_frames(registry):
    floor = FakeFloor(dead_rows={5})
    app, runner = floor_app(registry, floor)
    with TestClient(app) as client:
        body = client.get("/api/floor/status").json()
    rows = body["rows"]
    assert [r["row"] for r in rows] == list(range(8))
    assert [r["chain"] for r in rows] == [0, 1] * 4
    alive = rows[0]
    assert (alive["responding"], alive["state"], alive["tiles_found"], alive["uptime_s"]) == (True, "running", 8, 3725)
    assert rows[5]["responding"] is False and "did not answer" in rows[5]["error"]
    requests_are_between_frames(floor.events)
    assert runner.clock.dropped == 0


def test_version_reports_what_is_out_of_step(registry):
    floor = FakeFloor(dead_rows={2}, dirty_row=6)
    app, runner = floor_app(registry, floor)
    with TestClient(app) as client:
        body = client.get("/api/floor/version").json()
    rows = {r["row"]: r for r in body["rows"]}
    assert body["ok"] is False
    assert rows[0]["version"]["text"] and rows[0]["out_of_step"] is False
    assert rows[0]["version"]["git_sha"] == "1c00158a" and rows[0]["version"]["dirty"] is False
    assert rows[2]["responding"] is False and rows[2]["out_of_step"] is True
    assert rows[6]["version"]["dirty"] is True and rows[6]["out_of_step"] is True
    slot7 = rows[0]["tiles"][7]
    assert slot7["version"] is None and slot7["out_of_step"] is True  # no version cached
    assert rows[0]["tiles"][0]["out_of_step"] is False
    requests_are_between_frames(floor.events)
    assert runner.clock.dropped == 0


def test_a_clean_floor_is_ok(registry):
    floor = FakeFloor()
    floor.request = lambda row, cmd, payload=b"": Frame(row, Resp.VERSION_RESP, version_payload(GOOD, [GOOD] * 8))
    app, _ = floor_app(registry, floor)
    with TestClient(app) as client:
        assert client.get("/api/floor/version").json()["ok"] is True


def test_blackout_is_the_broadcast_on_the_render_thread(registry):
    floor = FakeFloor()
    app, _ = floor_app(registry, floor)
    with TestClient(app) as client:
        assert client.post("/api/floor/blackout").status_code == 204
        blackouts = [e for e in floor.events if e[0] == "blackout"]
    assert blackouts[0] == ("blackout", "render")
    requests_are_between_frames(floor.events[: floor.events.index(blackouts[0]) + 1])


def test_without_hardware_every_floor_route_is_503(registry):
    preview = PreviewSink()
    fanout = FanOut([NullSink(), preview])
    runner = Runner(registry, fanout, clock=FrameClock(fps=30.0, spin_margin=0.0, now=PacedFakeTime().now, sleep=PacedFakeTime().sleep))
    client = TestClient(create_app(AppContext(registry, None, fanout, runner, preview)))
    for method, path in [("GET", "/api/floor/status"), ("GET", "/api/floor/version"), ("POST", "/api/floor/blackout")]:
        response = client.request(method, path)
        assert response.status_code == 503, path
        assert "--no-hardware" in response.json()["detail"]


def test_with_the_runner_stopped_it_is_503(registry):
    app, runner = floor_app(registry, FakeFloor())
    client = TestClient(app)  # no lifespan: the render thread never started
    response = client.get("/api/floor/status")
    assert response.status_code == 503 and "not running" in response.json()["detail"]


# ---- encoder stats in the state -------------------------------------------------------------


def test_encoder_stats_appear_in_the_hardware_sink_health(registry):
    floor = FakeFloor()
    app, runner = floor_app(registry, floor)
    with TestClient(app) as client:
        deadline = time.monotonic() + 5
        while runner.state.frame < 3:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        hardware = client.get("/api/state").json()["sinks"]["hardware"]
    stats = hardware["encode_stats"]
    assert hardware["healthy"] is True and hardware.get("last_error") is None  # None values are left out
    assert len(stats["row_bytes"]) == len(stats["row_wire_ms"]) == 8
    assert [c["rows"] for c in stats["chains"]] == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert stats["wire_ms"] == pytest.approx(max(c["wire_ms"] for c in stats["chains"]))
    assert stats["wire_ms"] > 0 and stats["total_bytes"] == sum(stats["row_bytes"])
    # 10 bits a byte plus the 8-byte frame overhead, at 3.125 Mbps
    row_ms = [(8 + n) * 10 / 3_125_000 * 1000 for n in stats["row_bytes"]]
    assert stats["row_wire_ms"] == pytest.approx(row_ms)
    assert stats["chains"][0]["wire_ms"] == pytest.approx(sum(row_ms[0::2]))


def test_a_hardware_failure_shows_its_last_error(registry):
    floor = FakeFloor()
    floor.send_rows = MagicMock(side_effect=OSError("write failed"))
    app, runner = floor_app(registry, floor)
    with TestClient(app) as client:
        deadline = time.monotonic() + 5
        while runner.state.frame < 4:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        hardware = client.get("/api/state").json()["sinks"]["hardware"]
    assert hardware["healthy"] is False
    assert "OSError: write failed" in hardware["last_error"]
