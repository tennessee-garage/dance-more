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


OK, EMPTY, SILENT, FAILED = 0x01, 0x00, 0x02, 0x03


def status_payload(state=0x02, slots=(OK,) * 7 + (EMPTY,), uptime=3725) -> bytes:
    return bytes([state, sum(s == OK for s in slots)]) + bytes(slots) + uptime.to_bytes(4, "big")


def version_payload(row: FirmwareVersion, tiles: list[FirmwareVersion | None]) -> bytes:
    valid = sum(1 << i for i, t in enumerate(tiles) if t is not None)
    body = b"".join((t or GOOD).encode() for t in tiles)
    return row.encode() + bytes([valid]) + body


class FakeFloor:
    """Records what reaches the Row Bus, in order, and on which thread."""

    def __init__(self, dead_rows=(), dirty_row=None, slots=None, powerless_rows=()) -> None:
        self.chain_map = RowChainMap.alternating(2)
        self.events: list[tuple[str, str]] = []
        self.dead_rows = set(dead_rows)
        self.dirty_row = dirty_row
        self.slots = slots or {}  # row -> STATUS tile_status; default: 7 tiles, slot 7 empty
        self.powerless_rows = set(powerless_rows)  # answer STATUS but not POWER
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
            return Frame(row, Resp.STATUS_RESP, status_payload(slots=self.slots.get(row, (OK,) * 7 + (EMPTY,))))
        if cmd == Cmd.POWER:
            if row in self.powerless_rows:
                raise RowNotResponding(f"row 0x{row:02X} did not answer cmd 0x03 after 3 attempts")
            # 12.1 V, 1.25 A + 100 mA per row, so each row's reading is its own
            mA = 1250 + 100 * row
            return Frame(row, Resp.POWER_RESP, (12100).to_bytes(2, "big") + mA.to_bytes(2, "big") + (12100 * mA // 1000).to_bytes(2, "big"))
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
    floor = FakeFloor(dead_rows={5}, powerless_rows={6})
    app, runner = floor_app(registry, floor)
    with TestClient(app) as client:
        body = client.get("/api/floor/status").json()
    rows = body["rows"]
    assert [r["row"] for r in rows] == list(range(8))
    assert [r["chain"] for r in rows] == [0, 1] * 4
    alive = rows[0]
    assert (alive["responding"], alive["state"], alive["tiles_found"], alive["uptime_s"]) == (True, "running", 7, 3725)
    assert alive["tile_status"] == [OK] * 7 + [EMPTY]
    assert rows[5]["responding"] is False and "did not answer" in rows[5]["error"]
    assert (alive["voltage_mV"], alive["current_mA"], alive["power_mW"]) == (12100, 1250, 15125)
    assert rows[3]["current_mA"] == 1550
    assert rows[5]["voltage_mV"] is None  # a dead row is not asked for POWER
    assert rows[6]["responding"] is True and rows[6]["state"] == "running" and rows[6]["voltage_mV"] is None
    asked_power = [e for e, _ in floor.events if e.startswith("request")]
    assert "request 5" in asked_power and asked_power.count("request 5") == 1  # STATUS only
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
    assert slot7["version"] is None and slot7["state"] == "empty" and slot7["out_of_step"] is False
    assert rows[0]["tiles"][0]["out_of_step"] is False
    requests_are_between_frames(floor.events)
    assert runner.clock.dropped == 0


def test_the_majorities_are_served_as_what_everything_is_compared_against(registry):
    floor = FakeFloor(dirty_row=6)
    app, _ = floor_app(registry, floor)
    with TestClient(app) as client:
        body = client.get("/api/floor/version").json()
    assert body["row_version"]["git_sha"] == "1c00158a" and body["row_version"]["dirty"] is False
    assert body["tile_version"]["text"] == body["rows"][0]["tiles"][0]["version"]["text"]


def test_a_missing_tile_version_is_labelled_from_the_slot_status(registry):
    """VERSION alone cannot tell an empty slot from a tile that is there
    and never answered; each row's STATUS can."""
    floor = FakeFloor(slots={
        0: (OK,) * 7 + (EMPTY,),
        1: (OK,) * 7 + (OK,),  # found, but its version never arrived
        3: (OK,) * 7 + (SILENT,),
        4: (OK,) * 7 + (FAILED,),
    })
    app, runner = floor_app(registry, floor)
    with TestClient(app) as client:
        body = client.get("/api/floor/version").json()
    slot7 = {r["row"]: r["tiles"][7] for r in body["rows"]}
    assert (slot7[0]["state"], slot7[0]["out_of_step"]) == ("empty", False)
    assert (slot7[1]["state"], slot7[1]["out_of_step"]) == ("no version", True)
    assert (slot7[3]["state"], slot7[3]["out_of_step"]) == ("not responding", True)
    assert (slot7[4]["state"], slot7[4]["out_of_step"]) == ("test failed", True)
    assert body["rows"][0]["tiles"][0]["state"] == "ok"
    requests_are_between_frames(floor.events)
    assert runner.clock.dropped == 0


def test_a_floor_whose_only_gaps_are_empty_slots_is_ok(registry):
    app, _ = floor_app(registry, FakeFloor())  # every row: 7 tiles, slot 7 empty
    with TestClient(app) as client:
        body = client.get("/api/floor/version").json()
    assert body["ok"] is True
    assert all(r["tiles"][7]["state"] == "empty" for r in body["rows"])


def test_a_clean_floor_is_ok(registry):
    floor = FakeFloor()

    def full_floor(row, cmd, payload=b""):
        if cmd == Cmd.STATUS:
            return Frame(row, Resp.STATUS_RESP, status_payload(slots=(OK,) * 8))
        return Frame(row, Resp.VERSION_RESP, version_payload(GOOD, [GOOD] * 8))

    floor.request = full_floor
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
