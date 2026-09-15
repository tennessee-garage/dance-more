import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from df2_pi.effects import FADE, Effect
from df2_pi.encode import FrameEncoder, parse_send_data
from df2_pi.engine.clock import FrameClock, FrameInfo
from df2_pi.output import (
    CallbackSink,
    FanOut,
    HardwareSink,
    Mailbox,
    NullSink,
    PreviewSink,
    RecorderSink,
    Sink,
    decode_preview,
    encode_preview,
    read_recording,
)
from df2_pi.output.preview import HEADER_SIZE, wait_for
from df2_pi.pixels import PixelFrame, TileFrame
from df2_pi.protocol.constants import TileCmd

FPS = 30.0


def info(n: int) -> FrameInfo:
    return FrameInfo(n, n / FPS, 1000.0 + n / FPS)


def frame_with(value: int) -> PixelFrame:
    f = PixelFrame.black()
    f.data[:] = value
    return f


@pytest.fixture
def fanout():
    fo = FanOut()
    yield fo
    fo.close()


# ---- the invariant --------------------------------------------------------------------


def test_a_sink_that_sleeps_half_a_second_per_frame_delays_the_fanout_by_under_a_millisecond(fanout):
    slow = CallbackSink(lambda f, i: time.sleep(0.5), name="slow")
    seen = []
    fast = CallbackSink(lambda f, i: seen.append(i.n), name="fast")
    fanout.attach(slow)
    fanout.attach(fast)
    frame = frame_with(1)
    fanout.submit(frame, info(0))  # starts the threads; the slow one is now asleep
    assert wait_for(lambda: seen == [0])
    durations = []
    for n in range(1, 31):
        t0 = time.perf_counter()
        fanout.submit(frame, info(n))
        durations.append(time.perf_counter() - t0)
        assert wait_for(lambda: seen[-1] == n)  # paced like the clock: one at a time
    assert max(durations) < 0.001
    assert seen == list(range(31))
    assert slow.frames_handled == 0 and slow.dropped == 29


def test_latest_wins_leaves_exactly_one_pending_and_it_is_the_newest(fanout):
    release = threading.Event()
    handled = []

    def stall(frame, i):
        handled.append(i.n)
        release.wait()

    sink = CallbackSink(stall, name="stalled")
    fanout.attach(sink)
    fanout.submit(frame_with(0), info(0))
    assert wait_for(lambda: handled == [0])  # the worker is now holding frame 0
    for n in range(1, 100):
        fanout.submit(frame_with(n % 256), info(n))
    assert sink.mailbox.pending == 1
    pending_frame, pending_info = sink.mailbox.peek()
    assert pending_info.n == 99 and pending_frame.data[0, 0, 0] == 99
    assert sink.dropped == 98  # 1..98 were overwritten
    release.set()
    assert wait_for(lambda: handled == [0, 99])


def test_a_sink_that_raises_every_frame_is_detached_after_n_and_the_others_keep_receiving(fanout):
    seen = []
    bad = CallbackSink(lambda f, i: 1 / 0, name="bad", max_failures=3)
    good = CallbackSink(lambda f, i: seen.append(i.n), name="good")
    fanout.attach(bad)
    fanout.attach(good)
    for n in range(10):
        fanout.submit(frame_with(1), info(n))
        assert wait_for(lambda: good.frames_handled == n + 1)
        if bad.degraded:
            fanout.submit(frame_with(1), info(n))  # the fan-out notices on its next submit
            break
    assert wait_for(lambda: bad.degraded)
    fanout.submit(frame_with(1), info(50))
    assert "bad" not in [s.name for s in fanout.sinks]
    assert "ZeroDivisionError" in fanout.detached["bad"]
    assert wait_for(lambda: seen[-1] == 50)
    state = fanout.state()
    assert state["sinks"]["bad"] == {"attached": False, "reason": fanout.detached["bad"]}
    assert state["sinks"]["good"]["attached"] and state["sinks"]["good"]["degraded"] is False


def test_a_sink_whose_submit_itself_raises_is_isolated_and_then_detached(fanout):
    class Explodes:
        name = "explodes"

        def submit(self, frame, info, effects=None):
            raise RuntimeError("boom")

        def close(self):
            pass

    seen = []
    fanout.attach(Explodes())
    fanout.attach(CallbackSink(lambda f, i: seen.append(i.n), name="good"))
    fanout.max_failures = 2
    fanout.submit(frame_with(1), info(0))
    assert fanout.get("explodes") is not None
    assert wait_for(lambda: seen == [0])
    fanout.submit(frame_with(1), info(1))
    assert fanout.get("explodes") is None
    assert fanout.detached["explodes"] == "RuntimeError: boom"
    assert wait_for(lambda: seen == [0, 1])


def test_attaching_and_detaching_mid_stream_loses_no_frames_for_the_others(fanout):
    a, b, c = [], [], []
    sa = CallbackSink(lambda f, i: a.append(i.n), name="a")
    sb = CallbackSink(lambda f, i: b.append(i.n), name="b")
    sc = CallbackSink(lambda f, i: c.append(i.n), name="c")
    fanout.attach(sa)
    for n in range(20):
        if n == 5:
            fanout.attach(sb)
        if n == 10:
            fanout.detach(sb)
            fanout.attach(sc)
        if n == 15:
            fanout.detach("c")
        fanout.submit(frame_with(1), info(n))
        assert wait_for(lambda: bool(a) and a[-1] == n)
        for lst, sink in ((b, sb), (c, sc)):
            if fanout.get(sink.name) is not None:
                assert wait_for(lambda: bool(lst) and lst[-1] == n)
    assert a == list(range(20))
    assert b == list(range(5, 10))
    assert c == list(range(10, 15))


def test_observer_frames_are_non_writeable(fanout):
    captured = []
    fanout.attach(CallbackSink(lambda f, i: captured.append(f), name="cap"))
    frame = frame_with(3)
    assert not frame.frozen
    fanout.submit(frame, info(0))
    assert wait_for(lambda: len(captured) == 1)
    assert captured[0] is frame and frame.frozen
    with pytest.raises(ValueError):
        captured[0].data[0, 0] = 0


def test_duplicate_names_are_rejected_and_null_sink_counts(fanout):
    fanout.attach(NullSink("n"))
    with pytest.raises(ValueError, match="already attached"):
        fanout.attach(NullSink("n"))
    fanout.submit(frame_with(0), info(0))
    assert fanout.get("n").frames == 1
    assert isinstance(fanout.get("n"), Sink)


def test_mailbox_semantics():
    m = Mailbox()
    assert m.take(timeout=0.01) is None
    m.put(1)
    m.put(2)
    assert m.replaced == 1 and m.pending == 1 and m.peek() == 2
    assert m.take() == 2 and m.pending == 0
    m.close()
    assert m.take() is None and m.closed


# ---- PreviewSink -------------------------------------------------------------------------


def test_preview_with_one_stalled_subscriber_still_delivers_to_the_others_at_full_rate(fanout):
    pv = PreviewSink()
    fanout.attach(pv)
    release = threading.Event()
    stalled = pv.subscribe(lambda rec: release.wait(), "full", name="stalled")
    fine = []
    healthy = pv.subscribe(fine.append, "full", name="fine")
    for n in range(30):
        fanout.submit(frame_with(n), info(n))
        assert wait_for(lambda: len(fine) == n + 1)
    assert [decode_preview(r).frame_no for r in fine] == list(range(30))
    assert stalled.dropped >= 27  # everything after the first two was overwritten
    release.set()
    assert wait_for(lambda: stalled.sent >= 2)
    assert healthy.sent == 30


def test_a_subscriber_attaching_mid_stream_receives_the_next_frame_and_no_backlog(fanout):
    pv = PreviewSink()
    fanout.attach(pv)
    early = []
    pv.subscribe(early.append, "tiles", name="early")
    for n in range(10):
        fanout.submit(frame_with(n), info(n))
        assert wait_for(lambda: len(early) == n + 1)
    late = []
    pv.subscribe(late.append, "tiles", name="late")
    fanout.submit(frame_with(10), info(10))
    assert wait_for(lambda: len(late) == 1)
    assert decode_preview(late[0]).frame_no == 10
    assert len(late) == 1


def test_frame_no_increments_by_more_than_one_across_rate_limited_and_dropped_frames(fanout):
    pv = PreviewSink()
    fanout.attach(pv)
    slow = []
    pv.subscribe(slow.append, "tiles", max_fps=10, name="10fps")
    for n in range(30):
        fanout.submit(frame_with(1), info(n))
        pv.mailbox.take  # noqa: B018 - no-op; the worker consumes
        assert wait_for(lambda: pv.frames_handled == n + 1)
    assert wait_for(lambda: len(slow) == 10)
    assert [decode_preview(r).frame_no for r in slow] == list(range(0, 30, 3))
    # a dropped frame at the clock shows as a gap too
    fanout.submit(frame_with(1), info(45))
    assert wait_for(lambda: len(slow) == 11)
    assert decode_preview(slow[-1]).frame_no == 45


def test_a_subscriber_whose_send_keeps_failing_is_dropped_with_a_reason(fanout):
    pv = PreviewSink()
    fanout.attach(pv)
    bad = pv.subscribe(lambda rec: (_ for _ in ()).throw(ConnectionError("gone")), "full", name="bad", max_failures=2)
    good = []
    pv.subscribe(good.append, "full", name="good")
    for n in range(4):
        fanout.submit(frame_with(1), info(n))
        assert wait_for(lambda: len(good) == n + 1)
    assert wait_for(lambda: bad not in pv.subscribers)
    assert "ConnectionError" in pv.dropped_subscribers["bad"]
    assert [s.name for s in pv.subscribers] == ["good"]


def test_unsubscribe_and_bad_arguments(fanout):
    pv = PreviewSink()
    fanout.attach(pv)
    sub = pv.subscribe(lambda r: None)
    pv.unsubscribe(sub)
    assert pv.subscribers == () and sub.mailbox.closed
    with pytest.raises(ValueError):
        pv.subscribe(lambda r: None, "jpeg")
    with pytest.raises(ValueError):
        pv.subscribe(lambda r: None, max_fps=0)


# ---- the wire format --------------------------------------------------------------------


def test_preview_wire_format_full_and_tiles():
    p = PixelFrame.black()
    p.tile(0)[:] = (255, 0, 0)
    rec = encode_preview(p, 7, "full")
    assert len(rec) == HEADER_SIZE + 11520
    assert rec[:7] == bytes([1, 1, 0, 0, 0, 0, 7])
    assert rec[7:10] == b"\xff\x00\x00"
    back = decode_preview(rec)
    assert (back.frame_no, back.fmt, back.tile_source) == (7, "full", False)
    assert back.frame == p

    rec = encode_preview(p, 0xFFFFFFFF + 3, "tiles")
    assert len(rec) == HEADER_SIZE + 192
    back = decode_preview(rec)
    assert back.frame_no == 2  # wraps like a uint32
    assert back.frame == p.to_tiles()
    assert back.frame[0, 0].tolist() == [255, 0, 0]  # uniform tile reduces exactly


def test_preview_wire_format_flags_a_tile_frame_source():
    t = TileFrame.black()
    t[1, 2] = (9, 8, 7)
    rec = encode_preview(t, 1, "full")
    assert rec[2] == 0x01
    back = decode_preview(rec)
    assert back.tile_source and back.frame == t.to_pixels()
    rec = encode_preview(t, 1, "tiles")
    assert decode_preview(rec).frame == t


def test_preview_decode_rejects_garbage():
    with pytest.raises(ValueError):
        decode_preview(b"\x01\x01")
    with pytest.raises(ValueError, match="version"):
        decode_preview(bytes([2, 1, 0, 0, 0, 0, 0]) + bytes(192))
    with pytest.raises(ValueError, match="format"):
        decode_preview(bytes([1, 9, 0, 0, 0, 0, 0]) + bytes(192))


# ---- RecorderSink ----------------------------------------------------------------------------


def test_recorder_round_trips_frames_with_their_time(tmp_path, fanout):
    rec = RecorderSink(tmp_path / "take1.df2rec", "full")
    fanout.attach(rec)
    frames = [frame_with(n * 20) for n in range(5)]
    for n, f in enumerate(frames):
        fanout.submit(f, info(n))
        assert wait_for(lambda: rec.records == n + 1)
    fanout.detach(rec)  # closes the file
    played = list(read_recording(tmp_path / "take1.df2rec"))
    assert [r.frame_no for r in played] == [0, 1, 2, 3, 4]
    assert [r.t for r in played] == pytest.approx([n / FPS for n in range(5)])
    assert all(r.frame == f for r, f in zip(played, frames))


def test_recorder_tiles_format_and_bad_file(tmp_path, fanout):
    rec = RecorderSink(tmp_path / "t.df2rec", "tiles")
    fanout.attach(rec)
    fanout.submit(TileFrame.black(), info(3))
    assert wait_for(lambda: rec.records == 1)
    fanout.detach(rec)
    [only] = read_recording(tmp_path / "t.df2rec")
    assert isinstance(only.frame, TileFrame) and only.frame_no == 3
    (tmp_path / "junk").write_bytes(b"nope")
    with pytest.raises(ValueError, match="not a df2 recording"):
        list(read_recording(tmp_path / "junk"))
    with pytest.raises(ValueError):
        RecorderSink(tmp_path / "x", "png")


# ---- HardwareSink -----------------------------------------------------------------------------


def test_hardware_sink_sends_every_row_then_exactly_one_latch_in_row_order():
    floor = MagicMock()
    hw = HardwareSink(floor, FrameEncoder(gamma=1.0))
    t = TileFrame.black()
    t[2, 5] = (1, 2, 3)
    hw.submit(t, info(0))
    floor.send_rows.assert_called_once()
    floor.latch.assert_not_called()
    payloads = floor.send_rows.call_args.args[0]
    assert len(payloads) == 8
    assert parse_send_data(payloads[2])[5] == (TileCmd.SET_COLOR, bytes((1, 2, 3)))
    assert all(len(p) == 32 for p in payloads)
    hw.latch()
    floor.latch.assert_called_once()
    assert hw.frames == 1 and hw.healthy
    assert [c[0] for c in floor.method_calls] == ["send_rows", "latch"]


def test_hardware_sink_passes_effects_through_to_the_encoder():
    floor = MagicMock()
    hw = HardwareSink(floor, FrameEncoder(gamma=1.0))
    hw.submit(TileFrame.black(), info(0), {9: Effect(FADE, (230, 0, 0, 0))})
    payloads = floor.send_rows.call_args.args[0]
    assert parse_send_data(payloads[1])[1] == (TileCmd.SET_EFFECT, bytes((FADE, 230, 0, 0, 0)))


def test_a_serial_exception_in_hardware_sink_is_counted_not_propagated():
    floor = MagicMock()
    floor.send_rows.side_effect = OSError("write failed")
    hw = HardwareSink(floor, max_failures=3)
    for n in range(3):
        hw.submit(TileFrame.black(), info(n))  # does not raise
        hw.latch()
    assert hw.failures == 3 and hw.consecutive_failures == 3
    assert not hw.healthy
    assert "OSError: write failed" in hw.last_error
    floor.send_rows.side_effect = None
    hw.submit(TileFrame.black(), info(3))
    hw.latch()
    assert hw.healthy and hw.consecutive_failures == 0 and hw.frames == 1
    floor.latch.side_effect = OSError("latch failed")
    hw.latch()
    assert hw.failures == 4 and "latch" in hw.last_error


def test_hardware_sink_holds_brightness_and_marks_the_clock():
    floor = MagicMock()
    clock = FrameClock(now=lambda: 0.0, sleep=lambda s: None)
    hw = HardwareSink(floor, clock=clock)
    hw.brightness = 100
    assert hw.encoder.brightness == 100
    clock.mark("start")
    hw.submit(TileFrame.black(), info(0))
    assert set(clock.telemetry().phases) == {"encode", "wire"}


def test_fanout_latch_reaches_the_hardware_sink_and_orders_it_first(fanout):
    floor = MagicMock()
    fanout.attach(NullSink("first"))
    hw = HardwareSink(floor)
    fanout.attach(hw)
    assert [s.name for s in fanout.sinks] == ["hardware", "first"]
    fanout.submit(TileFrame.black(), info(0))
    fanout.latch()
    floor.latch.assert_called_once()
    assert fanout.state()["sinks"]["hardware"]["healthy"] is True
