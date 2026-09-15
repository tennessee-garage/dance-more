import os
import time
from collections import deque

import numpy as np
import pytest

from df2_pi.engine import FrameClock, FrameInfo, Percentiles

FPS = 30.0
PERIOD = 1.0 / FPS


class FakeTime:
    """An injectable clock. `now()` creeps forward 1 us per call so a spin
    loop terminates; `sleep()` advances exactly the requested time plus a
    fixed overshoot, like the real thing; `work()` is the loop body
    burning time."""

    def __init__(self, overshoot: float = 0.0002) -> None:
        self.t = 1000.0
        self.overshoot = overshoot
        self.sleeps: list[float] = []

    def now(self) -> float:
        self.t += 1e-6
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.sleeps.append(seconds)
        self.t += seconds + self.overshoot

    def work(self, seconds: float) -> None:
        self.t += seconds


def make_clock(fake: FakeTime, **kwargs) -> FrameClock:
    kwargs.setdefault("fps", FPS)
    return FrameClock(now=fake.now, sleep=fake.sleep, **kwargs)


def drive(clock: FrameClock, fake: FakeTime, frames: int, work=lambda tick: 0.010):
    """Run `frames` ticks with `work(tick)` seconds of body time each.
    Returns (ticks, latch times)."""
    ticks: list[FrameInfo] = []
    latches: list[float] = []
    for tick in clock.run(latch=lambda: latches.append(fake.t)):
        ticks.append(tick)
        fake.work(work(tick))
        clock.mark("render")
        if len(ticks) == frames:
            clock.stop()
    return ticks, latches


# ---- steady state -------------------------------------------------------------------


def test_deadlines_are_evenly_spaced_at_exactly_one_period_while_keeping_up():
    fake = FakeTime()
    clock = make_clock(fake)
    ticks, latches = drive(clock, fake, 50)
    deadlines = np.array([t.deadline for t in ticks])
    np.testing.assert_allclose(np.diff(deadlines), PERIOD, atol=1e-9)
    np.testing.assert_allclose(np.diff(latches), PERIOD, atol=2e-6)  # spin lands within the fake's epsilon
    assert [t.n for t in ticks] == list(range(50))
    assert clock.dropped == 0 and clock.reanchors == 0 and clock.frames == 49


def test_t_advances_by_exactly_one_period_regardless_of_how_late_a_frame_rendered():
    fake = FakeTime()
    clock = make_clock(fake)
    rng = np.random.default_rng(64)
    # body time varies between 2 ms and 40 ms: some frames are late (up to
    # ~7 ms past the deadline) but none by more than the half-period tolerance
    ticks, latches = drive(clock, fake, 40, work=lambda t: float(rng.uniform(0.002, 0.040)))
    ts = np.array([t.t for t in ticks])
    np.testing.assert_allclose(np.diff(ts), PERIOD, atol=1e-12)
    assert clock.dropped == 0
    jitter = clock.telemetry().jitter_ms
    assert jitter.max > 1.0  # some frames really were late...
    assert jitter.max < PERIOD * 500  # ...but within tolerance


def test_frame_0_gets_a_full_period_to_prepare():
    fake = FakeTime()
    clock = make_clock(fake)
    start = fake.t
    ticks, _ = drive(clock, fake, 2, work=lambda t: 0.025)
    assert ticks[0].deadline == pytest.approx(start + PERIOD, abs=1e-5)
    assert clock.dropped == 0


def test_latch_fires_at_the_deadline_with_the_hybrid_sleep():
    fake = FakeTime(overshoot=0.0008)  # a bad sleep(): 0.8 ms overshoot
    clock = make_clock(fake, spin_margin=0.0015)
    ticks, latches = drive(clock, fake, 20)
    for tick, latched in zip(ticks, latches):
        assert 0 <= latched - tick.deadline < 5e-6  # the spin absorbs the overshoot
    # each frame slept once, to spin_margin before the deadline
    assert len(fake.sleeps) == 19
    assert all(abs((PERIOD - 0.010 - 0.0015) - s) < 1e-4 for s in fake.sleeps)


def test_spin_margin_adapts_to_a_host_that_overshoots_more_than_the_margin():
    fake = FakeTime(overshoot=0.003)  # macOS-like: 3 ms past every sleep
    clock = make_clock(fake, spin_margin=0.0015)
    ticks, latches = drive(clock, fake, 20)
    late = [latched - tick.deadline for tick, latched in zip(ticks, latches)]
    assert late[0] == pytest.approx(0.0015, abs=1e-5)  # first frame: fixed margin, 1.5 ms late
    assert all(x < 5e-6 for x in late[2:])  # then the margin has grown past the overshoot
    assert clock.effective_spin_margin == pytest.approx(0.0033, abs=1e-5)
    snap = clock.telemetry()
    assert snap.sleep_overshoot_ms.p95 == pytest.approx(3.0, abs=0.01)
    assert snap.spin_margin_ms == pytest.approx(3.3, abs=0.01)
    # and it is capped
    capped = make_clock(FakeTime(overshoot=0.050), spin_margin=0.0015, max_spin_margin=0.008)
    drive(capped, capped._now.__self__, 5)
    assert capped.effective_spin_margin == 0.008


def test_spin_margin_zero_relies_on_sleep_alone():
    fake = FakeTime(overshoot=0.0008)
    clock = make_clock(fake, spin_margin=0.0)
    ticks, latches = drive(clock, fake, 20)
    assert all(abs((latched - tick.deadline) - 0.0008) < 5e-6 for tick, latched in zip(ticks, latches))


# ---- overruns ------------------------------------------------------------------------


def test_a_frame_overrunning_by_two_and_a_half_periods_skips_two_deadlines():
    fake = FakeTime()
    clock = make_clock(fake)
    ticks, latches = drive(clock, fake, 12, work=lambda t: 2.5 * PERIOD if t.n == 4 else 0.010)
    ns = [t.n for t in ticks]
    assert ns == [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]  # 5 and 6 dropped
    assert [t.dropped for t in ticks] == [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    assert clock.dropped == 2
    # frame 4 latched on the next real deadline (d4 + 2 periods), not in a
    # burst, and the sequence stays on the original grid afterwards
    gaps = np.diff(latches)
    np.testing.assert_allclose(gaps[3], 3 * PERIOD, atol=1e-5)
    np.testing.assert_allclose(np.delete(gaps, 3), PERIOD, atol=1e-5)
    np.testing.assert_allclose([t.t for t in ticks], np.array(ns) * PERIOD)


def test_a_little_lateness_is_jitter_not_a_drop():
    fake = FakeTime()
    clock = make_clock(fake)
    ticks, latches = drive(clock, fake, 10, work=lambda t: PERIOD + 0.004 if t.n == 3 else 0.010)
    assert clock.dropped == 0
    assert [t.n for t in ticks] == list(range(10))
    assert latches[3] - ticks[3].deadline == pytest.approx(0.004, abs=1e-5)
    assert latches[4] - ticks[4].deadline < 5e-6  # back on the grid next frame


def test_dropped_counts_exactly_the_skipped_deadlines():
    fake = FakeTime()
    clock = make_clock(fake, reanchor_after=10)
    overruns = {2: 1.6, 5: 3.2, 9: 2.0}  # in periods of body time
    ticks, _ = drive(clock, fake, 15, work=lambda t: overruns.get(t.n, 0.3) * PERIOD)
    # body of 1.6 periods -> 0.6 late -> 1 missed; 3.2 -> 2.2 late -> 2 missed
    # (2.2 - 0.5 = 1.7 -> floor 1 + 1); 2.0 -> 1.0 late -> 1 missed
    assert [t.dropped for t in ticks if t.dropped] == [1, 2, 1]
    assert clock.dropped == 4
    assert ticks[-1].n == 14 + 4
    assert clock.reanchors == 0


# ---- re-anchoring -----------------------------------------------------------------------


def test_a_long_stall_reanchors_instead_of_bursting():
    fake = FakeTime()
    clock = make_clock(fake, reanchor_after=4)
    stall = 0.5  # 15 periods
    ticks, latches = drive(clock, fake, 20, work=lambda t: stall if t.n == 6 else 0.010)
    assert clock.reanchors == 1
    stalled_at = 6
    # no burst: the stalled frame latched immediately when it was ready
    # (its body began right after the previous latch and took `stall`)
    gaps = np.diff(latches)
    assert gaps[stalled_at - 1] == pytest.approx(stall, abs=1e-4)
    # the new grid starts at the stalled frame's latch and is even after it
    np.testing.assert_allclose(gaps[stalled_at:], PERIOD, atol=1e-5)
    after = ticks[stalled_at + 1]
    assert after.deadline == pytest.approx(latches[stalled_at] + PERIOD, abs=1e-5)
    # lost time is abandoned: n jumps by the missed deadlines, t follows n
    late = stall - PERIOD  # ready time minus the stalled frame's own deadline
    assert after.dropped == int((late - PERIOD / 2) // PERIOD) + 1 == 14
    assert after.n == stalled_at + after.dropped + 1
    assert after.t == pytest.approx(after.n * PERIOD)
    assert clock.dropped == after.dropped


def test_stop_ends_the_loop_without_a_final_latch():
    fake = FakeTime()
    clock = make_clock(fake)
    latches = []
    seen = 0
    for tick in clock.run(latch=lambda: latches.append(fake.t)):
        seen += 1
        if seen == 3:
            clock.stop()
    assert seen == 3 and len(latches) == 2
    assert not clock.running


def test_run_without_a_latch_callable():
    fake = FakeTime()
    clock = make_clock(fake)
    ticks, _ = drive(clock, fake, 3)
    assert len(ticks) == 3 and clock.frames == 2


# ---- telemetry ------------------------------------------------------------------------


def test_percentiles_match_hand_computed_values():
    p = Percentiles.of(deque(range(100)))  # 0..99
    assert (p.p50, p.p95, p.max, p.count) == (49.5, 94.05, 99.0, 100)
    p = Percentiles.of([3.0])
    assert (p.p50, p.p95, p.max, p.count) == (3.0, 3.0, 3.0, 1)
    empty = Percentiles.of([])
    assert empty.count == 0 and np.isnan(empty.p50)


def test_telemetry_reports_phases_slack_and_jitter():
    fake = FakeTime()
    clock = make_clock(fake, window=8)
    for tick in clock.run(latch=lambda: None):
        fake.work(0.010)
        clock.mark("render")
        fake.work(0.002)
        clock.mark("encode")
        fake.work(0.005)
        clock.mark("wire")
        if tick.n == 19:
            clock.stop()
    snap = clock.telemetry()
    assert snap.fps == FPS and snap.window == 8
    assert snap.frames == 19 and snap.dropped == 0 and snap.reanchors == 0
    assert set(snap.phases) == {"render", "encode", "wire"}
    assert snap.phases["render"].p50 == pytest.approx(10.0, abs=0.01)
    assert snap.phases["encode"].p50 == pytest.approx(2.0, abs=0.01)
    assert snap.phases["wire"].p50 == pytest.approx(5.0, abs=0.01)
    assert snap.phases["render"].count == 8  # bounded by the window, not 20
    assert snap.slack_ms.p50 == pytest.approx((PERIOD - 0.017) * 1000, abs=0.05)
    assert 0 <= snap.jitter_ms.p95 < 0.01


def test_the_rolling_window_bounds_memory():
    fake = FakeTime()
    clock = make_clock(fake, window=300)
    for i in range(1_000_000):
        clock._slack.append(float(i))
        clock._jitter.append(float(i))
    assert len(clock._slack) == 300 and len(clock._jitter) == 300
    assert clock.telemetry().slack_ms.max == 999_999.0
    clock._last_mark = fake.now()
    for _ in range(100_000):
        clock.mark("render")
    assert len(clock._phases["render"]) == 300


def test_gc_management_is_off_by_default_and_restores_state():
    import gc

    fake = FakeTime()
    clock = make_clock(fake)
    assert not clock.manage_gc and not clock.gc_freeze and not clock.realtime
    was_enabled = gc.isenabled()
    managed = make_clock(fake, manage_gc=True)
    for tick in managed.run():
        assert not gc.isenabled()
        if tick.n == 2:
            managed.stop()
    assert gc.isenabled() == was_enabled


def test_constructor_validation():
    with pytest.raises(ValueError):
        FrameClock(fps=0)
    with pytest.raises(ValueError):
        FrameClock(spin_margin=-1)
    with pytest.raises(ValueError):
        FrameClock(reanchor_after=0.5)
    with pytest.raises(ValueError):
        FrameClock(late_tolerance=1.0)
    with pytest.raises(ValueError):
        FrameClock(window=0)


# ---- the real clock ------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("DF2_SLOW_TESTS"), reason="set DF2_SLOW_TESTS=1 to run")
def test_real_clock_p95_jitter_under_2ms():
    clock = FrameClock(fps=FPS, spin_margin=0.0015)
    latched = []
    for tick in clock.run(latch=lambda: latched.append(time.perf_counter())):
        time.sleep(0.005)  # a cheap "render"
        clock.mark("render")
        if tick.n == 150:  # 5 s
            clock.stop()
    snap = clock.telemetry()
    assert snap.dropped == 0
    assert snap.jitter_ms.p95 < 2.0, snap
