"""`FrameClock`: a drift-free frame scheduler with timing telemetry.

Keeping an animation at 30 FPS sounds trivial and is not: a `sleep(1/30)`
loop drifts, and on a Pi running CPython the things that actually eat the
frame budget are scheduler wake-up jitter and garbage-collection pauses.

    clock = FrameClock(fps=30.0)
    for tick in clock.run(latch=floor.latch):     # tick: FrameInfo(n, t, deadline)
        frame = run.render(t=tick.t)
        clock.mark("render")
        payloads = enc.encode(frame.frame, frame.effects)
        clock.mark("encode")
        for row, payload in enumerate(payloads):
            floor.send_data(row, payload)
        clock.mark("wire")
        # the clock now waits for tick.deadline, calls latch(), and yields
        # the next tick

The deadline is the LATCH moment. `LATCH` is a broadcast that lights all
64 tiles at once, so jitter on it is jitter everyone on the floor can see;
`Floor.broadcast()` keeps cross-chain skew in the microseconds and it
would be a waste to feed it from a +/-5 ms clock. So the loop is
pipelined: the body PREPARES frame n (render, encode, SEND_DATA) and the
clock fires the latch at exactly deadline n, then hands over the next
tick. `tick.t` is therefore the moment frame n will be SEEN.

Scheduled deadlines, not relative sleeps:

    deadline_n = anchor + n / fps      # yes
    sleep(1 / fps - elapsed)           # no

Frames are scheduled against a running deadline sequence, which buys three
things. `tick.t` is the frame's deadline rather than the wall time it
happened to render at, so animations get uniform time steps and a frame
arriving 3 ms late still advances motion by exactly 1/30 s. Lateness is
measurable, because "how late" only means something against a deadline.
And overrun handling is defined rather than emergent:

- A frame that is ready a little late (within `late_tolerance`, default
  half a period) is latched immediately; the lateness is recorded as
  jitter and nothing is dropped.
- A frame later than that has MISSED its deadline. The missed deadlines
  are dropped, counted, and the frame is latched on the next real one -
  never a burst of catch-up frames, which looks worse than the drop and
  can cascade. `tick.n` and `tick.t` skip ahead accordingly.
- A stall longer than `reanchor_after` periods (a GC pause, a suspended
  process, a breakpoint) is a re-anchor: the deadline sequence restarts
  from now and the lost time is abandoned, never repaid. Same principle,
  applied where counting individual missed deadlines is meaningless.

Hybrid sleep. `time.sleep()` on Linux overshoots by ~50 us to 1 ms
depending on load and the GIL can add more, so the clock sleeps until
`deadline - spin_margin` (default 1.5 ms) and busy-spins on
`perf_counter()` for the rest. The margin is a floor, not a fixed value:
the clock measures every sleep's overshoot and spins from a little above
the recent p95 when that is larger (macOS coalesces timers and overshoots
by 2-5 ms; a loaded Pi drifts too), capped at `max_spin_margin` so a
pathological host cannot turn the loop into a pure spin. `spin_margin=0`
disables spinning altogether for development, where latch jitter does
not matter.

GC and scheduling, all off by default and all measurable:

- `gc_freeze=True` calls `gc.freeze()` when the loop starts, so nothing
  allocated at import is ever rescanned.
- `manage_gc=True` disables automatic collection during playback and runs
  `gc.collect(0)` in the idle slack after the frame is on the wire, when
  there is time doing nothing, instead of wherever the allocator decides.
- `realtime=True` tries SCHED_FIFO and falls back to a negative nice; both
  need privileges and are a foot-gun on a shared machine, so this sits
  behind a flag. `clock.scheduling` says what was actually applied.

Telemetry is a rolling window (default 300 frames, 10 s) of per-phase
timings, because "the floor looks stuttery" has to resolve to WHICH
phase: whatever the body `mark()`s (render, encode, wire), plus
`slack_ms` (deadline minus the time the body finished), `jitter_ms`
(latch time minus deadline), and running counts of frames, dropped
deadlines and re-anchors. `telemetry()` returns a snapshot with p50 / p95
/ max of each. This is how an animation gets found out as too expensive
before it becomes visible.

`now` and `sleep` are injectable so the whole thing is testable against a
fake clock without a single real sleep.
"""

from __future__ import annotations

import gc
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np


@dataclass(frozen=True)
class FrameInfo:
    """One tick of the clock. `n` is the deadline index since the loop
    started (dropped deadlines advance it too), `t = n / fps` is the
    animation time this frame is for, `deadline` is its latch moment on
    the `now()` timebase, and `dropped` is how many deadlines were missed
    immediately before this one."""

    n: int
    t: float
    deadline: float
    dropped: int = 0


@dataclass(frozen=True)
class Percentiles:
    p50: float
    p95: float
    max: float
    count: int

    @classmethod
    def of(cls, values) -> Percentiles:
        if len(values) == 0:
            return cls(math.nan, math.nan, math.nan, 0)
        arr = np.fromiter(values, dtype=np.float64, count=len(values))
        p50, p95 = np.percentile(arr, [50, 95])
        return cls(float(p50), float(p95), float(arr.max()), int(arr.size))


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Percentiles over the rolling window, in milliseconds, plus the
    running counters. `phases` holds whatever the loop body marked."""

    fps: float
    window: int
    frames: int
    dropped: int
    reanchors: int
    slack_ms: Percentiles
    jitter_ms: Percentiles
    sleep_overshoot_ms: Percentiles
    spin_margin_ms: float
    phases: dict[str, Percentiles] = field(default_factory=dict)


class FrameClock:
    def __init__(
        self,
        fps: float = 30.0,
        spin_margin: float = 0.0015,
        *,
        max_spin_margin: float = 0.008,
        late_tolerance: float | None = None,
        reanchor_after: float = 4.0,
        window: int = 300,
        gc_freeze: bool = False,
        manage_gc: bool = False,
        gc_budget: float = 0.004,
        realtime: bool = False,
        now: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if spin_margin < 0:
            raise ValueError(f"spin_margin must be non-negative, got {spin_margin}")
        if reanchor_after < 1:
            raise ValueError(f"reanchor_after must be at least 1 period, got {reanchor_after}")
        if window < 1:
            raise ValueError(f"window must be at least 1 frame, got {window}")
        self.fps = float(fps)
        self.period = 1.0 / self.fps
        self.spin_margin = float(spin_margin)
        self.max_spin_margin = max(self.spin_margin, float(max_spin_margin))
        self.late_tolerance = self.period / 2 if late_tolerance is None else float(late_tolerance)
        if not 0 <= self.late_tolerance < self.period:
            raise ValueError("late_tolerance must be within one period")
        self.reanchor_after = float(reanchor_after)
        self.window = int(window)
        self.gc_freeze = gc_freeze
        self.manage_gc = manage_gc
        self.gc_budget = float(gc_budget)
        self.realtime = realtime
        self.scheduling: str = "default"
        self._now = now
        self._sleep = sleep

        self.frames = 0
        self.dropped = 0
        self.reanchors = 0
        self._running = False
        self._last_mark: float | None = None
        self._slack: deque[float] = deque(maxlen=self.window)
        self._jitter: deque[float] = deque(maxlen=self.window)
        self._phases: dict[str, deque[float]] = {}
        self._overshoot: deque[float] = deque(maxlen=self.window)

    # ---- the loop -----------------------------------------------------------------

    def run(self, latch: Callable[[], None] | None = None) -> Iterator[FrameInfo]:
        """Yield a tick per frame. Between one tick and the next, the body
        prepares that frame; when it hands control back, the clock waits
        for the tick's deadline, calls `latch()` (if given), and yields the
        next tick. Ends when `stop()` has been called or the consumer
        stops iterating."""
        self._running = True
        self._setup()
        try:
            # Frame 0 gets a full period to prepare, like every other frame.
            anchor = self._now() + self.period
            n_anchor = 0
            n = 0
            dropped_before = 0
            while self._running:
                deadline = anchor + (n - n_anchor) * self.period
                self._last_mark = self._now()
                yield FrameInfo(n, n * self.period, deadline, dropped_before)
                if not self._running:
                    return

                ready = self._now()
                late = ready - deadline
                self._slack.append((deadline - ready) * 1000.0)

                if late >= self.late_tolerance:
                    missed = int((late - self.late_tolerance) // self.period) + 1
                    self.dropped += missed
                    dropped_before = missed
                    n += missed
                    if late >= self.reanchor_after * self.period:
                        # A stall, not an overrun: restart the sequence from now.
                        self.reanchors += 1
                        anchor = ready
                        n_anchor = n
                        deadline = ready
                    else:
                        deadline = anchor + (n - n_anchor) * self.period
                else:
                    dropped_before = 0
                    self._collect_in_slack(deadline)

                self._wait_until(deadline)
                if latch is not None:
                    latch()
                self._jitter.append((self._now() - deadline) * 1000.0)
                self.frames += 1
                n += 1
        finally:
            self._running = False
            self._teardown()

    def stop(self) -> None:
        """Ask the loop to end after the current frame is prepared (it is
        not latched). Safe to call from another thread."""
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def mark(self, phase: str) -> None:
        """Record the time since the previous mark (or since the tick was
        yielded) against `phase`."""
        now = self._now()
        if self._last_mark is not None:
            self._phases.setdefault(phase, deque(maxlen=self.window)).append(
                (now - self._last_mark) * 1000.0
            )
        self._last_mark = now

    # ---- telemetry ------------------------------------------------------------------

    def telemetry(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            fps=self.fps,
            window=self.window,
            frames=self.frames,
            dropped=self.dropped,
            reanchors=self.reanchors,
            slack_ms=Percentiles.of(self._slack),
            jitter_ms=Percentiles.of(self._jitter),
            sleep_overshoot_ms=Percentiles.of([v * 1000.0 for v in self._overshoot]),
            spin_margin_ms=self.effective_spin_margin * 1000.0,
            phases={name: Percentiles.of(values) for name, values in self._phases.items()},
        )

    # ---- internals ------------------------------------------------------------------

    def _wait_until(self, deadline: float) -> None:
        """Hybrid sleep: coarse sleep to `deadline - margin`, then spin."""
        if self.spin_margin == 0.0:
            while (remaining := deadline - self._now()) > 0:
                self._sleep(remaining)
            return
        margin = self.effective_spin_margin
        while True:
            remaining = deadline - self._now()
            if remaining <= 0:
                return
            if remaining > margin:
                requested = remaining - margin
                before = self._now()
                self._sleep(requested)
                self._overshoot.append(self._now() - before - requested)
            # else: spin on now() until the deadline passes

    @property
    def effective_spin_margin(self) -> float:
        """The margin the next wait will spin from: `spin_margin`, or a
        little above the recent p95 sleep overshoot if that is larger,
        capped at `max_spin_margin`."""
        if self.spin_margin == 0.0 or not self._overshoot:
            return self.spin_margin
        observed = float(np.percentile(np.fromiter(self._overshoot, dtype=np.float64), 95))
        return min(self.max_spin_margin, max(self.spin_margin, observed + 0.0003))

    def _collect_in_slack(self, deadline: float) -> None:
        if self.manage_gc and deadline - self._now() > self.gc_budget:
            gc.collect(0)

    def _setup(self) -> None:
        if self.gc_freeze:
            gc.freeze()
        if self.manage_gc:
            self._gc_was_enabled = gc.isenabled()
            gc.disable()
        if self.realtime:
            self.scheduling = _raise_priority()

    def _teardown(self) -> None:
        if self.manage_gc and getattr(self, "_gc_was_enabled", False):
            gc.enable()


def _raise_priority() -> str:
    """Try SCHED_FIFO, then a negative nice. Returns what was applied."""
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(10))  # type: ignore[attr-defined]
        return "SCHED_FIFO"
    except (AttributeError, PermissionError, OSError):
        pass
    try:
        os.nice(-10)
        return "nice -10"
    except (PermissionError, OSError):
        return "default (raising priority failed: insufficient privileges)"
