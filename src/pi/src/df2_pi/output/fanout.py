"""`FanOut`: one frame in, every sink out.

Holds the sink list; sinks attach and detach at runtime (a preview client
connecting mid-playback, a recorder starting). On each `submit()`:

- the frame is frozen, so observers get a read-only view
- every sink's `submit()` is called in order - the hardware sink first if
  it is attached, since it is the one that costs time and the observers
  only mailbox
- a sink raising is caught and logged and the frame still reaches the
  others; `max_failures` consecutive raises detaches it
- a `ThreadedSink` that degraded on its own thread is detached too

`detached` records why each sink went, for the runner's state snapshot,
and `state()` is that snapshot. `latch()` forwards to every attached sink
that has one (the hardware sink), so the frame clock gets a single
callable whatever the sink mix.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable, Mapping

from df2_pi.effects import Effect
from df2_pi.engine.clock import FrameInfo
from df2_pi.output.sink import DEFAULT_MAX_FAILURES, Sink
from df2_pi.pixels import Frame

log = logging.getLogger(__name__)


class FanOut:
    def __init__(self, sinks: Iterable[Sink] = (), *, max_failures: int = DEFAULT_MAX_FAILURES) -> None:
        self._lock = threading.Lock()
        self._sinks: list[Sink] = []
        self._consecutive: dict[str, int] = {}
        self.max_failures = max_failures
        self.detached: dict[str, str] = {}
        self.frames = 0
        for sink in sinks:
            self.attach(sink)

    # ---- membership -----------------------------------------------------------------------

    @property
    def sinks(self) -> tuple[Sink, ...]:
        with self._lock:
            return tuple(self._sinks)

    def attach(self, sink: Sink) -> None:
        with self._lock:
            if any(s.name == sink.name for s in self._sinks):
                raise ValueError(f"a sink named {sink.name!r} is already attached")
            # Hardware first: it is synchronous and owns the deadline;
            # everything after it just drops into a mailbox.
            if hasattr(sink, "latch"):
                self._sinks.insert(0, sink)
            else:
                self._sinks.append(sink)
            self._consecutive[sink.name] = 0
            self.detached.pop(sink.name, None)

    def detach(self, sink: Sink | str, *, reason: str | None = None, close: bool = True) -> Sink | None:
        name = sink if isinstance(sink, str) else sink.name
        with self._lock:
            found = next((s for s in self._sinks if s.name == name), None)
            if found is None:
                return None
            self._sinks.remove(found)
            self._consecutive.pop(name, None)
            if reason is not None:
                self.detached[name] = reason
        if close:
            try:
                found.close()
            except Exception:
                log.exception("sink %s failed while closing", name)
        return found

    def get(self, name: str) -> Sink | None:
        with self._lock:
            return next((s for s in self._sinks if s.name == name), None)

    # ---- frames ----------------------------------------------------------------------------

    def submit(self, frame: Frame, info: FrameInfo, effects: Mapping[int, Effect] | None = None) -> None:
        if not frame.frozen:
            frame.freeze()
        self.frames += 1
        for sink in self.sinks:
            if getattr(sink, "degraded", False):
                self.detach(sink, reason=getattr(sink, "failure_reason", None) or "degraded")
                continue
            try:
                sink.submit(frame, info, effects)
            except Exception as exc:
                self._failed(sink, info, exc)
            else:
                self._consecutive[sink.name] = 0

    def latch(self) -> None:
        for sink in self.sinks:
            latch = getattr(sink, "latch", None)
            if latch is not None:
                latch()

    def blackout(self) -> None:
        """Forward to every sink that can black out (the hardware)."""
        for sink in self.sinks:
            blackout = getattr(sink, "blackout", None)
            if blackout is not None:
                blackout()

    def unblackout(self) -> None:
        for sink in self.sinks:
            unblackout = getattr(sink, "unblackout", None)
            if unblackout is not None:
                unblackout()

    def set_brightness(self, value: int) -> None:
        """Forward to every sink with a brightness (the hardware)."""
        for sink in self.sinks:
            if hasattr(sink, "brightness"):
                sink.brightness = value

    def close(self) -> None:
        for sink in self.sinks:
            self.detach(sink)

    # ---- state -----------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """A snapshot for the admin page: each attached sink's counters and
        health, plus what was detached and why."""
        sinks = {}
        for sink in self.sinks:
            entry: dict[str, Any] = {"attached": True}
            for attr in ("frames", "frames_handled", "dropped", "failures", "consecutive_failures", "healthy", "degraded", "muted"):
                value = getattr(sink, attr, None)
                if value is not None:
                    entry[attr] = value
            sinks[sink.name] = entry
        for name, reason in self.detached.items():
            sinks[name] = {"attached": False, "reason": reason}
        return {"frames": self.frames, "sinks": sinks}

    def _failed(self, sink: Sink, info: FrameInfo, exc: Exception) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        with self._lock:
            count = self._consecutive.get(sink.name, 0) + 1
            self._consecutive[sink.name] = count
        log.warning("sink %s raised on frame %d: %s", sink.name, info.n, reason)
        if count >= self.max_failures:
            log.error("sink %s detached after %d consecutive failures", sink.name, count)
            self.detach(sink, reason=reason)
