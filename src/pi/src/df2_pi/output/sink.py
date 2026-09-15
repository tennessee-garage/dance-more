"""Sinks: where rendered frames go, and the rule that keeps them harmless.

Every frame goes to the hardware AND to any number of observers - the
admin page's preview, a recorder, a terminal view. The invariant:

    Nothing except the hardware sink may ever block the frame clock.

A browser on bad wifi, an SD card mid-flush, a debugger on a breakpoint -
none of them may cost the floor a frame. So every observer sink runs on
its own thread behind a single-slot, LATEST-WINS mailbox: `submit()` just
replaces whatever unconsumed frame was there and returns in microseconds.
A slow consumer sees 8 FPS instead of 30 and the floor never notices.
There are no unbounded queues anywhere here, because an unbounded queue
in front of a slow consumer is a memory leak with extra steps.

    Sink            the protocol: name, submit(frame, info), close()
    Mailbox         the one-slot latest-wins handoff
    ThreadedSink    base for observers: a mailbox, a worker thread, and
                    failure accounting that marks the sink `degraded`
    NullSink        headless runs and benchmarks
    CallbackSink    calls a function per frame; the general-purpose hook

`HardwareSink` (hardware.py) is the one exception - synchronous on the
render thread, because it IS the real-time work. `PreviewSink` and
`RecorderSink` are in preview.py; `FanOut` (fanout.py) holds the list.

Frames handed to observers are frozen (read-only), so a misbehaving
observer cannot corrupt the next frame's `previous`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from df2_pi.effects import Effect
from df2_pi.engine.clock import FrameInfo
from df2_pi.pixels import Frame

log = logging.getLogger(__name__)

DEFAULT_MAX_FAILURES = 5


@runtime_checkable
class Sink(Protocol):
    name: str

    def submit(self, frame: Frame, info: FrameInfo, effects: Mapping[int, Effect] | None = None) -> None:
        """Take a frame. Must not block for observers; see the module docstring."""

    def close(self) -> None: ...


class Mailbox:
    """A single-slot handoff: `put()` replaces the pending item, `take()`
    blocks for the newest one. Closing wakes any waiter with None."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Any = None
        self._pending = False
        self._closed = False
        self.replaced = 0  # items overwritten before anyone took them

    def put(self, item: Any) -> None:
        with self._cond:
            if self._pending:
                self.replaced += 1
            self._item = item
            self._pending = True
            self._cond.notify()

    def take(self, timeout: float | None = None) -> Any:
        """The pending item, or None if closed (or `timeout` elapsed)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._pending or self._closed, timeout):
                return None
            if not self._pending:
                return None
            item, self._item, self._pending = self._item, None, False
            return item

    def peek(self) -> Any:
        with self._cond:
            return self._item if self._pending else None

    @property
    def pending(self) -> int:
        with self._cond:
            return 1 if self._pending else 0

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._item = None
            self._pending = False
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


class ThreadedSink:
    """An observer: `submit()` drops the frame in a mailbox and returns;
    a worker thread calls `handle(frame, info)` with the newest one.

    `handle()` raising is counted; after `max_failures` consecutive
    raises the sink is `degraded` (with `failure_reason`) and its worker
    stops. The `FanOut` notices on its next submit and detaches it.
    """

    name: str

    def __init__(self, name: str, *, max_failures: int = DEFAULT_MAX_FAILURES) -> None:
        self.name = name
        self.max_failures = max_failures
        self.mailbox = Mailbox()
        self.frames_submitted = 0
        self.frames_handled = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.failure_reason: str | None = None
        self._degraded = False
        self._thread = threading.Thread(target=self._run, name=f"sink:{name}", daemon=True)
        self._started = False

    # ---- the Sink protocol -------------------------------------------------------

    def submit(self, frame: Frame, info: FrameInfo, effects: Mapping[int, Effect] | None = None) -> None:
        if self._degraded or self.mailbox.closed:
            return
        if not self._started:
            self._started = True
            self._thread.start()
        self.frames_submitted += 1
        self.mailbox.put((frame, info))

    def close(self) -> None:
        self.mailbox.close()
        if self._started and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    # ---- for subclasses ------------------------------------------------------------

    def handle(self, frame: Frame, info: FrameInfo) -> None:
        raise NotImplementedError

    def on_close(self) -> None:
        """Called on the worker thread when it exits (closed or degraded)."""

    # ---- health --------------------------------------------------------------------

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def dropped(self) -> int:
        """Frames overwritten in the mailbox before this sink got to them."""
        return self.mailbox.replaced

    def _run(self) -> None:
        try:
            while True:
                item = self.mailbox.take()
                if item is None:
                    return
                frame, info = item
                try:
                    self.handle(frame, info)
                except Exception as exc:
                    self.failures += 1
                    self.consecutive_failures += 1
                    self.failure_reason = f"{type(exc).__name__}: {exc}"
                    log.warning("sink %s failed on frame %d: %s", self.name, info.n, self.failure_reason)
                    if self.consecutive_failures >= self.max_failures:
                        self._degraded = True
                        log.error(
                            "sink %s degraded after %d consecutive failures; detaching",
                            self.name,
                            self.consecutive_failures,
                        )
                        return
                else:
                    self.frames_handled += 1
                    self.consecutive_failures = 0
        finally:
            self.mailbox.close()
            try:
                self.on_close()
            except Exception:
                log.exception("sink %s failed while closing", self.name)


class NullSink:
    """Counts frames and does nothing else. For headless runs and benchmarks."""

    def __init__(self, name: str = "null") -> None:
        self.name = name
        self.frames = 0

    def submit(self, frame: Frame, info: FrameInfo, effects: Mapping[int, Effect] | None = None) -> None:
        self.frames += 1

    def close(self) -> None:
        pass


class CallbackSink(ThreadedSink):
    """Calls `fn(frame, info)` for each frame, on its own thread."""

    def __init__(
        self,
        fn: Callable[[Frame, FrameInfo], None],
        name: str = "callback",
        *,
        max_failures: int = DEFAULT_MAX_FAILURES,
    ) -> None:
        super().__init__(name, max_failures=max_failures)
        self.fn = fn

    def handle(self, frame: Frame, info: FrameInfo) -> None:
        self.fn(frame, info)
