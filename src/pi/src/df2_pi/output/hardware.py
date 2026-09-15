"""`HardwareSink`: the floor. The one sink that runs on the render thread.

It is the real-time work and owns the deadline: `submit()` encodes the
frame (encode.py) and puts every row's SEND_DATA on the wire with
`Floor.send_rows()`, both chains concurrently; `latch()` fires the
broadcast that lights the floor, and is what the frame clock calls at
the deadline (`FrameClock.run(latch=sink.latch)`).

Serial write failures are counted and logged, not raised - one bad frame
must not stop playback. `consecutive_failures` reaching `max_failures`
clears `healthy`, which the admin page shows; it recovers on the next
successful frame. Global brightness lives in the encoder's LUT, so this
sink is where it is set. With a `FrameClock` attached it marks the
"encode" and "wire" phases into the clock's telemetry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping

from df2_pi.effects import Effect
from df2_pi.encode import FrameEncoder
from df2_pi.engine.clock import FrameClock, FrameInfo
from df2_pi.pixels import Frame

if TYPE_CHECKING:  # the transport pulls in gpiozero; a laptop must not
    from df2_pi.transport.floor import Floor

log = logging.getLogger(__name__)


class HardwareSink:
    def __init__(
        self,
        floor: Floor,
        encoder: FrameEncoder | None = None,
        *,
        clock: FrameClock | None = None,
        name: str = "hardware",
        max_failures: int = 3,
    ) -> None:
        self.name = name
        self.floor = floor
        self.encoder = encoder if encoder is not None else FrameEncoder()
        self.clock = clock
        self.max_failures = max_failures
        self.frames = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.last_error: str | None = None
        self.muted = False
        self._sent = False

    # ---- settings --------------------------------------------------------------------

    @property
    def brightness(self) -> int:
        return self.encoder.brightness

    @brightness.setter
    def brightness(self, value: int) -> None:
        self.encoder.brightness = value

    @property
    def healthy(self) -> bool:
        return self.consecutive_failures < self.max_failures

    # ---- the Sink protocol -------------------------------------------------------------

    def submit(self, frame: Frame, info: FrameInfo, effects: Mapping[int, Effect] | None = None) -> None:
        """Encode and put the frame on the Row Bus. Latching is separate.
        While `muted` (a blackout), every row gets the black payload
        instead, so observers still see the live content and the floor
        stays dark."""
        if self.muted:
            payloads = [self.encoder.blackout_payload()] * self.encoder.geometry.tile_rows
        else:
            payloads = self.encoder.encode(frame, effects)
        if self.clock is not None:
            self.clock.mark("encode")
        try:
            self.floor.send_rows(payloads)
        except Exception as exc:
            self._fail("send_rows", info, exc)
        else:
            self._sent = True
        if self.clock is not None:
            self.clock.mark("wire")

    def latch(self) -> None:
        """Light the frame. Called by the clock at the deadline."""
        try:
            self.floor.latch()
        except Exception as exc:
            self._fail("latch", None, exc)
            return
        if self._sent:
            self.frames += 1
            self.consecutive_failures = 0
        self._sent = False

    def blackout(self) -> None:
        """Broadcast BLACKOUT - clears every tile's pixel buffer AND effect
        register - and mute, so frames keep flowing but go out black
        until `unblackout()`."""
        self.muted = True
        try:
            self.floor.blackout()
        except Exception as exc:
            self._fail("blackout", None, exc)

    def unblackout(self) -> None:
        self.muted = False

    def close(self) -> None:
        pass  # the Floor's lifetime is the caller's

    def _fail(self, what: str, info: FrameInfo | None, exc: Exception) -> None:
        self.failures += 1
        self.consecutive_failures += 1
        self.last_error = f"{what}: {type(exc).__name__}: {exc}"
        frame = f" frame {info.n}" if info is not None else ""
        if self.consecutive_failures == self.max_failures:
            log.error("hardware sink unhealthy: %d consecutive failures (%s)", self.consecutive_failures, self.last_error)
        else:
            log.warning("hardware sink%s: %s", frame, self.last_error)
