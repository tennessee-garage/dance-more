"""Show-level controls: what an operator does to whatever is playing.

    overlays = Overlays(strobe_max_hz=10.0)
    overlays.set_tint((255, 80, 0), 0.5)
    frame = overlays.apply(frame, t)          # per frame, after render

A MIDI controller, a lighting desk or the web UI shapes the floor with
these without knowing which animation is up. They act on the rendered
frame (after crossfades, before the floor rotation), so they behave the
same over every animation - and over External content from a media
server. The runner owns one `Overlays` and calls `apply()` on its render
thread; every setter is called from there too, via the control queue.

Order, fixed:

    freeze -> colour correction -> tint -> bump -> strobe

Freeze holds the rendered picture while animations keep running
underneath; everything after it still acts on the held picture, the way a
lighting desk's master and strobe keep working over a frozen look.

- Colour correction (`hue_shift` in turns, `saturation` 0..2) is one 3x3
  matrix on the encoded bytes - the W3C Filter Effects hue-rotate and
  saturate matrices, the ones CSS uses - so pulling the floor toward a
  room's palette costs a single matmul.
- Tint colourises in linear light: each pixel's luminance, in the tint's
  colour, mixed in by `amount`. Black stays black - a dark LED is never
  lit by a tint - and full white becomes the tint at full strength.
- Bump is a momentary flash toward white by `level`, fading linearly to
  nothing over `decay_s`. At level 1 that is a full-white frame: the
  floor's full-white current draw.
- Strobe shutters the picture: shown for one frame at the start of each
  period, dark for the rest, with periods counted from the moment it was
  switched on - so the first frame is always a flash and the next is never
  a second one. `rate_hz` is capped by `strobe_max_hz`, an admin setting
  rather than a live control.

`speed` is held here so the runner's state has it in one place, but it is
not a frame transform: the runner applies it to the time an animation
sees (`ctx.t`, `ctx.dt`).

Every stage is skipped when it is at identity, so an untouched floor pays
nothing; a tile frame stays a tile frame throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from df2_pi.gamma import from_linear, to_linear
from df2_pi.pixels import Frame

SPEED_MAX = 4.0
SATURATION_MAX = 2.0
DECAY_MAX_S = 10.0
STROBE_MAX_HZ_LIMIT = 15.0  # the admin setting's own ceiling: half of 30 fps

# Rec. 709 luma weights, the ones the W3C matrices are built on.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class ShowState:
    """The show controls as the runner's state reports them."""

    speed: float = 1.0
    frozen: bool = False
    strobe_hz: float = 0.0
    strobe_max_hz: float = 10.0
    tint: tuple[int, int, int] = (255, 255, 255)
    tint_amount: float = 0.0
    hue_shift: float = 0.0
    saturation: float = 1.0


def color_matrix(hue_shift: float, saturation: float) -> np.ndarray:
    """The 3x3 that rotates hue by `hue_shift` turns and scales saturation
    by `saturation`, applied as `rgb @ M.T` (W3C Filter Effects matrices)."""
    a = 2.0 * math.pi * hue_shift
    c, s = math.cos(a), math.sin(a)
    hue = np.array(
        [
            [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
            [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
            [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
        ]
    )
    k = saturation
    sat = np.array(
        [
            [0.213 + 0.787 * k, 0.715 - 0.715 * k, 0.072 - 0.072 * k],
            [0.213 - 0.213 * k, 0.715 + 0.285 * k, 0.072 - 0.072 * k],
            [0.213 - 0.213 * k, 0.715 - 0.715 * k, 0.072 + 0.928 * k],
        ]
    )
    return (sat @ hue).astype(np.float32)


class Overlays:
    def __init__(self, strobe_max_hz: float = 10.0) -> None:
        self.speed = 1.0
        self.frozen = False
        self.strobe_hz = 0.0
        self.strobe_max_hz = check_strobe_max(strobe_max_hz)
        self.tint: tuple[int, int, int] = (255, 255, 255)
        self.tint_amount = 0.0
        self.hue_shift = 0.0
        self.saturation = 1.0
        self._held: Frame | None = None
        self._bump_level = 0.0
        self._bump_at = 0.0
        self._bump_decay = 0.25
        self._strobe_period: int | None = None
        self._strobe_from = 0.0
        self._matrix: np.ndarray | None = None

    # ---- setters (render thread) ------------------------------------------------------

    def set_speed(self, speed: float) -> None:
        self.speed = speed

    def set_freeze(self, on: bool) -> None:
        self.frozen = on
        if not on:
            self._held = None

    def bump(self, level: float, decay_s: float, t: float) -> None:
        """Flash toward white by `level` now, fading over `decay_s`. A bump
        weaker than what is still showing of the last one is ignored."""
        if level >= self._bump_now(t):
            self._bump_level, self._bump_at, self._bump_decay = level, t, decay_s

    def set_strobe(self, rate_hz: float, t: float = 0.0) -> None:
        self.strobe_hz = rate_hz
        self._strobe_from = t
        self._strobe_period = None  # the next frame is an on-frame

    def set_strobe_max(self, hz: float) -> None:
        self.strobe_max_hz = check_strobe_max(hz)

    def set_tint(self, rgb: tuple[int, int, int], amount: float) -> None:
        self.tint, self.tint_amount = tuple(rgb), amount

    def set_hue_shift(self, turns: float) -> None:
        self.hue_shift = turns % 1.0
        self._matrix = None

    def set_saturation(self, k: float) -> None:
        self.saturation = k
        self._matrix = None

    def reset(self) -> None:
        """Every control back to identity; the strobe cap is a setting and stays."""
        cap = self.strobe_max_hz
        self.__init__(cap)

    def state(self) -> ShowState:
        return ShowState(
            speed=self.speed,
            frozen=self.frozen,
            strobe_hz=self.strobe_hz,
            strobe_max_hz=self.strobe_max_hz,
            tint=self.tint,
            tint_amount=self.tint_amount,
            hue_shift=self.hue_shift,
            saturation=self.saturation,
        )

    # ---- per frame -------------------------------------------------------------------

    def apply(self, frame: Frame, t: float) -> Frame:
        """`frame` with every active control applied, for the frame shown at
        `t`. Returns `frame` itself when nothing is active."""
        if self.frozen:
            if self._held is None:
                self._held = frame
            frame = self._held
        if self.hue_shift != 0.0 or self.saturation != 1.0:
            frame = self._color_correct(frame)
        bump = self._bump_now(t)
        if self.tint_amount > 0.0 or bump > 0.0:
            frame = self._in_linear(frame, bump)
        rate = min(self.strobe_hz, self.strobe_max_hz)
        if rate > 0.0:
            period = math.floor((t - self._strobe_from) * rate + 1e-6)  # frame times are float sums
            lit = period != self._strobe_period
            self._strobe_period = period
            if not lit:
                frame = type(frame).black(frame.geometry)
        return frame

    def _bump_now(self, t: float) -> float:
        if self._bump_level <= 0.0:
            return 0.0
        remaining = 1.0 - (t - self._bump_at) / self._bump_decay
        return self._bump_level * min(1.0, max(0.0, remaining))

    def _color_correct(self, frame: Frame) -> Frame:
        if self._matrix is None:
            self._matrix = color_matrix(self.hue_shift, self.saturation)
        flat = frame.data.reshape(-1, 3).astype(np.float32) @ self._matrix.T
        out = np.clip(np.rint(flat), 0, 255).astype(np.uint8).reshape(frame.data.shape)
        return type(frame)(out, frame.geometry)

    def _in_linear(self, frame: Frame, bump: float) -> Frame:
        lin = to_linear(frame.data)
        if self.tint_amount > 0.0:
            tint = to_linear(np.array(self.tint, dtype=np.uint8))
            peak = float(tint.max())
            luma = lin @ LUMA
            tinted = luma[..., None] * (tint / peak) if peak > 0 else np.zeros_like(lin)
            lin = lin + (tinted - lin) * self.tint_amount
        if bump > 0.0:
            lin = lin + (1.0 - lin) * bump
        return type(frame)(from_linear(lin), frame.geometry)


def check_strobe_max(hz: float) -> float:
    hz = float(hz)
    if not 0.0 <= hz <= STROBE_MAX_HZ_LIMIT:
        raise ValueError(f"strobe_max_hz must be 0..{STROBE_MAX_HZ_LIMIT}, got {hz}")
    return hz
