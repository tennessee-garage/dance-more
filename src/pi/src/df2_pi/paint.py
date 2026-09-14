"""Continuous-coordinate painting onto a PixelFrame.

For animations that think geometrically rather than per-LED: a spot here,
a line from there to there, a ring of this radius. Coordinates are (x, y)
in cell units - 0..width along the row (east) and 0..height across rows
(north) - so an animation written this way is resolution-independent.

Each primitive rasterises onto LIT CELLS ONLY. It computes a distance from
every LED's cell-centre position (`FloorGeometry.led_positions`) to the
shape, turns that into a 0..1 weight through `falloff`, and blends
`colour * weight` into the frame. A diagonal line across the floor lights
whichever edge LEDs it passes near and nothing in the dark interiors,
which is exactly what someone standing on the floor sees. Nothing is ever
written to a dark cell because dark cells are not in the frame at all.

    falloff   'flat'      1 inside the radius, 0 outside
              'linear'    1 at the centre, 0 at the radius
              'gaussian'  exp(-d^2 / 2 sigma^2) with sigma = radius / 2,
                          cut to 0 beyond the radius (e^-2 there)

    blend     'set'       dst = src              overwrite with the shaded colour
              'add'       dst = dst + src        saturating - the default
              'max'       dst = max(dst, src)    per channel
              'alpha'     dst = dst * (1 - w) + src   mix by weight

`src` is `colour * weight` in every mode. Blending happens on the encoded
bytes, like `gain()` - this is animation-level compositing, not
photometry; see pixels.py for which operations work in linear light.

These MUTATE the frame - they are drawing operations - which is why the
idiom is to paint onto a copy of the previous frame, never onto the frozen
`previous` itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from df2_pi.pixels import PixelFrame

FALLOFFS = ("flat", "linear", "gaussian")
BLENDS = ("set", "add", "max", "alpha")

Point = tuple[float, float]


def splat(
    frame: PixelFrame,
    x: float,
    y: float,
    color,
    radius: float = 4.0,
    falloff: str = "gaussian",
    blend: str = "add",
) -> None:
    """A spot of `color` centred at (x, y), fading to nothing at `radius`."""
    px, py = _positions(frame)
    d = np.hypot(px - x, py - y)
    _paint(frame, _weights(d, radius, falloff), color, blend)


def line(
    frame: PixelFrame,
    start: Point,
    end: Point,
    color,
    width: float = 1.5,
    falloff: str = "flat",
    blend: str = "add",
) -> None:
    """A segment from `start` to `end`, `width` cells wide."""
    px, py = _positions(frame)
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        t = 0.0
    else:
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / length_sq, 0.0, 1.0)
    d = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
    _paint(frame, _weights(d, width / 2.0, falloff), color, blend)


def circle(
    frame: PixelFrame,
    cx: float,
    cy: float,
    r: float,
    color,
    width: float = 1.0,
    falloff: str = "flat",
    blend: str = "add",
) -> None:
    """A ring of radius `r` centred at (cx, cy), `width` cells wide."""
    if r < 0:
        raise ValueError(f"radius must be non-negative, got {r}")
    px, py = _positions(frame)
    d = np.abs(np.hypot(px - cx, py - cy) - r)
    _paint(frame, _weights(d, width / 2.0, falloff), color, blend)


# ---- internals ----------------------------------------------------------------


def _positions(frame: PixelFrame) -> tuple[np.ndarray, np.ndarray]:
    """Every LED's (x, y) cell-centre position as two `(tiles, leds)` arrays."""
    pos = frame.geometry.led_positions  # (tiles, leds, 2) as (y, x)
    return pos[..., 1], pos[..., 0]


def _weights(d: np.ndarray, radius: float, falloff: str) -> np.ndarray:
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if falloff == "flat":
        return (d <= radius).astype(np.float64)
    if falloff == "linear":
        return np.clip(1.0 - d / radius, 0.0, 1.0)
    if falloff == "gaussian":
        sigma = radius / 2.0
        w = np.exp(-(d * d) / (2.0 * sigma * sigma))
        w[d > radius] = 0.0
        return w
    raise ValueError(f"falloff must be one of {FALLOFFS}, got {falloff!r}")


def _paint(frame: PixelFrame, w: np.ndarray, color, blend: str) -> None:
    if blend not in BLENDS:
        raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}")
    c = np.asarray(color, dtype=np.float64)
    if c.shape != (3,):
        raise ValueError(f"color must be an (r, g, b) triple, got shape {c.shape}")
    mask = w > 0.0
    if not mask.any():
        return
    wv = w[mask][:, None]  # (k, 1)
    src = c[None, :] * wv  # (k, 3)
    dst = frame.data[mask].astype(np.float64)
    if blend == "set":
        out = src
    elif blend == "add":
        out = dst + src
    elif blend == "max":
        out = np.maximum(dst, src)
    else:
        out = dst * (1.0 - wv) + src
    frame.data[mask] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
