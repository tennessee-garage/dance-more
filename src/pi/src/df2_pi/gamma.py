"""The one gamma curve: frame bytes <-> linear light.

Frame data is gamma-ENCODED bytes: 128 is not half as bright as 255. A
naive mean of encoded bytes comes out too dark - a tile half black and half
full white averages to 128, which is 0.22 of full light rather than 0.5.
Anything that is arithmetic on light rather than on perception goes through
`to_linear()` first and back through `from_linear()` after; `pixels.py`
does so for `PixelFrame.to_tiles()` and `blend()`, and `encode.py` decodes
through the same exponent on the way to the LEDs. The preview and the
floor must agree about what a byte means, so this is defined once, here.

The curve is a plain power law, `linear = (byte / 255) ** GAMMA`, rather
than the piecewise sRGB transfer function: the LEDs are driven by PWM duty,
which is linear in light, and 2.2 is the conventional perceptual fit. The
value is a module constant rather than a parameter on every call so the
two directions can never be built from different exponents.
"""

from __future__ import annotations

import numpy as np

GAMMA = 2.2

# byte -> linear light in 0.0..1.0, one gather per frame. float32 keeps the
# 256-entry table tiny and is more precision than an 8-bit output needs.
DECODE_LUT = ((np.arange(256, dtype=np.float64) / 255.0) ** GAMMA).astype(np.float32)


def to_linear(data: np.ndarray) -> np.ndarray:
    """Gamma-encoded uint8 -> linear light float32 in 0..1, same shape."""
    if data.dtype != np.uint8:
        raise TypeError(f"expected uint8 encoded data, got {data.dtype}")
    return DECODE_LUT[data]


def from_linear(lin: np.ndarray) -> np.ndarray:
    """Linear light in 0..1 -> gamma-encoded uint8, same shape. Values
    outside 0..1 saturate rather than wrap.

    Round-trips exactly: `from_linear(to_linear(v)) == v` for every byte
    value. That is a construction invariant the rest of the driver relies
    on (a uniform tile must reduce to its own colour), and it is easy to
    break with a truncating cast instead of a rounding one.
    """
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    return np.rint(lin ** (1.0 / GAMMA) * 255.0).astype(np.uint8)

