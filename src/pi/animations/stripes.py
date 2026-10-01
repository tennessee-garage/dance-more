"""Stripes: a band of colour sliding along each row. From the v1 floor.

Ported from v1's `processor/stripes.py` (2017). Each of the eight tile rows
carries one stripe: a single palette colour, bright at its peak and fading
either side as 1/n - 1/2, 1/3, 1/4 ... down to 1/length - over `length`
steps. The row is a window eight tiles wide onto that strip, padded with
eight dark tiles at each end, and the strip slides through it one way or
the other at its own speed. When it has passed, the row gets a new stripe:
another colour from the palette, a new speed, a new direction.

v1 drove its LEDs with plain linear values, so its 1/n fade was a fade in
light. Here a frame is perceptual bytes, so the stripe is built in linear
light and encoded with `gamma.from_linear()` - the tails come out as dim
as they were on v1, not the much brighter 1/n of a byte value.

v1 stepped once per frame at 24 fps; speeds here are the same steps per
second, scaled by Speed (and the show's Speed control).
"""

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.gamma import from_linear
from df2_pi.pixels import TileFrame

# v1's color_utils palettes, verbatim.
PALETTES = {
    "rainbow_bunny": ["31cb00", "f9c80e", "f86624", "f86624", "ea3546", "662e9b", "43bccd"],
    "new_mexico": ["004777", "a30000", "ff7700", "efd28d", "00afb5"],
    "desert": ["ff9f1c", "ffbf69", "ffffff", "cbf3f0", "2ec4b6"],
    "druids": ["483c46", "3c6e71", "70ae6e", "beee62", "f4743b"],
    "autumn": ["8ea604", "f5bb00", "ec9f05", "d76a03", "bf3100"],
    "unicorns": ["dec5e3", "cdedfd", "b6dcfe", "a9f8fb", "81f7e5"],
    "linoleum": ["d33f49", "d7c0d0", "eff0d1", "77ba99", "806c89"],
    "wedding1": ["f8aeaa", "abaab2", "f5927b"],
    "rygw": ["ff0000", "ffff00", "00ff00", "ffffff"],
}
V1_FPS = 24
MIN_SPEED, MAX_SPEED = 0.2, 1.0  # v1's defaults, in steps per v1 frame
WINDOW = 8  # tiles across a row, and the dark padding each end


def _linear(hex_colour: str) -> np.ndarray:
    """A v1 palette entry as linear light: v1 sent hex/255 straight to the LEDs."""
    return np.array([int(hex_colour[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


@animation(
    name="Stripes",
    description="A band of colour sliding along each row, fading away either side. From the v1 floor.",
    author="garth",
    format="tile",
    tags=["colour", "v1"],
    params={
        "speed": Param(float, default=1.0, min=0.1, max=4.0, label="Speed", curve="log"),
        "length": Param(int, default=100, min=2, max=200, label="Fade length", help="Steps from a stripe's peak to its faintest", macro=1),
        "palette": Param(str, default="random", choices=["random", *PALETTES], label="Palette"),
    },
)
def render(previous: TileFrame, ctx) -> TileFrame:
    state = ctx.state
    choice = ctx.params["palette"]
    if state.get("choice") != choice:  # frame 0, or a new palette: the rows keep their stripes until they pass
        name = choice if choice != "random" else ctx.rng.choice(list(PALETTES))
        state["choice"] = choice
        state["palette"] = [_linear(c) for c in PALETTES[name]]
    rows = ctx.geometry.tile_rows
    stripes = state.setdefault("stripes", [_stripe(ctx) for _ in range(rows)])

    frame = TileFrame.black(ctx.geometry)
    step = ctx.dt * ctx.params["speed"] * V1_FPS
    for row, stripe in enumerate(stripes):
        strip, colour, speed, direction = stripe["strip"], stripe["colour"], stripe["speed"], stripe["direction"]
        start = int(stripe["start"])
        window = strip[start : start + WINDOW]
        # v1 numbered rows from the top as displayed; row 0 here is the bottom
        frame.data[rows - 1 - row, :] = from_linear(window[:, None] * colour[None, :])
        end = len(strip) - WINDOW
        stripe["start"] += direction * speed * step
        if (direction > 0 and stripe["start"] >= end) or (direction < 0 and stripe["start"] <= 0):
            stripes[row] = _stripe(ctx)
    return frame


def _stripe(ctx) -> dict:
    """A new stripe: a colour, a speed and a direction, starting off the row."""
    length = ctx.params["length"]
    fade = 1.0 / np.arange(2, length + 1, dtype=np.float32)
    dark = np.zeros(WINDOW, dtype=np.float32)
    strip = np.concatenate([dark, fade[::-1], [1.0], fade, dark])  # peak in the middle
    direction = 1 if ctx.rng.random() > 0.5 else -1
    return {
        "strip": strip,
        "colour": ctx.state["palette"][ctx.rng.randrange(len(ctx.state["palette"]))],
        "speed": ctx.rng.uniform(MIN_SPEED, MAX_SPEED),
        "direction": direction,
        "start": 0.0 if direction > 0 else float(len(strip) - WINDOW),
    }
