"""Spiral: colours wound out from the middle of the floor, one tile at a time.

Ported from the v1 floor's `processor/spiral.py` (2018). A train of 64
colours lies along a fixed spiral through every tile, starting at the
middle and winding clockwise out to a corner. Each step pushes one new
colour onto the middle and shifts the rest one tile further along, so
the oldest falls off the end.

The new colour's hue drifts slowly, and on a rhythm set by the palette it
jumps to one of the drifting hue's triad partners - the two hues 150
degrees either side of it - which is what lays bands and accents into the
spiral as it unwinds:

- accents: every fourth tile, alternating between the two partners
- bands:   a run of 32, a band of 8 in one partner, 32, a band of 8 in the other
- stripes: alternating with one partner, then a short band of the other
- rainbow: no jumps, just a faster drift

v1 stepped once per frame at 24 fps; here a step is clocked by `ctx.t`, so
Speed sets the steps per second and the show's Speed control applies.
"""

import colorsys

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.pixels import TileFrame

# v1's tile numbering is row-major from the top-left as displayed (0..7 is the top row, 56..63
# the bottom), and this is its order verbatim: the middle, then clockwise outward.
READ_ORDER = [
    27, 28, 36, 35, 34, 26, 18, 19, 20, 21, 29, 37, 45, 44, 43, 42,
    41, 33, 25, 17, 9, 10, 11, 12, 13, 14, 22, 30, 38, 46, 54, 53,
    52, 51, 50, 49, 48, 40, 32, 24, 16, 8, 0, 1, 2, 3, 4, 5,
    6, 7, 15, 23, 31, 39, 47, 55, 63, 62, 61, 60, 59, 58, 57, 56,
]  # fmt: skip
# Row 0 is the bottom as displayed here, so v1's row r from the top is row 7 - r.
PATH_ROWS = np.array([7 - i // 8 for i in READ_ORDER])
PATH_COLS = np.array([i % 8 for i in READ_ORDER])

TRIAD = [0.0, 0.5 + 1 / 12, 0.5 - 1 / 12]  # the hue itself, and its two partners
PALETTES = {  # name: (hue drift per step, which TRIAD entry each step jumps by, cycled)
    "accents": (0.001, [0, 0, 0, 1, 0, 0, 0, 2]),
    "bands": (0.001, [0] * 32 + [1] * 8 + [0] * 32 + [2] * 8),
    "stripes": (0.001, [0, 1] * 16 + [2] * 4),
    "rainbow": (0.005, [0]),
}


@animation(
    name="Spiral",
    description="Colours wound out from the middle of the floor, banded with triad accents. From the v1 floor.",
    author="garth",
    format="tile",
    tags=["colour", "geometric", "v1"],
    params={
        "speed": Param(float, default=24.0, min=2.0, max=120.0, label="Steps per second", curve="log"),
        "palette": Param(str, default="bands", choices=list(PALETTES), label="Palette", macro=1),
    },
)
def render(previous: TileFrame, ctx) -> TileFrame:
    if not ctx.state:
        ctx.state["train"] = np.zeros((len(READ_ORDER), 3), dtype=np.uint8)  # newest first
        ctx.state["hue"] = 0.0
        ctx.state["count"] = 0  # where in the palette's jump cycle the next step is
        ctx.state["steps"] = 0
    drift, jumps = PALETTES[ctx.params["palette"]]
    train = ctx.state["train"]

    # Catch up on the steps due by now; past a full train, earlier ones would only fall off the end.
    due = int(ctx.t * ctx.params["speed"] + 1e-9)
    for _ in range(min(due - ctx.state["steps"], len(train))):
        ctx.state["hue"] = (ctx.state["hue"] + drift) % 1.0
        hue = (ctx.state["hue"] + TRIAD[jumps[ctx.state["count"] % len(jumps)]]) % 1.0
        ctx.state["count"] = (ctx.state["count"] + 1) % len(jumps)
        train[1:] = train[:-1].copy()
        train[0] = [int(v * 255) for v in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
    ctx.state["steps"] = max(ctx.state["steps"], due)

    frame = TileFrame.black(ctx.geometry)
    frame.data[PATH_ROWS, PATH_COLS] = train
    return frame
