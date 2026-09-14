"""A hue gradient that sweeps diagonally across the floor.

The reference animation: the smallest complete example of the file format
in df2_pi/animation/meta.py. Copy it to start a new one.
"""

import colorsys

from df2_pi.animation import Param, animation
from df2_pi.pixels import TileFrame


@animation(
    name="Rainbow Sweep",
    description="A hue gradient that sweeps diagonally across the floor.",
    author="garth",
    format="tile",
    tags=["ambient", "colour"],
    params={
        "speed": Param(float, default=1.0, min=0.1, max=5.0, label="Speed"),
        "saturation": Param(float, default=1.0, min=0.0, max=1.0, label="Saturation"),
    },
    period=5.0,  # one full hue cycle at speed 1.0
    preview_hint="loop",
    energy="low",
)
def render(previous: TileFrame, ctx) -> TileFrame:
    frame = TileFrame.black(ctx.geometry)
    rows, cols = ctx.geometry.tile_rows, ctx.geometry.tile_cols
    phase = ctx.t * ctx.params["speed"] * 0.2
    for r in range(rows):
        for c in range(cols):
            h = ((r + c) / (rows + cols - 2) + phase) % 1.0
            frame[r, c] = tuple(
                int(v * 255) for v in colorsys.hsv_to_rgb(h, ctx.params["saturation"], 1.0)
            )
    return frame
