"""One colour, everywhere. The smallest animation there is - copy this one.

An animation is a file with one decorated function. `render()` gets the
previous frame (read-only) and a context, and returns a NEW frame. That
is the whole contract; everything else is optional.
"""

import colorsys

from df2_pi.animation import Param, animation
from df2_pi.pixels import TileFrame


@animation(
    name="Solid",
    description="One colour on every tile.",
    author="df2",
    format="tile",  # a TileFrame: one colour per tile, 8x8. Cheapest thing on the wire.
    tags=["basic"],
    params={
        # Params become sliders in the web UI and `--param hue=0.3` on the command line.
        "hue": Param(float, default=0.6, min=0.0, max=1.0, label="Hue", help="0 red, 0.33 green, 0.66 blue"),
    },
)
def render(previous: TileFrame, ctx) -> TileFrame:
    frame = TileFrame.black(ctx.geometry)
    r, g, b = colorsys.hsv_to_rgb(ctx.params["hue"], 1.0, 1.0)
    frame.data[:] = (int(r * 255), int(g * 255), int(b * 255))
    return frame
