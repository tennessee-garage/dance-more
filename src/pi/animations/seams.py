"""The floor's grid lines, with a pulse running along every seam at once.

Demonstrates `ctx.geometry.seams`: the 112 pairs of facing interior
edges. Two adjacent tiles do not share a line of LEDs - they present two
parallel runs of 15 facing each other across the tile frames - and the
pairing guarantees `a.leds[i]` sits directly opposite `b.leds[i]`. So a
pulse at index `i` on both halves of every seam moves in lockstep, and
if any tile were assembled with its strip in from the wrong end, that
seam's pulse would run the other way. This is the wiring check you can
watch.

Edges are ordered in FLOOR orientation (west->east, south->north),
whatever the chain does on that side, so `i` means the same place on
both halves.

Also a performance habit worth copying: the 224 index arrays are gathered
ONCE into `ctx.state` and the per-frame work is a single numpy write. A
Python loop over 224 edges every frame is fine on a laptop and a
noticeable slice of the 33 ms budget on a Pi.
"""

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.edges import Axis
from df2_pi.pixels import PixelFrame


@animation(
    name="Seams",
    description="The grid lines between tiles, with a pulse along each.",
    author="df2",
    format="pixel",
    tags=["edges", "diagnostic"],
    params={
        "speed": Param(float, default=10.0, min=1.0, max=60.0, label="LEDs per second"),
        "base": Param(int, default=30, min=0, max=120, label="Grid brightness"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    n = geo.leds_per_side
    if not ctx.state:  # frame 0: gather every seam edge's LEDs and colour, once
        edges = [edge for pair in geo.seams for edge in pair]
        ctx.state["leds"] = np.array([e.flat_leds for e in edges])  # (224, 15) flat indices
        ctx.state["colour"] = np.array(
            [(255, 120, 0) if e.axis is Axis.X else (0, 160, 255) for e in edges], dtype=np.float32
        )  # (224, 3): horizontal seams orange, vertical blue

    pulse = int(ctx.t * ctx.params["speed"]) % n  # the same index along every seam
    glow = np.zeros(n, dtype=np.float32)
    for k in range(4):  # a short bright head with a tail behind it
        glow[(pulse - k) % n] = 1.0 - k * 0.25

    frame = PixelFrame.black(geo)
    lit = glow[None, :, None] * ctx.state["colour"][:, None, :]  # (224, 15, 3)
    frame.flat[ctx.state["leds"]] = np.maximum(ctx.params["base"], lit).astype(np.uint8)
    return frame
