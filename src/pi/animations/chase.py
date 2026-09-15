"""Comets chasing each other around the edge of the floor.

Demonstrates `ctx.geometry.floor_ring`: the floor's outer boundary as one
ordered ring of 480 LED indices, clockwise from the top-left corner as
you look down on the floor. An index into it is a position around the
edge, so a chase is `ring[(head - i) % 480]` for a tail of `i`. The
indices are FLAT - `tile * 60 + led` - and go straight into
`frame.flat`, the `(3840, 3)` view of every LED.
"""

import colorsys

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.pixels import PixelFrame


@animation(
    name="Chase",
    description="Comets running round the floor's edge.",
    author="df2",
    format="pixel",
    tags=["edges", "rhythm"],
    params={
        "comets": Param(int, default=3, min=1, max=12, label="Comets"),
        "speed": Param(float, default=120.0, min=10.0, max=480.0, label="LEDs per second"),
        "tail": Param(int, default=40, min=2, max=200, label="Tail length"),
    },
    period=4.0,  # one lap at the default speed: 480 / 120
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    ring = ctx.geometry.floor_ring
    n = len(ring)
    comets = ctx.params["comets"]
    tail = ctx.params["tail"]
    fade = np.linspace(1.0, 0.0, tail, endpoint=False)  # bright head, dark tail

    frame = PixelFrame.black(ctx.geometry)
    head = int(ctx.t * ctx.params["speed"])
    for c in range(comets):
        colour = np.array(colorsys.hsv_to_rgb(c / comets, 1.0, 1.0)) * 255
        offset = head + c * n // comets  # spread evenly around the ring
        positions = ring[(offset - np.arange(tail)) % n]  # head first
        frame.flat[positions] = np.maximum(frame.flat[positions], (fade[:, None] * colour).astype(np.uint8))
    return frame
