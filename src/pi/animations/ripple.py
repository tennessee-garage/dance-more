"""Ripples spreading from random points, like rain on the floor.

Demonstrates continuous floor coordinates. `ctx.geometry.led_positions`
is every LED's (y, x) cell-centre position as a `(64, 60, 2)` float
array, aligned with the frame's own `(64, 60, 3)` data - so a distance
field is one numpy expression and indexes straight into the frame, no
loops. `frame.splat()` paints a soft spot the same way. Everything is in
cell units (0..136 on both axes), so the code does not know or care how
many LEDs there are.

Also the ownership idiom: `previous` is read-only, so a fading trail is
`previous.copy().gain(k)` and this frame's content on top.
"""

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.pixels import PixelFrame


@animation(
    name="Ripple",
    description="Expanding rings from random drops, fading behind.",
    author="df2",
    format="pixel",
    tags=["ambient", "geometric"],
    params={
        "rate": Param(float, default=1.5, min=0.1, max=10.0, label="Drops per second"),
        "speed": Param(float, default=40.0, min=5.0, max=150.0, label="Ring speed (cells/s)"),
        "width": Param(float, default=4.0, min=1.0, max=20.0, label="Ring width (cells)"),
        "decay": Param(float, default=0.85, min=0.5, max=0.98, label="Trail persistence"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    drops = ctx.state.setdefault("drops", [])  # (t0, y, x, colour)

    # new drops: Poisson-ish, on average `rate` per second
    if ctx.rng.random() < ctx.params["rate"] * ctx.dt:
        colour = np.array([ctx.rng.randrange(80, 256) for _ in range(3)], dtype=np.float32)
        drops.append((ctx.t, ctx.rng.uniform(0, geo.height), ctx.rng.uniform(0, geo.width), colour))
    # forget rings that have left the floor
    max_r = np.hypot(geo.height, geo.width)
    drops[:] = [d for d in drops if (ctx.t - d[0]) * ctx.params["speed"] < max_r]

    frame = previous.copy().gain(ctx.params["decay"])  # the trail
    width = ctx.params["width"]
    for t0, y, x, colour in drops:
        r = (ctx.t - t0) * ctx.params["speed"]
        # distance of every LED from the ring: (64, 60), aligned with frame.data
        d = np.abs(np.linalg.norm(geo.led_positions - np.array([y, x], dtype=np.float32), axis=-1) - r)
        w = np.exp(-(d * d) / (2 * (width / 2) ** 2))  # 0..1, peaked on the ring
        frame.data[...] = np.maximum(frame.data, (w[..., None] * colour).astype(np.uint8))
        if r < width:  # the splash at the drop point, briefly
            frame.splat(x, y, colour, radius=width, blend="max")
    return frame
