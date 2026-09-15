"""Lightning: bolts that cross the floor along the tile edges.

The animation this whole driver was shaped around. The floor is not an
image with the middles missing - it is 256 line segments, 15 LEDs each,
meeting at tile corners - and a bolt is a WALK on that structure.

`ctx.geometry.edge_graph()` has edges as nodes, adjacent where they meet
at a corner. `walk()` is a random self-avoiding walk from a starting
edge with a `turn_bias` (0 runs straight whenever it can, 1 turns at
every chance), `branch()` forks new walks off it, and a `Path` gives its
LEDs in travel order as flat indices - so a head-to-tail gradient is
just an index into that sequence. The walk takes `ctx.np_rng`, the
numpy generator seeded alongside `ctx.rng`, so a strike is reproducible.

Strikes flash white, then decay: `previous.copy().gain(k)` is the
afterglow, and the bolt is repainted on top for a few frames at falling
brightness, forks dimmer than the trunk.
"""

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.geometry import Side
from df2_pi.pixels import PixelFrame


@animation(
    name="Lightning",
    description="Bolts crossing the floor along tile edges, with forks.",
    author="df2",
    format="pixel",
    tags=["edges", "dramatic"],
    params={
        "rate": Param(float, default=1.0, min=0.1, max=5.0, label="Strikes per second"),
        "length": Param(int, default=14, min=2, max=60, label="Bolt length (edges)"),
        "turn_bias": Param(float, default=0.35, min=0.0, max=1.0, label="Turn bias", help="0 straight, 1 jagged"),
        "forks": Param(int, default=2, min=0, max=6, label="Forks"),
        "decay": Param(float, default=0.72, min=0.3, max=0.95, label="Afterglow"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    graph = geo.edge_graph()
    bolts = ctx.state.setdefault("bolts", [])  # (t_struck, trunk_leds, fork_leds)

    if ctx.rng.random() < ctx.params["rate"] * ctx.dt:
        # Strike from a random edge on the north wall and wander from there.
        start = geo.edge(geo.tiles - geo.tile_cols + ctx.rng.randrange(geo.tile_cols), Side.NORTH)
        trunk = graph.walk(start, ctx.params["length"], ctx.np_rng, turn_bias=ctx.params["turn_bias"])
        forks = graph.branch(trunk, ctx.np_rng, n=ctx.params["forks"], length=max(2, len(trunk) // 3))
        bolts.append((ctx.t, trunk.leds(), [f.leds() for f in forks]))
    bolts[:] = [b for b in bolts if ctx.t - b[0] < 0.25]  # a bolt is repainted for ~7 frames

    frame = previous.copy().gain(ctx.params["decay"])  # the afterglow of everything before
    for t_struck, trunk, forks in bolts:
        age = (ctx.t - t_struck) / 0.25  # 0 fresh .. 1 gone
        level = (1.0 - age) ** 2
        # bright at the strike point, dimming toward the tip
        ramp = np.linspace(1.0, 0.4, len(trunk))[:, None]
        colour = np.array([210, 225, 255], dtype=np.float32) * level
        frame.flat[trunk] = np.maximum(frame.flat[trunk], (ramp * colour).astype(np.uint8))
        for fork in forks:
            frame.flat[fork] = np.maximum(frame.flat[fork], (colour * 0.5).astype(np.uint8))
    return frame
