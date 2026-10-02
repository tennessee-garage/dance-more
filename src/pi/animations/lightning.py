"""Lightning: bolts that come down the floor along the tile edges.

The animation this whole driver was shaped around. The floor is not an
image with the middles missing - it is 256 line segments, 15 LEDs each,
meeting at tile corners - and a bolt is a path through that structure.

`ctx.geometry.edge_graph()` has edges as nodes, meeting at junctions (tile
corners). A bolt here is built junction by junction: from a point on the
top wall it heads straight down, and at every corner after that it goes
down again or jogs sideways - sideways with the Turn bias probability -
but never climbs, so it can't curl back on itself. It ends where it meets
the bottom of the floor (or runs out of length). Either half of a seam
will do for the next step; once a span is used, both halves are spent.

Forks are what make it read as lightning rather than a crack, and real
ones spread out as the bolt nears the ground. So a fork leaves from a
corner chosen with a weight that grows with the square of how far along
the trunk it is - none in the first fifth, most toward the end - and then
descends the same way, a little more jagged, never crossing the trunk.

Strikes flash white, then decay: `previous.copy().gain(k)` is the
afterglow, and the bolt is repainted on top for a few frames at falling
brightness, forks dimmer than the trunk. All the randomness is
`ctx.rng`, so a strike is reproducible from the run's seed.
"""

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.pixels import PixelFrame

QUIET_TOP = 0.2  # the first fifth of the trunk never forks
FORK_JAGGED = 0.2  # forks turn this much more often than the trunk


@animation(
    name="Lightning",
    description="Bolts coming down the floor along tile edges, forking as they near the ground.",
    author="df2",
    format="pixel",
    tags=["edges", "dramatic"],
    params={
        "rate": Param(float, default=1.0, min=0.1, max=5.0, label="Strikes per second", role="density", curve="log"),
        "length": Param(int, default=14, min=2, max=60, label="Bolt length (edges)"),
        "turn_bias": Param(float, default=0.35, min=0.0, max=1.0, label="Turn bias", help="0 straight down, 1 jagged", role="variation"),
        "forks": Param(int, default=2, min=0, max=6, label="Forks", macro=1),
        "decay": Param(float, default=0.72, min=0.3, max=0.95, label="Afterglow"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    bolts = ctx.state.setdefault("bolts", [])  # (t_struck, trunk_leds, fork_leds)

    if ctx.rng.random() < ctx.params["rate"] * ctx.dt:
        trunk, forks = strike(ctx.geometry, ctx.rng, ctx.params)
        bolts.append((ctx.t, _leds(trunk), [_leds(f) for f in forks]))
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


def strike(geo, rng, params) -> tuple[list, list[list]]:
    """One bolt: the trunk and its forks, each a list of
    ((edge, forward), end_junction) steps in travel order, every one
    heading down or sideways."""
    graph = geo.edge_graph()
    spent: set = set()  # edges used, with their seam partners: no span twice
    top = (geo.tile_rows, rng.randrange(1, geo.tile_cols))  # an interior corner on the top wall
    trunk = _descend(graph, top, params["length"], rng, params["turn_bias"], spent, first_down=True)

    forks = []
    corners = [junction for _, junction in trunk[:-1]]  # where each step ends; not the tip
    weights = np.array([max(0.0, (i + 1) / len(trunk) - QUIET_TOP) ** 2 for i in range(len(corners))])
    for _ in range(params["forks"]):
        if weights.sum() <= 0:
            break
        i = _weighted(rng, weights)
        weights[i] = 0.0  # one fork per corner
        remaining = len(trunk) - (i + 1)
        fork = _descend(graph, corners[i], max(2, remaining // 2 + 1), rng, min(1.0, params["turn_bias"] + FORK_JAGGED), spent)
        if fork:
            forks.append(fork)
    return trunk, forks


def _descend(graph, junction, length, rng, turn_bias, spent, first_down=False) -> list:
    """Up to `length` steps from `junction`, each down or (with
    probability `turn_bias` when both are open) sideways, never up.
    Returns [((edge, forward), end_junction), ...] in travel order."""
    steps = []
    for n in range(length):
        down, sideways = [], []
        for edge in graph.edges_at(junction):
            if edge in spent:
                continue
            forward = edge.junctions[0] == junction
            far = edge.junctions[1] if forward else edge.junctions[0]
            if far[0] < junction[0]:
                down.append((edge, forward, far))
            elif far[0] == junction[0] and not (first_down and n == 0):
                sideways.append((edge, forward, far))
        if not down and not sideways:
            break  # the bottom of the floor, or boxed in
        group = sideways if (sideways and (not down or rng.random() < turn_bias)) else down
        edge, forward, far = group[rng.randrange(len(group))]
        spent.add(edge)
        partner = graph.partner(edge)
        if partner is not None:
            spent.add(partner)
        steps.append((edge, forward, far))
        junction = far
        if junction[0] == 0:
            break  # grounded
    return [((edge, forward), end) for edge, forward, end in steps]


def _leds(path) -> np.ndarray:
    """Flat LED indices along a path, in travel order."""
    if not path:
        return np.zeros(0, dtype=np.intp)
    return np.concatenate([edge.flat_leds if forward else edge.flat_leds[::-1] for (edge, forward), _ in path])


def _weighted(rng, weights: np.ndarray) -> int:
    pick = rng.random() * weights.sum()
    return int(min(np.searchsorted(np.cumsum(weights), pick, side="right"), len(weights) - 1))
