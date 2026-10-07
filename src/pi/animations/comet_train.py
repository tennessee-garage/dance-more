"""Comet Train: comets nose to tail along every edge, pulsing across the floor and turning as one.

Every edge running one way - every horizontal side, or every vertical
one - carries a comet: a near-white head and a tail in a palette colour
behind it, fading to black by the 15th LED. A comet is as long as an edge,
so along each line the head of one comet sits at the tail of the next.

On every pulse (a beat, or a few) the whole train surges a few LEDs
forward, slamming into place and holding until the next pulse. Comets run
off the far side of the floor and new ones come in from the near side.

The step divides a tile side, so every few pulses the heads land exactly
on the tile corners. Now and then, from a corner, the next pulse turns
them instead: each head swings round the corner onto the edge leading
away at 90 degrees and runs along it, LED by LED, the tail following it
round, until the whole comet lies on the new edge. All of them turn at
once, the same way - up or down if they were running across, left or
right if they were running up or down - so a horizontal flow suddenly
pulses into a vertical one.

The floor is treated as a window onto an endless lattice of comets, which
is what makes the turn work. Every grid line has two lanes, one either
side of it (adjacent tiles don't share a line of LEDs), so two lanes of
comets arrive at each tile corner and two leave it. They turn like cars
in two lanes: the lane on the inside of the turn takes the inside lane
out. Comets turned off the floor go; edges that would be fed from off the
floor get new comets, sliding in from the side.

The state is which way the comets run, how far past the corners their
heads are (`offset`), and a palette position for each comet, kept by the
edge it set out from - including a ring of edges just off the floor, so a
comet half way on keeps its colour. Each pulse is drawn as one path per
comet - the edge it set out from, then the edge it is heading for, with -1
for LEDs off the floor - with the head slid along it.
"""

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.edges import Axis
from df2_pi.palette import choice, palette_param
from df2_pi.pixels import PixelFrame

WHITE = 0.8  # how far the head is pushed from its colour towards white
EASE = 3  # a step eases out: 1 - (1 - x)^EASE, so it leaves fast and lands soft


@animation(
    name="Comet Train",
    description="Comets nose to tail on every edge, pulsing across the floor on the beat and now and then turning a corner together.",
    author="df2",
    format="pixel",
    tags=["edges", "rhythm", "palette"],
    sync="beat",
    params={
        "tail": Param(int, default=14, min=1, max=14, label="Tail length (LEDs)", help="Behind the head: 14 makes each comet a whole tile side, 15 LEDs", role="scale"),
        "step": Param(int, default=3, choices=[1, 3, 5, 15], label="LEDs per pulse", help="Divides a tile side, so the heads keep landing on the corners"),
        "beats": Param(float, default=1.0, choices=[0.5, 1.0, 2.0, 4.0], label="Beats per pulse"),
        "move": Param(float, default=0.3, min=0.05, max=1.0, label="Step time", help="The part of each pulse spent stepping: short slams into place, 1 never stops"),
        "turns": Param(float, default=0.3, min=0.0, max=1.0, label="Turn chance", help="The chance, each time the heads reach the tile corners, that the next pulse turns them", role="variation", macro=1),
        "palette": palette_param(),
        "drift": Param(float, default=0.05, min=0.0, max=0.5, label="Colour drift", help="How far round the palette each new line of comets moves on"),
        "variety": Param(float, default=0.15, min=0.0, max=1.0, label="Colour variety", help="How much new comets' colours differ from each other"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    p = ctx.params
    state = ctx.state
    if not state:
        state["leds"] = {axis: _lane_leds(geo, axis) for axis in Axis}
        state["axis"] = Axis.X if ctx.rng.random() < 0.5 else Axis.Y
        state["dir"] = 1 if ctx.rng.random() < 0.5 else -1
        state["offset"] = 0  # LEDs the heads are past the corners
        state["base"] = ctx.rng.random()
        lanes, edges = state["leds"][state["axis"]].shape[:2]
        # As if it had been running: each comet an edge's drift behind the one ahead of it.
        behind = np.arange(-1, edges + 1)[:: state["dir"]] + 1  # 0 at the side they come in from
        state["u"] = state["base"] - p["drift"] * np.broadcast_to(behind, (lanes + 2, edges + 2)) + _jitter(ctx, (lanes + 2, edges + 2))
        state["pulse"] = None

    position = ctx.t_beats / p["beats"]
    pulse = int(np.floor(position))
    if pulse != state["pulse"]:
        if state["pulse"] is not None:
            _finish(ctx, state["move"])
        state["pulse"] = pulse
        turn = 0
        if state["offset"] == 0 and ctx.rng.random() < p["turns"]:
            turn = 1 if ctx.rng.random() < 0.5 else -1
        state["move"] = _plan(ctx, turn)

    move = state["move"]
    x = position - pulse
    if move["turn"]:
        # An even run round the corner, landing on the next beat (or the next pulse, if that comes sooner).
        progress = min(x * max(p["beats"], 1.0), 1.0)
    else:
        progress = 1.0 - (1.0 - min(x / p["move"], 1.0)) ** EASE
    head = move["from"] + (move["to"] - move["from"]) * progress
    return _draw(ctx, move, head)


def _lane_leds(geo, axis: Axis) -> np.ndarray:
    """(lanes, edges, n) flat LED indices for every edge running along
    `axis`, each ascending (west->east, south->north). Lane i is rail i: two
    per grid line, so lane i lies on line (i + 1) // 2, above it (or to its
    east) when i is even and below it (or to its west) when odd."""
    lanes = 2 * (geo.tile_rows if axis is Axis.X else geo.tile_cols)
    n = geo.leds_per_side
    return np.stack([geo.rails(axis, i).reshape(-1, n) for i in range(lanes)])


def _plan(ctx, turn: int) -> dict:
    """One pulse. `turn` is 0 to carry on, or the direction (+1 / -1)
    along the other axis to turn to. Every comet on, or partly on, the floor
    gets a path: the edge it set out from, then the edge it is heading for.
    Its head is at `n - 1 + offset` along that path and moves to `to`."""
    state = ctx.state
    axis, d = state["axis"], state["dir"]
    src = state["leds"][axis]
    lanes, edges, n = src.shape
    to_axis = axis if turn == 0 else (Axis.Y if axis is Axis.X else Axis.X)
    to_dir = d if turn == 0 else turn
    dst = state["leds"][to_axis]
    to_lanes, to_edges = dst.shape[:2]

    routes = []  # (source lane, edge, destination lane, edge), all within a ring of edges round the floor
    for i in range(-1, lanes + 1):
        for e in range(-1, edges + 1):
            if turn == 0:
                j, f = i, e + d
            else:
                line, side = (i + 1) // 2, 1 if i % 2 == 0 else -1
                corner = e + 1 if d > 0 else e  # the grid line the head is on, across the new axis
                # The lane on the inside of the turn (on the side it turns to) takes the inside lane out.
                out_side = -d if side == turn else d
                j = 2 * corner if out_side > 0 else 2 * corner - 1
                f = line if turn > 0 else line - 1
            if -1 <= j <= to_lanes and -1 <= f <= to_edges:
                routes.append((i, e, j, f))

    none = np.full(n, -1)
    paths, u = [], []
    for i, e, j, f in routes:
        on_src = 0 <= i < lanes and 0 <= e < edges
        on_dst = 0 <= j < to_lanes and 0 <= f < to_edges
        if not (on_src or on_dst):
            continue
        before = (src[i, e] if d > 0 else src[i, e][::-1]) if on_src else none
        after = (dst[j, f] if to_dir > 0 else dst[j, f][::-1]) if on_dst else none
        paths.append(np.concatenate([before, after]))
        u.append(state["u"][i + 1, e + 1])
    start = n - 1 + state["offset"]
    return {
        "paths": np.array(paths),
        "u": np.array(u, dtype=np.float64),
        "routes": routes,
        "turn": turn,
        "axis": to_axis,
        "dir": to_dir,
        "shape": (to_lanes, to_edges),
        "from": start,
        "to": 2 * n - 1 if turn else min(start + ctx.params["step"], 2 * n - 1),
    }


def _finish(ctx, move: dict) -> None:
    """The pulse is over. If the heads reached the next corners, every
    comet now sets out from the edge it was heading for."""
    state = ctx.state
    n = state["leds"][Axis.X].shape[2]
    offset = int(round(move["to"])) - (n - 1)
    if offset < n:
        state["offset"] = offset
        return
    lanes, edges = move["shape"]
    state["base"] += ctx.params["drift"]
    u = state["base"] + _jitter(ctx, (lanes + 2, edges + 2))  # for edges fed from beyond the ring: new comets
    for i, e, j, f in move["routes"]:
        u[j + 1, f + 1] = state["u"][i + 1, e + 1]
    state["u"], state["axis"], state["dir"], state["offset"] = u, move["axis"], move["dir"], 0


def _jitter(ctx, shape):
    return ctx.params["variety"] * (ctx.np_rng.random(shape) - 0.5)


def _draw(ctx, move: dict, head: float) -> PixelFrame:
    tail = ctx.params["tail"]
    colour = choice(ctx, ctx.params["palette"]).at(move["u"]).astype(np.float32)  # (comets, 3)
    hot = colour + (255.0 - colour) * WHITE
    behind = head - np.arange(move["paths"].shape[1])  # how far each LED of a path is behind the head
    # Sampled at fractional distances, so the comets glide between LEDs rather than stepping.
    head_w = np.clip(1.0 - np.abs(behind), 0.0, 1.0)
    tail_w = np.interp(behind, [0.0, 1.0, tail + 1.0], [0.0, 1.0, 0.0], left=0.0, right=0.0)
    lit = head_w[None, :, None] * hot[:, None, :] + tail_w[None, :, None] * colour[:, None, :]

    on = move["paths"] >= 0
    light = np.zeros((ctx.geometry.tiles * ctx.geometry.leds_per_tile, 3), dtype=np.float32)
    np.maximum.at(light, move["paths"][on], lit[on])
    frame = PixelFrame.black(ctx.geometry)
    frame.flat[:] = np.clip(light, 0, 255).astype(np.uint8)
    return frame
