"""Comet Train: comets nose to tail along every edge, pulsing across the floor and turning as one.

Every edge running one way - every horizontal side, or every vertical
one - carries a comet: a near-white head at the edge's far end and a tail
in a palette colour behind it, fading to black by the 15th LED. Each fills
its edge, so along each line the head of one comet sits at the tail of the
next.

On every pulse (a beat, or a few) the whole train surges forward one tile
side, slamming into place and then holding until the next pulse. Comets
run off the far side of the floor and new ones come in from the near side.

Now and then a pulse turns them instead: every comet runs round the tile
corner its head is sitting at and comes to rest on the edge leading away
at 90 degrees, so a horizontal flow suddenly pulses into a vertical one.
All of them turn at once, the same way - up or down if they were running
across, left or right if they were running up or down.

The floor is treated as a window onto an endless lattice of comets, which
is what makes the turn work. Every grid line has two lanes, one either
side of it (adjacent tiles don't share a line of LEDs), so two lanes of
comets arrive at each tile corner and two leave it. They turn like cars
in two lanes: the lane on the inside of the turn takes the inside lane
out. Comets turned off the floor go; edges that would be fed from off the
floor get new comets, sliding in from the side.

Between pulses the floor is exactly one comet per edge, so the state is
just which way they run and a palette position for each one; a pulse is
planned as one path per comet - its edge, then the edge it is going to,
with -1 for LEDs off the floor - and drawn by sliding the head along it.
"""

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.edges import Axis
from df2_pi.palette import choice, palette_param
from df2_pi.pixels import PixelFrame

WHITE = 0.8  # how far the head is pushed from its colour towards white
EASE = 3  # the move eases out: 1 - (1 - x)^EASE, so it leaves fast and lands soft


@animation(
    name="Comet Train",
    description="Comets nose to tail on every edge, pulsing across the floor on the beat and now and then turning 90 degrees together.",
    author="df2",
    format="pixel",
    tags=["edges", "rhythm", "palette"],
    sync="beat",
    params={
        "tail": Param(int, default=14, min=1, max=14, label="Tail length (LEDs)", help="Behind the head: 14 makes each comet a whole tile side, 15 LEDs", role="scale"),
        "beats": Param(float, default=1.0, choices=[0.5, 1.0, 2.0, 4.0], label="Beats per pulse"),
        "move": Param(float, default=0.3, min=0.05, max=1.0, label="Move time", help="The part of each pulse spent moving: short slams into place, 1 never stops"),
        "turns": Param(float, default=0.25, min=0.0, max=1.0, label="Turn chance", help="The chance each pulse turns the comets 90 degrees instead of carrying on", role="variation", macro=1),
        "palette": palette_param(),
        "drift": Param(float, default=0.05, min=0.0, max=0.5, label="Colour drift", help="How far round the palette each pulse's new comets move on"),
        "variety": Param(float, default=0.15, min=0.0, max=1.0, label="Colour variety", help="How much new comets' colours differ from each other"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    n = geo.leds_per_side
    p = ctx.params
    state = ctx.state
    if not state:
        state["leds"] = {axis: _lane_leds(geo, axis) for axis in Axis}
        state["axis"] = Axis.X if ctx.rng.random() < 0.5 else Axis.Y
        state["dir"] = 1 if ctx.rng.random() < 0.5 else -1
        state["base"] = ctx.rng.random()
        lanes, edges = state["leds"][state["axis"]].shape[:2]
        # As if it had been running: each comet a pulse's drift behind the one ahead of it.
        behind = np.arange(edges) if state["dir"] < 0 else np.arange(edges)[::-1]
        state["u"] = state["base"] - p["drift"] * np.broadcast_to(behind, (lanes, edges)) + _jitter(ctx, (lanes, edges))
        state["pulse"] = None

    position = ctx.t_beats / p["beats"]
    pulse = int(np.floor(position))
    if pulse != state["pulse"]:
        if state["pulse"] is not None:
            _arrive(state)
        state["pulse"] = pulse
        turn = 0
        if ctx.rng.random() < p["turns"]:
            turn = 1 if ctx.rng.random() < 0.5 else -1
        state["move"] = _plan(ctx, turn)

    progress = min((position - pulse) / p["move"], 1.0)
    head = (n - 1) + n * (1.0 - (1.0 - progress) ** EASE)  # along a 2n-LED path: the end of one edge to the end of the next
    return _draw(ctx, state["move"], head)


def _lane_leds(geo, axis: Axis) -> np.ndarray:
    """(lanes, edges, n) flat LED indices for every edge running along
    `axis`, each ascending (west->east, south->north). Lane i is rail i: two
    per grid line, so lane i lies on line (i + 1) // 2, above it (or to its
    east) when i is even and below it (or to its west) when odd."""
    lanes = 2 * (geo.tile_rows if axis is Axis.X else geo.tile_cols)
    n = geo.leds_per_side
    return np.stack([geo.rails(axis, i).reshape(-1, n) for i in range(lanes)])


def _plan(ctx, turn: int) -> dict:
    """One pulse: where every comet on (or about to come onto) the floor
    goes. `turn` is 0 to carry on, or the direction (+1 / -1) along the
    other axis to turn to."""
    state = ctx.state
    axis, d = state["axis"], state["dir"]
    src = state["leds"][axis]
    lanes, edges, n = src.shape
    to_axis = axis if turn == 0 else (Axis.Y if axis is Axis.X else Axis.X)
    to_dir = d if turn == 0 else turn
    dst = state["leds"][to_axis]
    to_lanes, to_edges = dst.shape[:2]

    moves = []  # (source lane, source edge, destination lane, destination edge)
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
            on_src = 0 <= i < lanes and 0 <= e < edges
            on_dst = 0 <= j < to_lanes and 0 <= f < to_edges
            if on_src or on_dst:
                moves.append((i, e, j, f, on_src, on_dst))

    state["base"] += ctx.params["drift"]
    none = np.full(n, -1)
    paths, u = [], []
    for i, e, j, f, on_src, on_dst in moves:
        before = (src[i, e] if d > 0 else src[i, e][::-1]) if on_src else none
        after = (dst[j, f] if to_dir > 0 else dst[j, f][::-1]) if on_dst else none
        paths.append(np.concatenate([before, after]))
        u.append(state["u"][i, e] if on_src else state["base"] + _jitter(ctx, ()))
    arriving = np.array([(j, f) for _, _, j, f, _, on_dst in moves if on_dst]).reshape(-1, 2)
    return {
        "paths": np.array(paths),
        "u": np.array(u, dtype=np.float64),
        "arriving": arriving,
        "keep": np.array([m[5] for m in moves]),
        "axis": to_axis,
        "dir": to_dir,
        "shape": (to_lanes, to_edges),
    }


def _arrive(state: dict) -> None:
    """The pulse is over: every comet is on its new edge."""
    move = state["move"]
    u = np.zeros(move["shape"])
    u[move["arriving"][:, 0], move["arriving"][:, 1]] = move["u"][move["keep"]]
    state["u"], state["axis"], state["dir"] = u, move["axis"], move["dir"]


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
