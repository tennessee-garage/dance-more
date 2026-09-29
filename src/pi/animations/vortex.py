"""Vortex: comets circling every concentric ring of tile edges at once.

The tile edges that face the centre of the floor make concentric squares.
The outermost is `geo.floor_ring`, the chase track; inside it, every grid
line square contributes two rings, one per half of the seam (the tiles
outside it facing in, the tiles inside it facing out); and the four middle
tiles' inner edges form a cross, walked arm by arm as a pinwheel. On the
8x8 floor that is eight rings, 2,040 of the 3,840 LEDs. The edges running
radially between them stay dark.

Every ring is walked clockwise from its NW corner, so a fraction of the
way round one ring is the same angle on all of them. Each carries the same
number of comets, placed by angle - when the rings turn together the
comets line up into radial arms, and when they don't the arms wind into
spirals.

One ring is the lead, spun at the Speed param. The others are spun only by
their neighbours: each ring's rotation is pulled toward the rings either
side of it and bled away by friction, so the pull fades ring by ring away
from the lead and a ring keeps some momentum after the lead leaves it.
Pull sets how far the drag reaches. The lead itself works its way inward
one ring at a time, turns round at the middle, and back out.

Ring brightness follows how fast it is turning relative to the lead, so
the lead is the brightest ring and the stragglers dim.
"""

import colorsys
import math

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.geometry import Side
from df2_pi.pixels import PixelFrame

RING_TIME = 5.0  # seconds the lead spends on each ring, handover included
HANDOVER = 0.4  # the last fraction of RING_TIME, easing the lead to the next ring
COUPLING = 6.0  # 1/s: how hard neighbouring rings pull each other's rotation together
DRIVE = 60.0  # 1/s: how tightly the lead ring holds the Speed param
DIM = 0.3  # brightness of a ring standing still; the lead is 1.0
HUE_SPREAD = 0.06  # hue step from one ring to the next, outside in


@animation(
    name="Vortex",
    description="Comets on every concentric ring of edges, a lead ring dragging the rest round.",
    author="df2",
    format="pixel",
    tags=["edges"],
    params={
        "speed": Param(float, default=0.2, min=0.02, max=2.0, label="Lead ring speed (turns/s)", curve="log"),
        "comets": Param(int, default=4, min=1, max=12, label="Comets per ring", role="density"),
        "pull": Param(
            float, default=0.5, min=0.0, max=1.0, label="Pull",
            help="How far the lead ring's drag reaches: 0 barely, 1 nearly rigid",
            role="variation", macro=1,
        ),
        "trail": Param(float, default=0.5, min=0.05, max=1.0, label="Trail", help="As a fraction of the gap between comets"),
        "hue": Param(float, default=0.55, min=0.0, max=1.0, label="Hue"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    if not ctx.state:  # frame 0: lay every ring end to end, once
        rings = _rings(geo)
        lengths = np.array([len(r) for r in rings])
        ctx.state["leds"] = np.concatenate(rings)
        ctx.state["ring_of"] = np.repeat(np.arange(len(rings)), lengths)  # which ring each LED is on
        ctx.state["index"] = np.concatenate([np.arange(n) for n in lengths])  # and how far round it
        ctx.state["length"] = lengths[ctx.state["ring_of"]]
        ctx.state["omega"] = np.zeros(len(rings))  # turns/s, per ring
        ctx.state["theta"] = np.zeros(len(rings))  # turns, per ring
    omega, theta = ctx.state["omega"], ctx.state["theta"]
    count = len(omega)
    speed = ctx.params["speed"]

    # How much each ring is the lead right now: 1 on it, shared between two during a handover.
    lead = _lead_position(ctx.t, count)
    drive = np.clip(1.0 - np.abs(np.arange(count) - lead), 0.0, 1.0)

    # Each ring is pulled toward its neighbours' rotation and slowed by friction; the lead is held
    # at `speed`. Friction relative to coupling sets how fast the pull fades ring to ring: at
    # Pull 0.5 each ring out from the lead settles at about three quarters of the one before.
    friction = COUPLING * 10 ** (-2 * ctx.params["pull"])
    steps = max(1, math.ceil(ctx.dt * COUPLING * 4))  # small enough steps to stay stable
    h = ctx.dt / steps
    for _ in range(steps):
        pull = np.zeros(count)
        gap = np.diff(omega)
        pull[:-1] += gap
        pull[1:] -= gap
        omega += h * (COUPLING * pull - friction * omega)
        omega[:] = speed + (omega - speed) * np.exp(-DRIVE * drive * h)
        theta[:] = (theta + omega * h) % 1.0

    ring_of, length = ctx.state["ring_of"], ctx.state["length"]
    spacing = length / ctx.params["comets"]  # LEDs between comets on each ring
    behind = (theta[ring_of] * length - ctx.state["index"]) % spacing  # LEDs behind the nearest head
    level = np.clip(1.0 - behind / (ctx.params["trail"] * spacing), 0.0, 1.0)

    brightness = DIM + (1.0 - DIM) * np.clip(omega / speed, 0.0, 1.0)
    colours = np.array(
        [colorsys.hsv_to_rgb((ctx.params["hue"] + HUE_SPREAD * i) % 1.0, 1.0, 1.0) for i in range(count)]
    ) * (255 * brightness[:, None])

    frame = PixelFrame.black(geo)
    frame.flat[ctx.state["leds"]] = (level[:, None] * colours[ring_of]).astype(np.uint8)
    return frame


def _lead_position(t: float, count: int) -> float:
    """Which ring leads at time `t`, as a float: ring 0 (outermost) inward
    to ring count-1 and back, holding on each ring and easing between them."""
    bounce = 2 * (count - 1)

    def ring(step: int) -> int:
        k = step % bounce
        return k if k < count else bounce - k

    step, u = divmod(t / RING_TIME, 1.0)
    move = np.clip((u - (1.0 - HANDOVER)) / HANDOVER, 0.0, 1.0)
    move = move * move * (3 - 2 * move)
    a, b = ring(int(step)), ring(int(step) + 1)
    return a + (b - a) * move


def _rings(geo) -> list[np.ndarray]:
    """The concentric rings of centre-facing edges, outermost first, each
    as flat LED indices clockwise (as displayed) from its NW corner. Ring 0
    is `geo.floor_ring`; an even-sized floor ends with the middle cross."""
    size = geo.tile_rows
    if geo.tile_cols != size:
        raise ValueError(f"Vortex needs a square floor, not {geo.tile_rows}x{geo.tile_cols} tiles")

    def run(row: int, col: int, side: Side, reverse: bool = False) -> np.ndarray:
        leds = geo.edge(row * size + col, side).flat_leds  # west->east or south->north
        return leds[::-1] if reverse else leds

    rings = []
    for k in range((size + 1) // 2):
        lo, hi = k, size - 1 - k  # the square of tiles lo..hi on both axes
        if k > 0:  # the tiles just outside it, facing in
            rings.append(np.concatenate(
                [run(hi + 1, c, Side.SOUTH) for c in range(lo, hi + 1)]
                + [run(r, hi + 1, Side.WEST, True) for r in range(hi, lo - 1, -1)]
                + [run(lo - 1, c, Side.NORTH, True) for c in range(hi, lo - 1, -1)]
                + [run(r, lo - 1, Side.EAST) for r in range(lo, hi + 1)]
            ))
        rings.append(np.concatenate(  # its own outermost tiles, facing out
            [run(hi, c, Side.NORTH) for c in range(lo, hi + 1)]
            + [run(r, hi, Side.EAST, True) for r in range(hi, lo - 1, -1)]
            + [run(lo, c, Side.SOUTH, True) for c in range(hi, lo - 1, -1)]
            + [run(r, lo, Side.WEST) for r in range(lo, hi + 1)]
        ))
    if size % 2 == 0:
        # The middle four tiles' inner edges meet in a cross. Walk it clockwise as a pinwheel -
        # out along one half of each arm and back along the other - so, like the squares, an
        # eighth of the way round is due north, three eighths due east, and so on.
        m = size // 2
        rings.append(np.concatenate([
            run(m, m - 1, Side.EAST), run(m, m, Side.WEST, True),  # north arm
            run(m, m, Side.SOUTH), run(m - 1, m, Side.NORTH, True),  # east
            run(m - 1, m, Side.WEST, True), run(m - 1, m - 1, Side.EAST),  # south
            run(m - 1, m - 1, Side.NORTH, True), run(m, m - 1, Side.SOUTH),  # west
        ]))
    return rings
