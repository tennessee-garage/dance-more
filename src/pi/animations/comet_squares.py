"""Squares that slam in side by side, hop round the floor, then spill apart.

A square is one tile's four sides. They arrive one at a time - top, left,
right, bottom - each as a comet running along the rail that side lies on
(`geo.rails`), in from whichever end of the floor is further away. A comet
picks up speed as it comes, hitting at twice its launch rate, and the
side's 15 LEDs only fill in once its head reaches the far end of the side:
then they land at full strength with a flash of white, while the rest of
the trail flows in behind and drains away.

The finished square then hops a few tiles across the grid, one whole
tile at a time, pausing on each.

Last, it comes apart. Each side breaks in the middle and its LEDs roll off
both ends like marbles, the ones nearest a corner first, speeding up as
they go. At every tile corner a marble takes a random edge whose far end
is further from the square than where it stands (`edge_graph().edges_at`),
so they fan out along the grid lines and never roll back in.

Each square is a small object in `ctx.state` with a phase and a clock that
advances by `ctx.dt`, so the show's Speed control and live param edits
both apply smoothly mid-flight.
"""

import colorsys

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.edges import Axis
from df2_pi.geometry import Side
from df2_pi.pixels import PixelFrame

ARRIVAL_ORDER = (Side.NORTH, Side.WEST, Side.EAST, Side.SOUTH)  # top, left, right, bottom
LAUNCH = 0.5  # comet speed at launch, as a fraction of the Comet speed param...
IMPACT = 2.0  # ...rising steadily to this at impact
FLASH = 0.18  # seconds a landed side glows white-hot
DWELL = 1.2  # seconds a square sits on each tile
STEPS = (2, 6)  # tiles a square hops before it spills, inclusive
MARBLE_LIFE = 1.4  # seconds from a marble rolling off to it fading out
MARBLE_STAGGER = 0.06  # seconds between marbles leaving the same corner
MARBLE_ACCEL = 60.0  # LEDs/s^2
MARBLE_EDGES = 5  # how many edges ahead a marble's route is planned


@animation(
    name="Comet Squares",
    description="Tile squares slam in side by side, hop round the floor, then spill apart like marbles.",
    author="df2",
    format="pixel",
    tags=["edges"],
    params={
        "squares": Param(int, default=2, min=1, max=8, label="Squares", role="density"),
        "speed": Param(float, default=240.0, min=30.0, max=900.0, label="Comet speed (LEDs/s)", curve="log"),
        "trail": Param(int, default=40, min=2, max=120, label="Trail length", macro=1),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    if not ctx.state:
        ctx.state["rails"] = {}  # (axis, index) -> flat LED indices, gathered on first use
        ctx.state["squares"] = []
        ctx.state["hue"] = ctx.rng.random()
    squares = ctx.state["squares"]
    squares[:] = [s for s in squares if s.phase != "done"]
    while len(squares) < ctx.params["squares"]:
        # Stagger the opening squares so they don't form in lockstep.
        delay = len(squares) * 0.9 + ctx.rng.uniform(0, 0.4) if ctx.frame == 0 else ctx.rng.uniform(0.2, 1.0)
        squares.append(_spawn(ctx, squares, delay))

    frame = PixelFrame.black(ctx.geometry)
    for square in squares:
        square.update(ctx, squares)
        square.draw(frame, ctx)
    return frame


class Square:
    def __init__(self, row: int, col: int, colour: np.ndarray, delay: float) -> None:
        self.row, self.col = row, col
        self.colour = colour
        self.phase = "wait"  # wait -> form -> move -> spill -> done
        self.clock = -delay  # seconds into the current phase
        self.comets: list[list] = []  # [path, head, landed_at] per side, in ARRIVAL_ORDER
        self.came_from: tuple[int, int] | None = None
        self.steps_left = 0
        self.marbles: dict | None = None

    def update(self, ctx, squares: list["Square"]) -> None:
        n = ctx.geometry.leds_per_side
        speed, trail = ctx.params["speed"], ctx.params["trail"]
        self.clock += ctx.dt

        if self.phase == "wait" and self.clock >= 0:
            self.phase = "form"
            self.comets = [[_comet_path(ctx, ARRIVAL_ORDER[0], self.row, self.col), 0.0, None]]

        elif self.phase == "form":
            for comet in self.comets:
                path, head, landed_at = comet
                progress = min(head / (len(path) - 1), 1.0)
                comet[1] += speed * (LAUNCH + (IMPACT - LAUNCH) * progress) * ctx.dt
                if landed_at is None and comet[1] >= len(path) - 1:
                    comet[2] = self.clock
            path, head, landed_at = self.comets[-1]
            if landed_at is not None and len(self.comets) < len(ARRIVAL_ORDER):
                # The next side sets off as this one lands.
                side = ARRIVAL_ORDER[len(self.comets)]
                self.comets.append([_comet_path(ctx, side, self.row, self.col), 0.0, None])
            elif (
                len(self.comets) == len(ARRIVAL_ORDER)
                and landed_at is not None
                and self.clock - landed_at >= FLASH
                and all(h >= len(p) - n - 1 + trail for p, h, _ in self.comets)  # every trail has drained in
            ):
                self.phase, self.clock = "move", 0.0
                self.comets = []
                self.steps_left = ctx.rng.randint(*STEPS)

        elif self.phase == "move" and self.clock >= DWELL:
            self.clock = 0.0
            if self.steps_left == 0:
                self.phase = "spill"
                self.marbles = _spill(ctx, self.row, self.col)
            else:
                self.came_from = (self.row, self.col)
                self.row, self.col = self._next_tile(ctx, squares)
                self.steps_left -= 1

        elif self.phase == "spill":
            m = self.marbles
            rolling = self.clock >= m["release"]
            m["v"] += np.where(rolling, MARBLE_ACCEL * ctx.dt, 0.0)
            m["s"] = np.minimum(m["s"] + m["v"] * ctx.dt, m["length"] - 1)
            if self.clock >= m["release"].max() + MARBLE_LIFE:
                self.phase = "done"

    def _next_tile(self, ctx, squares: list["Square"]) -> tuple[int, int]:
        """A neighbouring tile: not back where it came from, and not one
        another square is on, when there is a choice."""
        geo = ctx.geometry
        options = [
            (self.row + dr, self.col + dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= self.row + dr < geo.tile_rows and 0 <= self.col + dc < geo.tile_cols
        ]
        taken = {(s.row, s.col) for s in squares if s is not self}
        onward = [t for t in options if t != self.came_from] or options
        free = [t for t in onward if t not in taken] or onward
        return free[ctx.rng.randrange(len(free))]

    def draw(self, frame: PixelFrame, ctx) -> None:
        geo = ctx.geometry
        n = geo.leds_per_side
        tile = self.row * geo.tile_cols + self.col

        if self.phase == "form":
            trail = ctx.params["trail"]
            for path, head, landed_at in self.comets:
                i = np.arange(len(path))
                level = np.clip(1.0 - (head - i) / trail, 0.0, 1.0) * (i <= head)  # bright head, fading tail
                if landed_at is None:
                    _paint(frame, path, level, self.colour)
                    continue
                # Landed: the side is lit whole, white-hot for a moment, while the trail drains in.
                _paint(frame, path[:-n], level[:-n], self.colour)
                heat = max(0.0, 1.0 - (self.clock - landed_at) / FLASH)
                _paint(frame, path[-n:], 1.0, self.colour + (255.0 - self.colour) * 0.7 * heat)

        elif self.phase == "move":
            for side in ARRIVAL_ORDER:
                _paint(frame, geo.edge(tile, side).flat_leds, 1.0, self.colour)

        elif self.phase == "spill":
            m = self.marbles
            age = self.clock - m["release"]
            level = np.where(age < 0, 1.0, np.clip(1.0 - age / MARBLE_LIFE, 0.0, 1.0))
            # Each marble straddles two LEDs by its fractional position, so it rolls rather than hops.
            at = np.floor(m["s"]).astype(np.intp)
            frac = m["s"] - at
            rows = np.arange(len(at))
            leds = np.concatenate([m["paths"][rows, at], m["paths"][rows, np.minimum(at + 1, m["length"] - 1)]])
            levels = np.concatenate([level * (1 - frac), level * frac])
            np.maximum.at(frame.flat, leds, np.multiply.outer(levels, self.colour).astype(np.uint8))


def _spawn(ctx, squares: list[Square], delay: float) -> Square:
    geo = ctx.geometry
    taken = {(s.row, s.col) for s in squares}
    free = [(r, c) for r in range(geo.tile_rows) for c in range(geo.tile_cols) if (r, c) not in taken]
    row, col = free[ctx.rng.randrange(len(free))]
    ctx.state["hue"] = (ctx.state["hue"] + 0.618034) % 1.0  # golden-ratio steps keep neighbours apart
    colour = np.array(colorsys.hsv_to_rgb(ctx.state["hue"], 1.0, 1.0), dtype=np.float32) * 255
    return Square(row, col, colour, delay)


def _comet_path(ctx, side: Side, row: int, col: int) -> np.ndarray:
    """A comet's route in to one side of tile (row, col): along the rail
    that side lies on, from the far end of the floor, ending on the side's
    last LED so its final 15 are the side. Rails run west->east /
    south->north, one tile's side after another."""
    geo = ctx.geometry
    n = geo.leds_per_side
    if side in (Side.NORTH, Side.SOUTH):
        key, offset = (Axis.X, 2 * row + (side is Side.NORTH)), col * n
    else:
        key, offset = (Axis.Y, 2 * col + (side is Side.EAST)), row * n
    rails = ctx.state["rails"]
    if key not in rails:
        rails[key] = geo.rails(*key)
    rail = rails[key]
    if offset < len(rail) // 2:  # nearer the low end, so come in from the high end
        return rail[offset:][::-1]
    return rail[: offset + n]


def _spill(ctx, row: int, col: int) -> dict:
    """Break the square at (row, col) into marbles: every LED of every side
    rolls to the nearer end of its side, then out from that corner. Returns
    the marbles as padded arrays so a frame is a few numpy expressions."""
    geo = ctx.geometry
    graph = geo.edge_graph()
    n = geo.leds_per_side
    tile = row * geo.tile_cols + col
    centre = (row + 0.5, col + 0.5)  # on the junction lattice, where tile corners are integers
    paths, release = [], []
    for side in ARRIVAL_ORDER:
        edge = geo.edge(tile, side)
        for i in range(n):
            if i > n // 2 or (i == n // 2 and ctx.rng.random() < 0.5):
                along, corner, from_corner = edge.flat_leds[i:], edge.junctions[1], n - 1 - i
            else:
                along, corner, from_corner = edge.flat_leds[i::-1], edge.junctions[0], i
            paths.append(np.concatenate([along, *_roll_away(graph, corner, centre, ctx.rng)]))
            release.append((from_corner + ctx.rng.random()) * MARBLE_STAGGER)
    length = np.array([len(p) for p in paths])
    return {
        "paths": np.array([np.pad(p, (0, length.max() - len(p)), mode="edge") for p in paths]),
        "length": length,
        "release": np.array(release),
        "s": np.zeros(len(paths)),  # position along the path, in LEDs
        "v": np.zeros(len(paths)),
    }


def _roll_away(graph, junction, centre, rng) -> list[np.ndarray]:
    """A marble's route out from a tile corner: up to MARBLE_EDGES edges,
    each ending further from `centre` than it started. Stops early at the
    floor's corner, where nothing leads further out."""

    def distance(j):
        return (j[0] - centre[0]) ** 2 + (j[1] - centre[1]) ** 2

    route = []
    for _ in range(MARBLE_EDGES):
        onward = []
        for edge in graph.edges_at(junction):
            forward = edge.junctions[0] == junction
            far = edge.junctions[1] if forward else edge.junctions[0]
            if distance(far) > distance(junction):
                onward.append((edge, forward, far))
        if not onward:
            break
        edge, forward, junction = onward[rng.randrange(len(onward))]
        route.append(edge.flat_leds if forward else edge.flat_leds[::-1])
    return route


def _paint(frame: PixelFrame, leds: np.ndarray, level, colour: np.ndarray) -> None:
    """Max-blend `colour` scaled by `level` (a scalar or one per LED) onto
    `leds`, so overlapping squares and trails don't darken each other."""
    lit = np.multiply.outer(np.broadcast_to(np.asarray(level, dtype=np.float32), leds.shape), colour)
    frame.flat[leds] = np.maximum(frame.flat[leds], lit.astype(np.uint8))
