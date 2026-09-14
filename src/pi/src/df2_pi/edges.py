"""Edge-aware access: the floor as 256 line segments, not a grid of pixels.

`PixelFrame.from_grid` handles the image case - render a picture and keep the
lit cells. This module handles the STRUCTURAL case: the floor is 64 tiles
with 15 LEDs along each side, so it is really 256 runs of 15, and an
animation that wants to travel along them (a chase around the boundary, a
lightning bolt crossing the floor by the edges, a wave rolling across the
grid lines) should be able to say so.

    Edge         one side of one tile: 15 ordered LEDs with a position and
                 direction in floor coordinates
    geo.edges    all 256 - 32 on the floor's outer boundary, 224 interior
    geo.seams    the 112 facing pairs of interior edges
    geo.floor_ring   the outer boundary as one 480-LED ordered ring
    geo.tile_ring(t) one tile's 60 LEDs, clockwise from its NW corner
    geo.rails(axis, i)  a straight 120-LED line across the whole floor
    EdgeGraph    edges as nodes, adjacent where they meet at a tile corner;
                 random walks, shortest paths and forks for the bolt

Ordering is in FLOOR orientation, never chain orientation. Every horizontal
edge runs west->east and every vertical edge runs south->north - ascending
x and ascending y - regardless of which way the WS2815 chain winds on that
side. That is what lets an effect run smoothly across tile boundaries
instead of zigzagging at each one; `FloorGeometry.side_leds` already orders
this way and edges are built from it.

Seams exist because two adjacent tiles do not share a line of LEDs: they
present two parallel runs of 15 facing each other across the tile frames,
one cell apart on the grid. An animation drawing "the grid lines" needs
both halves; one drawing a wave crossing a boundary needs to know they are
the same place. `seams` pairs them up so that `a.leds[i]` faces `b.leds[i]`.

LED indexing. `Edge.leds` are chain indices WITHIN the tile (0-59), like
`side_leds`. Anything that spans tiles - `Edge.flat_leds`, `floor_ring`,
`rails`, `Path.leds()` - uses FLAT indices, `tile * 60 + led`, which index
straight into `PixelFrame.flat` (the `(3840, 3)` view of a frame):

    frame.flat[geo.floor_ring[(i + phase) % 480]] = colour

Coordinates. `Edge.start`, `.end`, `.direction` and everything in paint.py
are continuous (x, y) pairs in cell units, x along the row (east) and y
across rows (north), because that is how animators think about geometry.
The integer lookup tables on `FloorGeometry` (`led_to_cell`, `tile_at`) are
(y, x) because that is how arrays index. Swap when crossing between them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from df2_pi.geometry import FloorGeometry, Side

Junction = tuple[int, int]
"""A tile corner, `(row_line, col_line)` on the (rows+1) x (cols+1) lattice
of tile-corner lines: tile (r, c) has its SW corner at (r, c) and its NE
corner at (r+1, c+1). Where four tiles meet, their four corner cells are
distinct dark cells but one junction."""


class Axis(Enum):
    """Which way an edge or rail runs. X is along a row (a NORTH or SOUTH
    edge, west->east); Y is across rows (an EAST or WEST edge,
    south->north)."""

    X = "x"
    Y = "y"


# Clockwise from north, as displayed - the order edges are numbered within a
# tile, so `edge.index == tile * 4 + SIDES.index(side)`.
SIDES = (Side.NORTH, Side.EAST, Side.SOUTH, Side.WEST)

_AXIS_OF = {Side.NORTH: Axis.X, Side.SOUTH: Axis.X, Side.EAST: Axis.Y, Side.WEST: Axis.Y}


@dataclass(frozen=True, eq=False)
class Edge:
    """One side of one tile: `leds_per_side` LEDs ordered along `direction`.

    `leds` are chain indices within `tile`; `flat_leds` index into
    `PixelFrame.flat`. `start` and `end` are the (x, y) cell-centre
    positions of the first and last LED, `direction` is the unit vector
    from one to the other. `junctions` are the tile corners at the start
    and end of the run, in that order - the graph is built on them.

    Equality and hashing are by (tile, side), so edges from the same
    geometry can be dictionary keys and set members.
    """

    index: int
    tile: int
    side: Side
    leds: np.ndarray
    flat_leds: np.ndarray
    start: tuple[float, float]
    end: tuple[float, float]
    direction: tuple[float, float]
    junctions: tuple[Junction, Junction]
    outer: bool

    @property
    def axis(self) -> Axis:
        return _AXIS_OF[self.side]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.tile == other.tile and self.side is other.side

    def __hash__(self) -> int:
        return hash((self.tile, self.side))

    def __repr__(self) -> str:
        return f"Edge(tile={self.tile}, side={self.side.name})"


def build_edges(geo: FloorGeometry) -> tuple[Edge, ...]:
    """All of a geometry's edges, numbered `tile * 4 + SIDES.index(side)`."""
    edges: list[Edge] = []
    last_row, last_col = geo.tile_rows - 1, geo.tile_cols - 1
    for tile in range(geo.tiles):
        row, col = divmod(tile, geo.tile_cols)
        for side in SIDES:
            leds = geo.side_leds(tile, side)
            first = geo.led_positions[tile, leds[0]]  # (y, x)
            last = geo.led_positions[tile, leds[-1]]
            if side is Side.NORTH:
                junctions = ((row + 1, col), (row + 1, col + 1))
                outer = row == last_row
            elif side is Side.SOUTH:
                junctions = ((row, col), (row, col + 1))
                outer = row == 0
            elif side is Side.EAST:
                junctions = ((row, col + 1), (row + 1, col + 1))
                outer = col == last_col
            else:
                junctions = ((row, col), (row + 1, col))
                outer = col == 0
            direction = (1.0, 0.0) if _AXIS_OF[side] is Axis.X else (0.0, 1.0)
            edges.append(
                Edge(
                    index=len(edges),
                    tile=tile,
                    side=side,
                    leds=leds,
                    flat_leds=tile * geo.leds_per_tile + leds,
                    start=(float(first[1]), float(first[0])),
                    end=(float(last[1]), float(last[0])),
                    direction=direction,
                    junctions=junctions,
                    outer=outer,
                )
            )
    return tuple(edges)


def edge_of(geo: FloorGeometry, tile: int, side: Side) -> Edge:
    return geo.edges[tile * len(SIDES) + SIDES.index(side)]


def build_seams(geo: FloorGeometry) -> tuple[tuple[Edge, Edge], ...]:
    """The facing pairs of interior edges: (EAST of a tile, WEST of its
    eastern neighbour) and (NORTH of a tile, SOUTH of its northern
    neighbour). Both edges of a pair run the same way, so `a.leds[i]`
    faces `b.leds[i]` one cell away."""
    seams: list[tuple[Edge, Edge]] = []
    for tile in range(geo.tiles):
        row, col = divmod(tile, geo.tile_cols)
        if col + 1 < geo.tile_cols:
            seams.append((edge_of(geo, tile, Side.EAST), edge_of(geo, tile + 1, Side.WEST)))
        if row + 1 < geo.tile_rows:
            north = tile + geo.tile_cols
            seams.append((edge_of(geo, tile, Side.NORTH), edge_of(geo, north, Side.SOUTH)))
    return tuple(seams)


def tile_ring(geo: FloorGeometry, tile: int) -> np.ndarray:
    """One tile's LEDs as a `(leds_per_tile,)` ring of chain indices,
    clockwise as displayed starting at the NW corner: across the top
    west->east, down the right, back along the bottom, up the left."""
    return np.concatenate(
        [
            edge_of(geo, tile, Side.NORTH).leds,
            edge_of(geo, tile, Side.EAST).leds[::-1],
            edge_of(geo, tile, Side.SOUTH).leds[::-1],
            edge_of(geo, tile, Side.WEST).leds,
        ]
    )


def build_floor_ring(geo: FloorGeometry) -> np.ndarray:
    """The floor's outer boundary as one ordered ring of flat LED indices,
    clockwise as displayed from the floor's NW corner. No LED is counted
    twice because corner cells are dark; consecutive entries are
    neighbours except where the ring steps across the two dark corner
    cells between adjacent tiles (a gap of 3 cells)."""
    rows, cols = geo.tile_rows, geo.tile_cols
    top = [edge_of(geo, (rows - 1) * cols + c, Side.NORTH).flat_leds for c in range(cols)]
    right = [
        edge_of(geo, r * cols + cols - 1, Side.EAST).flat_leds[::-1] for r in reversed(range(rows))
    ]
    bottom = [edge_of(geo, c, Side.SOUTH).flat_leds[::-1] for c in reversed(range(cols))]
    left = [edge_of(geo, r * cols, Side.WEST).flat_leds for r in range(rows)]
    return np.concatenate(top + right + bottom + left)


def rail(geo: FloorGeometry, axis: Axis, index: int) -> np.ndarray:
    """A straight line of LEDs across the whole floor, as flat indices in
    ascending order along `axis`. Each tile row contributes two X rails
    (its SOUTH line, then its NORTH line) and each tile column two Y rails
    (WEST, then EAST), so `index` runs 0..2*rows-1 or 0..2*cols-1,
    ascending with the rail's position on the floor."""
    axis = Axis(axis)
    if axis is Axis.X:
        count, per = geo.tile_rows, geo.tile_cols
        near, far = Side.SOUTH, Side.NORTH
    else:
        count, per = geo.tile_cols, geo.tile_rows
        near, far = Side.WEST, Side.EAST
    if not 0 <= index < 2 * count:
        raise ValueError(f"{axis.name} rail index must be 0..{2 * count - 1}, got {index}")
    line, side = divmod(index, 2)
    side = far if side else near
    if axis is Axis.X:
        tiles = [line * geo.tile_cols + c for c in range(per)]
    else:
        tiles = [r * geo.tile_cols + line for r in range(per)]
    return np.concatenate([edge_of(geo, t, side).flat_leds for t in tiles])


# ---- the graph ------------------------------------------------------------------

# Extension attempts a walk may spend backtracking before giving up. Dead
# ends are rare and shallow (a wall, a corner the walk boxed itself into),
# so real walks use a few dozen; this only stops a pathological request
# from turning into an exhaustive search.
_SEARCH_BUDGET = 20_000


@dataclass(frozen=True)
class Step:
    """One edge of a path and the direction it is travelled: `forward`
    means along the edge's own (ascending) ordering."""

    edge: Edge
    forward: bool

    @property
    def leds(self) -> np.ndarray:
        """Flat LED indices in travel order."""
        return self.edge.flat_leds if self.forward else self.edge.flat_leds[::-1]

    @property
    def start_junction(self) -> Junction:
        return self.edge.junctions[0 if self.forward else 1]

    @property
    def end_junction(self) -> Junction:
        return self.edge.junctions[1 if self.forward else 0]

    @property
    def heading(self) -> tuple[float, float]:
        dx, dy = self.edge.direction
        return (dx, dy) if self.forward else (-dx, -dy)


@dataclass(frozen=True)
class Path:
    """A connected sequence of edges with a travel direction on each -
    the output of `EdgeGraph.walk`, `shortest_path` and `branch`.

    `leds()` is the whole path as flat LED indices in travel order with no
    duplicates, so a head-to-tail gradient is just an index into it.
    """

    steps: tuple[Step, ...]
    geometry: FloorGeometry

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(step.edge for step in self.steps)

    def leds(self) -> np.ndarray:
        """Flat LED indices in travel order."""
        if not self.steps:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([step.leds for step in self.steps])

    def positions(self) -> np.ndarray:
        """`(n, 2)` float (x, y) cell-centre positions of `leds()`, in
        travel order - for effects that shade by distance along the walk."""
        yx = self.geometry.led_positions.reshape(-1, 2)[self.leds()]
        return yx[:, ::-1]

    def junctions(self) -> list[Junction]:
        """The tile corners the path passes through, `len(path) + 1` of them."""
        if not self.steps:
            return []
        return [self.steps[0].start_junction] + [step.end_junction for step in self.steps]


class EdgeGraph:
    """Edges as nodes, adjacent where they meet at a tile corner.

    Two edges are neighbours when they share exactly one junction. An
    edge's seam partner shares both and is not a neighbour - it is the
    same span on the other side of the frame, and stepping onto it is a
    U-turn, not a continuation. Where four tiles meet, eight edges do:
    the two on each tile that touch that corner.

    A walk is self-avoiding (no edge twice, so every LED in the result is
    distinct) and picks its way with `turn_bias`: at a junction offering
    both a straight continuation and a turn, it turns with that
    probability. 0 runs straight whenever it can; 1 turns at every chance.
    "Straight" includes the parallel run one cell over, so a bolt going up
    a seam may weave across it.
    """

    def __init__(self, geometry: FloorGeometry) -> None:
        self.geometry = geometry
        self.edges = geometry.edges
        self._partner: dict[Edge, Edge] = {}
        for a, b in geometry.seams:
            self._partner[a] = b
            self._partner[b] = a
        self._at_junction: dict[Junction, list[Edge]] = {}
        for edge in self.edges:
            for junction in edge.junctions:
                self._at_junction.setdefault(junction, []).append(edge)

    def partner(self, edge: Edge) -> Edge | None:
        """The facing edge across the seam, or None on the outer boundary."""
        return self._partner.get(edge)

    def edges_at(self, junction: Junction) -> tuple[Edge, ...]:
        """Every edge with an end at this tile corner."""
        return tuple(self._at_junction.get(junction, ()))

    def neighbors(self, edge: Edge) -> tuple[Edge, ...]:
        """Edges meeting this one at either end (sharing exactly one
        junction - never itself, never its seam partner)."""
        partner = self.partner(edge)
        out: list[Edge] = []
        for junction in edge.junctions:
            for other in self._at_junction[junction]:
                if other != edge and other != partner and other not in out:
                    out.append(other)
        return tuple(out)

    def random_edge(self, rng: np.random.Generator) -> Edge:
        return self.edges[int(rng.integers(len(self.edges)))]

    # ---- walks --------------------------------------------------------------------

    def walk(
        self,
        start: Edge,
        length: int,
        rng: np.random.Generator,
        turn_bias: float = 0.3,
        forward: bool | None = None,
    ) -> Path:
        """A random self-avoiding walk of `length` edges beginning with
        `start`, travelled forward or backward along it (`forward=None`:
        the rng decides). Backtracks internally when it walks into a dead
        end, so the result has exactly `length` edges whenever any such
        walk exists. Raises ValueError if none can (more edges than the
        graph has) or if the search gives up - bolts are a few dozen edges
        long; a walk approaching the size of the graph is a Hamiltonian
        path search, which this is not."""
        if length < 1:
            raise ValueError(f"walk length must be at least 1, got {length}")
        if length > len(self.edges):
            raise ValueError(f"walk length {length} exceeds the {len(self.edges)} edges")
        if not 0.0 <= turn_bias <= 1.0:
            raise ValueError(f"turn_bias must be in 0..1, got {turn_bias}")
        if forward is None:
            forward = bool(rng.integers(2))
        steps = [Step(start, forward)]
        if not self._extend(steps, length, rng, turn_bias, {start}, [_SEARCH_BUDGET]):
            raise ValueError(f"no self-avoiding walk of {length} edges found from {start!r}")
        return Path(tuple(steps), self.geometry)

    def _extend(
        self,
        steps: list[Step],
        length: int,
        rng: np.random.Generator,
        turn_bias: float,
        visited: set[Edge],
        budget: list[int],
    ) -> bool:
        if len(steps) == length:
            return True
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        current = steps[-1]
        junction = current.end_junction
        partner = self.partner(current.edge)
        candidates = [
            e for e in self._at_junction[junction] if e not in visited and e != partner
        ]
        straight = [e for e in candidates if e.axis is current.edge.axis]
        turns = [e for e in candidates if e.axis is not current.edge.axis]
        if straight and turns:
            first, second = (turns, straight) if rng.random() < turn_bias else (straight, turns)
        else:
            first, second = straight or turns, []
        for group in (first, second):
            for i in rng.permutation(len(group)):
                edge = group[int(i)]
                step = Step(edge, forward=edge.junctions[0] == junction)
                steps.append(step)
                visited.add(edge)
                if self._extend(steps, length, rng, turn_bias, visited, budget):
                    return True
                steps.pop()
                visited.discard(edge)
        return False

    def shortest_path(self, a: Edge, b: Edge) -> Path:
        """The fewest-edges path from `a` to `b` inclusive (breadth-first
        over `neighbors`), oriented in travel order."""
        previous: dict[Edge, Edge | None] = {a: None}
        queue: deque[Edge] = deque([a])
        while queue and b not in previous:
            edge = queue.popleft()
            for other in self.neighbors(edge):
                if other not in previous:
                    previous[other] = edge
                    queue.append(other)
        if b not in previous:
            raise ValueError(f"no path from {a!r} to {b!r}")
        edges: list[Edge] = []
        cursor: Edge | None = b
        while cursor is not None:
            edges.append(cursor)
            cursor = previous[cursor]
        edges.reverse()
        return Path(tuple(_orient(edges)), self.geometry)

    def branch(
        self,
        path: Path,
        rng: np.random.Generator,
        n: int = 2,
        length: int | None = None,
        turn_bias: float = 0.5,
    ) -> list[Path]:
        """Fork `n` new walks off `path` - lightning forks. Each leaves from
        a random junction along the path (never its tip), onto an edge the
        path does not use, and avoids the path's edges and their seam
        partners thereafter. `length` defaults to half the parent's. Fewer
        than `n` are returned only if the path leaves nowhere to fork
        from."""
        if length is None:
            length = max(1, len(path) // 2)
        taken = set(path.edges)
        for edge in path.edges:
            partner = self.partner(edge)
            if partner is not None:
                taken.add(partner)
        branches: list[Path] = []
        junctions = path.junctions()[:-1]
        for _ in range(n):
            for j in rng.permutation(len(junctions)):
                junction = junctions[int(j)]
                exits = [e for e in self._at_junction[junction] if e not in taken]
                if not exits:
                    continue
                edge = exits[int(rng.integers(len(exits)))]
                steps = [Step(edge, forward=edge.junctions[0] == junction)]
                visited = taken | {edge}
                if self._extend(steps, length, rng, turn_bias, visited, [_SEARCH_BUDGET]):
                    branch = Path(tuple(steps), self.geometry)
                    branches.append(branch)
                    taken |= set(branch.edges)
                    break
        return branches


def _orient(edges: list[Edge]) -> list[Step]:
    """Turn a connected edge sequence into steps with travel directions:
    each edge is travelled toward the junction it shares with the next."""
    if len(edges) == 1:
        return [Step(edges[0], True)]
    steps: list[Step] = []
    for i, edge in enumerate(edges):
        if i + 1 < len(edges):
            shared = set(edge.junctions) & set(edges[i + 1].junctions)
            forward = edge.junctions[1] in shared
        else:
            shared = set(edge.junctions) & set(edges[i - 1].junctions)
            forward = edge.junctions[0] in shared
        steps.append(Step(edge, forward))
    return steps
