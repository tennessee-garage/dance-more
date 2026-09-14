import numpy as np
import pytest

from df2_pi.edges import SIDES, Axis, Edge, EdgeGraph, Path
from df2_pi.geometry import FloorGeometry, Side
from df2_pi.pixels import PixelFrame, default_geometry


@pytest.fixture(scope="module")
def geo() -> FloorGeometry:
    return default_geometry()


@pytest.fixture(scope="module")
def graph(geo) -> EdgeGraph:
    return geo.edge_graph()


def cell_of(geo, flat: int) -> tuple[int, int]:
    """(y, x) of a flat LED index."""
    tile, led = divmod(int(flat), geo.leds_per_tile)
    return tuple(int(v) for v in geo.led_to_cell[tile, led])


def chebyshev(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# ---- edges ----------------------------------------------------------------------


def test_every_lit_led_belongs_to_exactly_one_edge(geo):
    assert len(geo.edges) == 4 * geo.tiles == 256
    seen = np.zeros(geo.led_count, dtype=int)
    for edge in geo.edges:
        assert edge.leds.shape == (geo.leds_per_side,) == (15,)
        seen[edge.flat_leds] += 1
    assert (seen == 1).all()


def test_edges_are_numbered_by_tile_then_side_clockwise_from_north(geo):
    for i, edge in enumerate(geo.edges):
        assert edge.index == i
        assert edge.tile == i // 4
        assert edge.side is SIDES[i % 4]
        assert geo.edge(edge.tile, edge.side) is edge


def test_edge_leds_are_the_geometry_side_run_and_flat_indices_match(geo):
    for edge in geo.edges:
        np.testing.assert_array_equal(edge.leds, geo.side_leds(edge.tile, edge.side))
        np.testing.assert_array_equal(edge.flat_leds, edge.tile * geo.leds_per_tile + edge.leds)


def test_horizontal_edges_run_west_to_east_and_vertical_south_to_north(geo):
    # The anti-zigzag guarantee: ordering is by floor position, never by
    # which way the chain winds on that side (which differs between the
    # two horizontal sides and between the two vertical sides).
    for edge in geo.edges:
        cells = geo.led_to_cell[edge.tile, edge.leds]
        if edge.side in (Side.NORTH, Side.SOUTH):
            assert edge.axis is Axis.X and edge.direction == (1.0, 0.0)
            assert len(set(cells[:, 0].tolist())) == 1
            assert (np.diff(cells[:, 1]) == 1).all()
        else:
            assert edge.axis is Axis.Y and edge.direction == (0.0, 1.0)
            assert len(set(cells[:, 1].tolist())) == 1
            assert (np.diff(cells[:, 0]) == 1).all()


def test_edge_start_and_end_are_xy_cell_centres_of_first_and_last_led(geo):
    edge = geo.edge(0, Side.NORTH)
    assert edge.start == (1.5, 16.5)
    assert edge.end == (15.5, 16.5)
    edge = geo.edge(9, Side.EAST)  # tile (1, 1): block origin (17, 17)
    assert edge.start == (17 + 16.5, 17 + 1.5)
    assert edge.end == (17 + 16.5, 17 + 15.5)


def test_outer_and_interior_edge_counts(geo):
    outer = [e for e in geo.edges if e.outer]
    assert len(outer) == 32
    assert len(geo.edges) - len(outer) == 224
    for edge in outer:
        # an outer edge's LEDs sit on the floor boundary
        cells = geo.led_to_cell[edge.tile, edge.leds]
        on_boundary = (
            (cells[:, 0] == 0)
            | (cells[:, 0] == geo.height - 1)
            | (cells[:, 1] == 0)
            | (cells[:, 1] == geo.width - 1)
        )
        assert on_boundary.all()


def test_edge_equality_and_hash_are_by_tile_and_side(geo):
    a = geo.edge(3, Side.WEST)
    assert a == Edge(**{**a.__dict__})
    assert a != geo.edge(3, Side.EAST)
    assert len({a, geo.edge(3, Side.WEST), geo.edge(4, Side.WEST)}) == 2


# ---- seams -----------------------------------------------------------------------


def test_seams_pair_every_interior_edge_once_facing_across_one_cell(geo):
    assert len(geo.seams) == 112
    paired = set()
    for a, b in geo.seams:
        assert a.tile != b.tile
        assert not a.outer and not b.outer
        assert a.axis is b.axis
        assert {a.side, b.side} in ({Side.EAST, Side.WEST}, {Side.NORTH, Side.SOUTH})
        ca = geo.led_to_cell[a.tile, a.leds].astype(int)
        cb = geo.led_to_cell[b.tile, b.leds].astype(int)
        diff = cb - ca
        # a.leds[i] faces b.leds[i]: same along-axis coordinate, one cell apart
        assert (diff == diff[0]).all()
        assert sorted(np.abs(diff[0]).tolist()) == [0, 1]
        paired |= {a, b}
    assert paired == {e for e in geo.edges if not e.outer}


def test_seam_partner_lookup(geo, graph):
    a, b = geo.seams[0]
    assert graph.partner(a) is b and graph.partner(b) is a
    assert graph.partner(geo.edge(0, Side.SOUTH)) is None


# ---- rings and rails ---------------------------------------------------------------


def test_floor_ring_is_the_whole_boundary_once_in_a_closed_clockwise_walk(geo):
    ring = geo.floor_ring
    assert ring.shape == (480,)
    assert len(set(ring.tolist())) == 480
    boundary = {e for e in geo.edges if e.outer}
    assert set(ring.tolist()) == {int(v) for e in boundary for v in e.flat_leds}
    cells = [cell_of(geo, v) for v in ring]
    # consecutive entries are neighbours, except stepping across the two
    # dark corner cells between adjacent tiles (Chebyshev 3); closed loop
    gaps = [chebyshev(cells[i], cells[(i + 1) % 480]) for i in range(480)]
    assert max(gaps) == 3
    assert gaps.count(1) + gaps.count(2) + gaps.count(3) == 480
    # starts at the NW corner and goes clockwise as displayed: along the
    # top west->east first
    assert cells[0] == (geo.height - 1, 1)
    assert cells[1] == (geo.height - 1, 2)
    assert cells[120] == (geo.height - 2, geo.width - 1)  # turned down the east side


def test_tile_ring_is_the_tiles_leds_clockwise_from_nw(geo):
    for tile in range(geo.tiles):
        ring = geo.tile_ring(tile)
        assert ring.shape == (geo.leds_per_tile,)
        assert sorted(ring.tolist()) == list(range(geo.leds_per_tile))
        local = geo.led_to_cell[tile, ring] - np.array(geo.tile_origin(tile))
        assert tuple(local[0]) == (geo.cell_size - 1, 1)  # NW: top row, first lit cell
        assert tuple(local[1]) == (geo.cell_size - 1, 2)  # heading east
        for i in range(geo.leds_per_tile):
            assert chebyshev(local[i], local[(i + 1) % geo.leds_per_tile]) == 1


@pytest.mark.parametrize("axis", [Axis.X, Axis.Y, "x", "y"])
def test_rails_are_straight_monotone_120_led_lines_ordered_by_position(geo, axis):
    a = Axis(axis)
    fixed, varying = (0, 1) if a is Axis.X else (1, 0)
    previous_line = -1
    for index in range(16):
        rail = geo.rails(axis, index)
        assert rail.shape == (120,)
        cells = np.array([cell_of(geo, v) for v in rail])
        assert len(set(cells[:, fixed].tolist())) == 1  # all on one line
        assert (np.diff(cells[:, varying]) > 0).all()  # ascending along it
        line = int(cells[0, fixed])
        assert line > previous_line  # rails ascend across the floor
        previous_line = line
    with pytest.raises(ValueError):
        geo.rails(axis, 16)


def test_rail_0_is_the_south_wall_and_rail_1_the_first_seam_line(geo):
    assert cell_of(geo, geo.rails(Axis.X, 0)[0]) == (0, 1)
    assert cell_of(geo, geo.rails(Axis.X, 1)[0]) == (geo.cell_size - 1, 1)
    assert cell_of(geo, geo.rails(Axis.X, 2)[0]) == (geo.cell_size, 1)


# ---- the graph -------------------------------------------------------------------


def test_neighbors_share_exactly_one_junction_and_exclude_the_partner(geo, graph):
    for edge in geo.edges:
        partner = graph.partner(edge)
        for other in graph.neighbors(edge):
            assert other != edge and other != partner
            assert len(set(edge.junctions) & set(other.junctions)) == 1
        # symmetric
        for other in graph.neighbors(edge):
            assert edge in graph.neighbors(other)


def test_neighbor_counts_reflect_the_junction_structure(geo, graph):
    # A corner-of-floor edge meets 1 edge at the floor's corner and 3 at
    # the two-tile junction along the wall (4 edges there, minus itself).
    corner = geo.edge(0, Side.SOUTH)  # SW tile, bottom wall
    assert len(graph.neighbors(corner)) == 1 + 3
    # A fully interior edge meets 7 at each end: 8 edges per 4-tile
    # junction, minus itself, and the partner is excluded at both.
    inner = geo.edge(9, Side.NORTH)  # tile (1, 1)
    assert len(graph.neighbors(inner)) == 2 * (8 - 2)


def test_walk_returns_n_connected_distinct_edges_with_no_backtracking(graph):
    rng = np.random.default_rng(61)
    for _ in range(50):
        n = int(rng.integers(1, 25))
        path = graph.walk(graph.random_edge(rng), n, rng, turn_bias=float(rng.random()))
        assert isinstance(path, Path) and len(path) == n
        assert len(set(path.edges)) == n
        leds = path.leds()
        assert len(leds) == n * 15 and len(set(leds.tolist())) == len(leds)
        junctions = path.junctions()
        assert len(junctions) == n + 1
        for step, start, end in zip(path.steps, junctions, junctions[1:]):
            assert step.start_junction == start and step.end_junction == end
            assert set(step.edge.junctions) == {start, end}
        for a, b in zip(path.steps, path.steps[1:]):
            assert b.edge != graph.partner(a.edge)


def test_walk_leds_are_in_travel_order(geo, graph):
    rng = np.random.default_rng(3)
    path = graph.walk(graph.random_edge(rng), 10, rng)
    cells = [cell_of(geo, v) for v in path.leds()]
    gaps = [chebyshev(a, b) for a, b in zip(cells, cells[1:])]
    assert max(gaps) <= 3  # within an edge 1; across a junction at most 3
    xy = path.positions()
    assert xy.shape == (150, 2)
    assert tuple(xy[0]) == (cells[0][1] + 0.5, cells[0][0] + 0.5)


def test_turn_bias_zero_runs_straight_along_a_wall(geo, graph):
    rng = np.random.default_rng(0)
    path = graph.walk(geo.edge(0, Side.SOUTH), 8, rng, turn_bias=0.0, forward=True)
    assert path.edges == tuple(geo.edge(t, Side.SOUTH) for t in range(8))
    assert all(step.forward for step in path.steps)


def test_turn_bias_one_turns_at_every_junction(geo, graph):
    rng = np.random.default_rng(0)
    for _ in range(20):
        path = graph.walk(geo.edge(27, Side.NORTH), 6, rng, turn_bias=1.0)
        axes = [step.edge.axis for step in path.steps]
        assert all(a is not b for a, b in zip(axes, axes[1:]))


def test_walk_backtracks_out_of_dead_ends(geo, graph):
    # 256 edges: a self-avoiding walk that long does not exist from
    # anywhere, but a much longer one than a greedy walk would find does.
    rng = np.random.default_rng(5)
    path = graph.walk(geo.edge(0, Side.SOUTH), 60, rng, turn_bias=0.5)
    assert len(path) == 60
    with pytest.raises(ValueError):
        graph.walk(geo.edge(0, Side.SOUTH), 257, rng)
    with pytest.raises(ValueError):
        graph.walk(geo.edge(0, Side.SOUTH), 0, rng)


def test_shortest_path_has_the_hand_computed_length(geo, graph):
    south, north = geo.edge(0, Side.SOUTH), geo.edge(0, Side.NORTH)
    path = graph.shortest_path(south, north)
    assert len(path) == 3  # SOUTH -> WEST or EAST -> NORTH
    assert path.edges[0] is south and path.edges[-1] is north
    assert len(graph.shortest_path(south, geo.edge(7, Side.SOUTH))) == 8  # along the wall
    assert len(graph.shortest_path(south, south)) == 1
    # seam partners are not neighbours; getting onto the facing run means
    # going round a corner and back
    a, b = geo.seams[0]
    assert len(graph.shortest_path(a, b)) == 3


def test_shortest_path_is_oriented_in_travel_order(geo, graph):
    path = graph.shortest_path(geo.edge(0, Side.SOUTH), geo.edge(7, Side.SOUTH))
    assert all(step.forward for step in path.steps)  # west -> east along the wall
    path = graph.shortest_path(geo.edge(7, Side.SOUTH), geo.edge(0, Side.SOUTH))
    assert not any(step.forward for step in path.steps)
    junctions = path.junctions()
    for step, start, end in zip(path.steps, junctions, junctions[1:]):
        assert (step.start_junction, step.end_junction) == (start, end)


def test_branch_forks_off_the_path_without_reusing_it(graph):
    rng = np.random.default_rng(9)
    trunk = graph.walk(graph.random_edge(rng), 12, rng)
    forks = graph.branch(trunk, rng, n=3)
    assert len(forks) == 3
    used = set(trunk.edges) | {graph.partner(e) for e in trunk.edges} - {None}
    for fork in forks:
        assert len(fork) == 6
        assert not (set(fork.edges) & used)
        assert fork.junctions()[0] in trunk.junctions()[:-1]
        used |= set(fork.edges)


def test_random_edge_covers_the_graph(graph):
    rng = np.random.default_rng(2)
    picked = {graph.random_edge(rng) for _ in range(3000)}
    assert len(picked) == 256


# ---- using it from a frame ------------------------------------------------------


def test_flat_indices_address_the_frame(geo):
    frame = PixelFrame.black()
    frame.flat[geo.floor_ring] = (1, 2, 3)
    assert (frame.grid[geo.lit_mask].sum(axis=-1) > 0).sum() == 480
    assert (frame.grid[0][geo.lit_mask[0]] == (1, 2, 3)).all()  # south wall lit
    assert not frame.grid[1, 1:-1].any()
    frame.flat[geo.edge(9, Side.WEST).flat_leds] = 255
    y0, x0 = geo.tile_origin(9)
    assert (frame.grid[y0 + 1 : y0 + 16, x0] == 255).all()
