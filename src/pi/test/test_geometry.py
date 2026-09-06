import numpy as np
import pytest

from df2_pi.geometry import FloorGeometry, Side


@pytest.fixture(scope="module")
def geo() -> FloorGeometry:
    return FloorGeometry.default()


def test_the_build_spec_numbers(geo):
    # The one test that restates the spec rather than deriving from it, so
    # changing the floor's LED geometry has to be a deliberate edit here.
    # 15 per side is what the ~10" ledge on a 15" tile frame fits at 300 LED/5m
    # - see docs/hardware-tile.md's "LED layout and chain order".
    assert geo.leds_per_side == 15
    assert geo.leds_per_tile == 60
    assert geo.cell_size == 17
    assert geo.tiles == 64
    assert geo.led_count == 3840
    assert (geo.height, geo.width) == (136, 136)


def test_lit_mask_lights_exactly_led_count_cells(geo):
    assert geo.lit_mask.sum() == geo.led_count


def test_every_tile_block_contributes_exactly_leds_per_tile_lit_cells(geo):
    for tile in range(geo.tiles):
        y0, x0 = geo.tile_origin(tile)
        block = geo.lit_mask[y0 : y0 + geo.cell_size, x0 : x0 + geo.cell_size]
        assert block.sum() == geo.leds_per_tile


def test_every_tile_side_is_a_full_run_and_corners_are_dark(geo):
    n = geo.leds_per_side
    for tile in range(geo.tiles):
        y0, x0 = geo.tile_origin(tile)
        block = geo.lit_mask[y0 : y0 + geo.cell_size, x0 : x0 + geo.cell_size]
        assert block[0, :].sum() == n  # south edge
        assert block[-1, :].sum() == n  # north edge
        assert block[:, 0].sum() == n  # west edge
        assert block[:, -1].sum() == n  # east edge
        for ly, lx in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            assert not block[ly, lx]


def test_tile_interior_is_dark(geo):
    end = geo.cell_size - 1
    for tile in range(geo.tiles):
        y0, x0 = geo.tile_origin(tile)
        interior = geo.lit_mask[y0 + 1 : y0 + end, x0 + 1 : x0 + end]
        assert interior.shape == (geo.leds_per_side, geo.leds_per_side)
        assert not interior.any()


def test_led_to_cell_and_cell_to_led_are_a_bijection(geo):
    seen_cells = set()
    for tile in range(geo.tiles):
        for led in range(geo.leds_per_tile):
            y, x = (int(v) for v in geo.led_to_cell[tile, led])
            assert (y, x) not in seen_cells, f"tile {tile} led {led} collides at ({y}, {x})"
            seen_cells.add((y, x))
            back_tile, back_led = (int(v) for v in geo.cell_to_led[y, x])
            assert (back_tile, back_led) == (tile, led)
    assert len(seen_cells) == geo.led_count


def test_tile_blocks_are_disjoint_cell_aligned_and_tile_the_full_grid(geo):
    origins = [geo.tile_origin(t) for t in range(geo.tiles)]
    assert all(y % geo.cell_size == 0 and x % geo.cell_size == 0 for y, x in origins)
    assert len(set(origins)) == geo.tiles
    assert max(y for y, _ in origins) + geo.cell_size == geo.height
    assert max(x for _, x in origins) + geo.cell_size == geo.width


def test_address_of_is_divmod_by_tile_cols(geo):
    for tile in range(geo.tiles):
        assert geo.address_of(tile) == divmod(tile, 8)


def test_to_display_reverses_y_only_and_is_its_own_inverse(geo):
    grid = np.arange(geo.height * geo.width * 3, dtype=np.uint8).reshape(geo.height, geo.width, 3)
    flipped = geo.to_display(grid)
    assert flipped.shape == grid.shape
    np.testing.assert_array_equal(flipped, grid[::-1])
    np.testing.assert_array_equal(geo.to_display(flipped), grid)
    np.testing.assert_array_equal(flipped[-1], grid[0])


def test_led_0_of_every_tile_is_bottom_of_the_left_side(geo):
    for tile in range(geo.tiles):
        y0, x0 = geo.tile_origin(tile)
        y, x = (int(v) for v in geo.led_to_cell[tile, 0])
        assert (y - y0, x - x0) == (1, 0)


def test_the_first_side_run_climbs_the_left_side(geo):
    n = geo.leds_per_side
    local = geo.led_to_cell[0, :n] - np.array(geo.tile_origin(0))
    assert (local[:, 1] == 0).all()
    assert list(local[:, 0]) == list(range(1, n + 1))


@pytest.mark.parametrize(
    "run,fixed_axis,fixed_at_far_edge,varying,ascending",
    [
        # run i covers LEDs [i*n, (i+1)*n). "far edge" means the fixed local
        # coordinate is cell_size-1 rather than 0.
        (0, 1, False, 0, True),  # west:  lx=0,           ly ascending
        (1, 0, True, 1, True),  # north: ly=cell_size-1, lx ascending
        (2, 1, True, 0, False),  # east:  lx=cell_size-1, ly descending
        (3, 0, False, 1, False),  # south: ly=0,           lx descending
    ],
)
def test_each_side_run_is_documented_exactly(
    geo, run, fixed_axis, fixed_at_far_edge, varying, ascending
):
    n = geo.leds_per_side
    fixed_value = geo.cell_size - 1 if fixed_at_far_edge else 0
    expected = list(range(1, n + 1)) if ascending else list(range(n, 0, -1))

    local = geo.led_to_cell[0, run * n : (run + 1) * n] - np.array(geo.tile_origin(0))
    assert (local[:, fixed_axis] == fixed_value).all()
    assert list(local[:, varying]) == expected


def test_consecutive_leds_are_orthogonal_neighbours_except_at_the_four_dark_corners(geo):
    n = geo.leds_per_side
    diagonal_steps = {n - 1, 2 * n - 1, 3 * n - 1, 4 * n - 1}
    for tile in range(geo.tiles):
        for led in range(geo.leds_per_tile):
            a = geo.led_to_cell[tile, led]
            b = geo.led_to_cell[tile, (led + 1) % geo.leds_per_tile]
            dy, dx = int(b[0]) - int(a[0]), int(b[1]) - int(a[1])
            manhattan = abs(dy) + abs(dx)
            if led in diagonal_steps:
                assert abs(dy) == 1 and abs(dx) == 1
            else:
                assert manhattan == 1


def test_a_tiles_chain_is_exactly_that_tiles_lit_cells_no_repeats(geo):
    for tile in range(geo.tiles):
        y0, x0 = geo.tile_origin(tile)
        block_lit = {
            (int(y), int(x))
            for y in range(y0, y0 + geo.cell_size)
            for x in range(x0, x0 + geo.cell_size)
            if geo.lit_mask[y, x]
        }
        chain_cells = {(int(y), int(x)) for y, x in geo.led_to_cell[tile]}
        assert chain_cells == block_lit
        assert len(chain_cells) == geo.leds_per_tile


def test_led_positions_are_cell_centres(geo):
    y, x = (int(v) for v in geo.led_to_cell[0, 0])
    py, px = (float(v) for v in geo.led_positions[0, 0])
    assert (py, px) == (y + 0.5, x + 0.5)


def test_worked_examples(geo):
    # Spelled-out coordinates, as a check on the derived tests above. The
    # originals came from issue #59 at 10 LEDs/side; these are the same
    # walk re-derived for the 15/side build spec.
    np.testing.assert_array_equal(geo.led_to_cell[0, 0], [1, 0])  # bottom of west
    np.testing.assert_array_equal(geo.led_to_cell[0, 19], [16, 5])  # 5th of north
    np.testing.assert_array_equal(geo.led_to_cell[9, 0], [18, 17])  # tile (1,1)
    np.testing.assert_array_equal(geo.cell_to_led[1, 0], [0, 0])
    np.testing.assert_array_equal(geo.cell_to_led[0, 0], [-1, -1])


def test_cell_to_led_dtype_minus_one_pair_marks_dark_cells(geo):
    assert not geo.lit_mask[0, 0]
    y, x = np.nonzero(~geo.lit_mask)[0][0], np.nonzero(~geo.lit_mask)[1][0]
    tile, led = geo.cell_to_led[y, x]
    assert (int(tile), int(led)) == (-1, -1)


def test_tile_at_returns_none_for_dark_cells_and_the_tile_for_lit_ones(geo):
    assert geo.tile_at(0, 0) is None  # dark corner
    assert geo.tile_at(5, 5) is None  # dark interior
    assert geo.tile_at(1, 0) == 0  # tile 0's LED 0


def test_tile_at_rejects_out_of_range_cells(geo):
    with pytest.raises(ValueError):
        geo.tile_at(-1, 0)
    with pytest.raises(ValueError):
        geo.tile_at(0, geo.width)


def test_side_leds_shapes_and_documented_orientation(geo):
    for side in Side:
        leds = geo.side_leds(0, side)
        assert len(leds) == geo.leds_per_side
        assert len(set(int(v) for v in leds)) == geo.leds_per_side

    # WEST/EAST ascend south->north; NORTH/SOUTH ascend west->east.
    for side, axis in [(Side.WEST, 0), (Side.EAST, 0), (Side.NORTH, 1), (Side.SOUTH, 1)]:
        leds = geo.side_leds(0, side)
        coords = geo.led_to_cell[0, leds][:, axis]
        assert list(coords) == sorted(coords.tolist())


def test_side_leds_west_is_the_natural_chain_order(geo):
    # WEST is the first run in ascending y, already the chain's own order.
    np.testing.assert_array_equal(
        geo.side_leds(0, Side.WEST), np.arange(geo.leds_per_side)
    )


def test_corners_populated_true_is_rejected():
    with pytest.raises(NotImplementedError):
        FloorGeometry(corners_populated=True)


def test_tables_are_cached_not_rebuilt(geo):
    assert geo.led_to_cell is geo.led_to_cell
    assert geo.lit_mask is geo.lit_mask
