import numpy as np
import pytest

from df2_pi import phase
from df2_pi.geometry import FloorGeometry
from df2_pi.tempo import lfo


@pytest.fixture(scope="module")
def geo() -> FloorGeometry:
    return FloorGeometry()


def every_map(geo):
    rng = np.random.default_rng(5)
    for name in phase.MAPS:
        for resolution in phase.RESOLUTIONS:
            if name == "perimeter" and resolution == "tile":
                continue
            for mirror in (False, True):
                yield name, resolution, mirror, phase.by_name(geo, name, rng=rng, resolution=resolution, mirror=mirror)


def test_every_map_has_its_resolutions_shape_and_stays_in_0_to_1(geo):
    for name, resolution, mirror, offsets in every_map(geo):
        expected = (geo.tile_rows, geo.tile_cols) if resolution == "tile" else (geo.tiles, geo.leds_per_tile)
        assert offsets.shape == expected, (name, resolution)
        assert offsets.min() >= 0.0 and offsets.max() <= 1.0, (name, resolution, mirror)


def test_rows_and_cols_start_at_row_0_nearest_the_pi_and_the_west_edge(geo):
    rows = phase.rows(geo)
    assert rows[:, 0].tolist() == pytest.approx([r / 8 for r in range(8)])  # row 0 is 0, rising away from the Pi
    assert phase.cols(geo)[0].tolist() == pytest.approx([c / 8 for c in range(8)])
    led_rows = phase.rows(geo, resolution="led")
    assert led_rows[:8].max() < led_rows[56:].min()  # tile row 0's LEDs all come before tile row 7's
    assert phase.diagonal(geo)[0, 0] == 0.0 and phase.diagonal(geo)[7, 7] == phase.diagonal(geo).max()


def test_angle_is_zero_due_north_and_runs_clockwise(geo):
    point = (3.5 * geo.cell_size, 3.5 * geo.cell_size)  # the centre of tile (3, 3)
    angle = phase.angle(geo, point)
    assert angle[5, 3] == pytest.approx(0.0)  # north: up as displayed, away from the Pi
    assert angle[3, 5] == pytest.approx(0.25)  # east
    assert angle[1, 3] == pytest.approx(0.5)  # south
    assert angle[3, 1] == pytest.approx(0.75)  # west


def test_radial_is_smallest_at_its_point_and_largest_at_the_far_corner(geo):
    radial = phase.radial(geo)
    centre = radial[3:5, 3:5]
    assert centre.max() == radial.min() and radial[0, 0] == radial.max()
    corner = phase.radial(geo, point=(0.0, 0.0))
    assert corner[0, 0] == corner.min() and corner[7, 7] == corner.max()


def test_checker_alternates_by_half_a_cycle(geo):
    checker = phase.checker(geo)
    assert set(np.unique(checker)) == {0.0, 0.5}
    assert (checker[:, 1:] != checker[:, :-1]).all() and (checker[1:] != checker[:-1]).all()
    led = phase.checker(geo, resolution="led")
    assert (led == checker.reshape(-1)[:, None]).all()  # every LED takes its tile's


def test_random_is_stable_for_a_generator_and_reproducible_from_its_seed(geo):
    rng = np.random.default_rng(11)
    first = phase.random(geo, rng)
    assert (phase.random(geo, rng) == first).all()  # the same run asking again
    again = phase.random(geo, np.random.default_rng(11))
    assert (again == first).all()  # a run from the same seed
    assert (phase.random(geo, np.random.default_rng(12)) != first).any()


def test_perimeter_follows_the_led_chain_round_each_tile(geo):
    perimeter = phase.perimeter(geo)
    n = geo.leds_per_tile
    assert (perimeter == (np.arange(n) / n)[None, :]).all()  # the chain index, what ledwalk lights in order
    cells = geo.led_to_cell[0].astype(float)  # (60, 2) y, x of tile 0's LEDs in chain order
    steps = np.linalg.norm(np.diff(np.vstack([cells, cells[:1]]), axis=0), axis=1)
    assert steps.max() <= np.sqrt(2) + 1e-9  # each next LED is a neighbour: a walk round the ring, closing
    assert cells[0, 1] == cells[:, 1].min() and cells[1, 0] > cells[0, 0]  # starts on the west side, going north: clockwise
    with pytest.raises(ValueError):
        phase.by_name(geo, "perimeter", resolution="tile")


def test_mirror_is_symmetric_about_the_centre(geo):
    rows = phase.rows(geo, mirror=True)
    assert (rows == rows[::-1]).all() and rows[0, 0] == 0.0 and rows[3, 0] == rows.max()
    cols = phase.cols(geo, mirror=True)
    assert (cols == cols[:, ::-1]).all()
    diagonal = phase.diagonal(geo, mirror=True)
    assert (diagonal == diagonal[::-1]).all() and (diagonal == diagonal[:, ::-1]).all()
    angle = phase.angle(geo, mirror=True)
    assert np.allclose(angle, angle[:, ::-1])  # east matches west
    radial = phase.radial(geo, point=(10.0, 20.0), mirror=True)
    assert np.allclose(radial, radial[::-1, ::-1])
    random = phase.random(geo, np.random.default_rng(3), mirror=True)
    assert (random == random[::-1, ::-1]).all()
    led_random = phase.random(geo, np.random.default_rng(3), resolution="led", mirror=True).reshape(-1)
    cells = geo.led_to_cell.reshape(-1, 2)
    grid = np.full((geo.height, geo.width), -1)
    grid[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
    partner = grid[geo.height - 1 - cells[:, 0], geo.width - 1 - cells[:, 1]]
    assert (led_random == led_random[partner]).all()


def test_reverse_and_spread(geo):
    rows = phase.rows(geo)
    assert np.allclose(phase.rows(geo, reverse=True), rows.max() - rows)
    assert np.allclose(phase.rows(geo, spread=0.5), rows * 0.5)
    assert (phase.radial(geo, spread=0.0) == 0).all()  # everything in phase


def test_maps_are_fresh_arrays(geo):
    first = phase.rows(geo)
    first[:] = 9.0
    assert phase.rows(geo).max() < 1.0


def test_a_full_spread_wave_puts_every_row_at_a_different_point_of_the_cycle(geo):
    level = lfo(0.0 - phase.rows(geo), shape="saw")
    assert len(np.unique(np.round(level[:, 0], 6))) == geo.tile_rows


def test_bad_names_and_options(geo):
    with pytest.raises(ValueError):
        phase.by_name(geo, "spiral")
    with pytest.raises(ValueError):
        phase.by_name(geo, "random")  # needs rng
    with pytest.raises(ValueError):
        phase.rows(geo, resolution="pixel")
