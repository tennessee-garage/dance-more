import numpy as np
import pytest

from df2_pi.edges import Axis
from df2_pi.geometry import FloorGeometry
from df2_pi.pixels import PixelFrame, default_geometry


@pytest.fixture(scope="module")
def geo() -> FloorGeometry:
    return default_geometry()


def lit(frame: PixelFrame) -> np.ndarray:
    """Flat indices of every LED with any channel on."""
    return np.flatnonzero(frame.flat.any(axis=-1))


def xy(geo, flat: np.ndarray) -> np.ndarray:
    """(n, 2) float (x, y) centres of flat LED indices."""
    return geo.led_positions.reshape(-1, 2)[flat][:, ::-1]


# ---- never a dark cell --------------------------------------------------------------


@pytest.mark.parametrize("blend", ["set", "add", "max", "alpha"])
@pytest.mark.parametrize("falloff", ["flat", "linear", "gaussian"])
def test_primitives_only_ever_light_leds(geo, blend, falloff):
    frame = PixelFrame.black()
    frame.splat(68.0, 68.0, (255, 255, 255), radius=30, falloff=falloff, blend=blend)
    frame.line((0.0, 0.0), (136.0, 136.0), (255, 0, 0), width=6, falloff=falloff, blend=blend)
    frame.circle(40.0, 90.0, 25.0, (0, 255, 0), width=4, falloff=falloff, blend=blend)
    assert len(lit(frame)) > 0
    grid = frame.grid
    assert not grid[~geo.lit_mask].any()
    # and the frame is still a valid PixelFrame that round-trips
    assert PixelFrame.from_grid(grid, strict=True) == frame


# ---- splat --------------------------------------------------------------------


def test_splat_lights_exactly_the_leds_within_radius(geo):
    frame = PixelFrame.black()
    frame.splat(17.0, 17.0, (255, 255, 255), radius=3.0, falloff="flat")
    on = lit(frame)
    d = np.hypot(*(xy(geo, np.arange(geo.led_count)) - (17.0, 17.0)).T)
    np.testing.assert_array_equal(on, np.flatnonzero(d <= 3.0))
    assert (frame.flat[on] == 255).all()
    # the four tiles meeting at junction (1, 1) each contribute
    assert set(on // geo.leds_per_tile) == {0, 1, 8, 9}


def test_splat_gaussian_falls_off_with_distance_and_stops_at_radius(geo):
    frame = PixelFrame.black()
    # centred on a LED on the south wall, so distances along the wall are integers
    cx, cy = 5.5, 0.5
    frame.splat(cx, cy, (255, 255, 255), radius=4.0, falloff="gaussian")
    row = frame.grid[0, :, 0].astype(int)
    assert row[5] == 255
    assert row[5] > row[6] > row[7] > row[8] > row[9] > 0  # d = 1..4
    assert row[10] == 0  # d = 5 > radius
    assert row[4] == row[6]  # symmetric
    assert row[9] == round(255 * np.exp(-2.0))  # e^-2 at the radius


def test_splat_linear_falloff(geo):
    frame = PixelFrame.black()
    frame.splat(5.5, 0.5, (200, 200, 200), radius=4.0, falloff="linear")
    row = frame.grid[0, :, 0].astype(int)
    assert row[5:10].tolist() == [200, 150, 100, 50, 0]


# ---- line and circle -----------------------------------------------------------------


def test_line_along_a_rail_lights_the_whole_rail_and_nothing_else(geo):
    frame = PixelFrame.black()
    frame.line((0.0, 0.5), (136.0, 0.5), (0, 0, 255), width=1.0)
    np.testing.assert_array_equal(np.sort(lit(frame)), np.sort(geo.rails(Axis.X, 0)))
    assert (frame.flat[geo.rails(Axis.X, 0)] == (0, 0, 255)).all()


def test_line_is_a_segment_not_an_infinite_line(geo):
    frame = PixelFrame.black()
    frame.line((1.5, 0.5), (8.5, 0.5), (255, 255, 255), width=1.0)
    assert frame.grid[0, 1:9].all()
    assert not frame.grid[0, 9:].any()


def test_diagonal_line_crosses_only_the_edges_it_passes(geo):
    # The exact y = x diagonal threads through corner and interior cells
    # only - no LED sits on it - so offset it to y = x + 3, which clips the
    # west and north runs of the tiles along the diagonal.
    frame = PixelFrame.black()
    frame.line((0.0, 3.0), (133.0, 136.0), (255, 255, 255), width=1.5)
    on = lit(frame)
    assert len(on) > 0
    all_xy = xy(geo, np.arange(geo.led_count))
    d = np.abs(all_xy[:, 1] - all_xy[:, 0] - 3.0) / np.sqrt(2.0)
    np.testing.assert_array_equal(on, np.flatnonzero(d <= 0.75))
    assert len(on) < geo.led_count * 0.05  # a line, not a wash


def test_circle_lights_leds_at_the_radius_only(geo):
    frame = PixelFrame.black()
    frame.circle(68.0, 68.0, 20.0, (255, 255, 255), width=2.0)
    on = xy(geo, lit(frame))
    d = np.hypot(on[:, 0] - 68.0, on[:, 1] - 68.0)
    assert len(on) > 0
    assert (np.abs(d - 20.0) <= 1.0).all()
    d_all = np.hypot(*(xy(geo, np.arange(geo.led_count)) - (68.0, 68.0)).T)
    assert len(on) == (np.abs(d_all - 20.0) <= 1.0).sum()


# ---- blend modes ----------------------------------------------------------------


def test_blend_modes(geo):
    base = PixelFrame.black()
    base.data[:] = 100
    at = geo.rails(Axis.X, 0)[3]  # a LED the splat will cover
    x, y = xy(geo, np.array([at]))[0]

    f = base.copy()
    f.splat(x, y, (200, 200, 200), radius=1.0, falloff="flat", blend="set")
    assert f.flat[at].tolist() == [200, 200, 200]

    f = base.copy()
    f.splat(x, y, (200, 200, 200), radius=1.0, falloff="flat", blend="add")
    assert f.flat[at].tolist() == [255, 255, 255]  # saturates

    f = base.copy()
    f.splat(x, y, (50, 200, 100), radius=1.0, falloff="flat", blend="max")
    assert f.flat[at].tolist() == [100, 200, 100]

    f = base.copy()
    f.splat(x, y, (200, 200, 200), radius=2.0, falloff="linear", blend="alpha")
    assert f.flat[at].tolist() == [200, 200, 200]  # weight 1 at the centre
    neighbour = geo.rails(Axis.X, 0)[4]  # 1 cell away: weight 0.5
    assert f.flat[neighbour].tolist() == [150, 150, 150]


def test_untouched_leds_are_left_alone_in_every_mode(geo):
    base = PixelFrame.black()
    base.data[:] = 100
    for blend in ("set", "add", "max", "alpha"):
        f = base.copy()
        f.splat(5.5, 0.5, (255, 255, 255), radius=1.0, falloff="flat", blend=blend)
        touched = lit(f)[f.flat[lit(f)].any(axis=-1) & (f.flat[lit(f)] != 100).any(axis=-1)]
        assert len(touched) == 3  # the LED and its two neighbours along the wall
        rest = np.setdiff1d(np.arange(geo.led_count), touched)
        assert (f.flat[rest] == 100).all()


def test_painting_mutates_the_frame_and_respects_freeze():
    frame = PixelFrame.black()
    frame.splat(5.5, 0.5, (1, 1, 1), radius=1.0)
    assert frame.data.any()
    frame.freeze()
    with pytest.raises(ValueError):
        frame.splat(5.5, 0.5, (1, 1, 1), radius=1.0)


def test_bad_arguments_are_rejected():
    frame = PixelFrame.black()
    with pytest.raises(ValueError):
        frame.splat(0, 0, (1, 1, 1), radius=0)
    with pytest.raises(ValueError):
        frame.splat(0, 0, (1, 1, 1), falloff="cubic")
    with pytest.raises(ValueError):
        frame.splat(0, 0, (1, 1, 1), blend="screen")
    with pytest.raises(ValueError):
        frame.splat(0, 0, (1, 1), radius=1)
    with pytest.raises(ValueError):
        frame.circle(0, 0, -1, (1, 1, 1))
