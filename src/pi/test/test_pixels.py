import numpy as np
import pytest

from df2_pi.geometry import FloorGeometry
from df2_pi.pixels import (
    FrameOwnershipError,
    PixelFrame,
    TileFrame,
    blend,
    check_ownership,
    default_geometry,
)


@pytest.fixture(scope="module")
def geo() -> FloorGeometry:
    return default_geometry()


@pytest.fixture
def noise() -> PixelFrame:
    """A PixelFrame with every LED a different, deterministic colour."""
    rng = np.random.default_rng(60)
    return PixelFrame(rng.integers(0, 256, PixelFrame.shape_for(default_geometry()), dtype=np.uint8))


# ---- shapes -------------------------------------------------------------------


def test_shapes_and_dtypes_follow_the_geometry(geo):
    t = TileFrame.black()
    p = PixelFrame.black()
    assert t.data.shape == (geo.tile_rows, geo.tile_cols, 3) == (8, 8, 3)
    assert p.data.shape == (geo.tiles, geo.leds_per_tile, 3) == (64, 60, 3)
    assert t.data.dtype == p.data.dtype == np.uint8
    assert not t.data.any()
    assert not p.data.any()
    assert t.geometry is geo and p.geometry is geo


def test_wrong_shape_or_dtype_is_rejected():
    with pytest.raises(ValueError):
        TileFrame(np.zeros((8, 8, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        PixelFrame(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(TypeError):
        TileFrame(np.zeros((8, 8, 3), dtype=np.float32))


def test_indexing_writes_through_to_data():
    t = TileFrame.black()
    t[2, 5] = (255, 0, 128)
    assert t.data[2, 5].tolist() == [255, 0, 128]
    assert t[2, 5].tolist() == [255, 0, 128]
    p = PixelFrame.black()
    p.tile(9)[:] = (1, 2, 3)
    assert (p.data[9] == (1, 2, 3)).all()
    assert not p.data[8].any() and not p.data[10].any()


def test_frames_follow_a_non_default_geometry():
    small = FloorGeometry(tile_grid=(2, 3), leds_per_side=4)
    t = TileFrame.black(small)
    p = t.to_pixels()
    assert t.data.shape == (2, 3, 3)
    assert p.data.shape == (6, 16, 3)
    assert p.grid.shape == (small.height, small.width, 3)
    assert PixelFrame.from_grid(p.grid, geometry=small) == p


# ---- tile <-> pixel conversion ---------------------------------------------------


def test_to_pixels_fills_every_led_of_each_tile(geo):
    t = TileFrame.black()
    t[1, 2] = (10, 20, 30)
    p = t.to_pixels()
    tile = 1 * geo.tile_cols + 2
    assert (p.tile(tile) == (10, 20, 30)).all()
    assert p.data.sum() == (10 + 20 + 30) * geo.leds_per_tile


def test_tile_frame_survives_a_round_trip_through_pixels_exactly():
    # A uniform tile's linear mean is its own linear value, so this reduces
    # to encode(decode(v)) == v for every byte - covered exhaustively by
    # sweeping all 256 values through the tiles.
    t = TileFrame.black()
    values = np.arange(256, dtype=np.uint8)
    t.data[..., 0] = values[:64].reshape(8, 8)
    t.data[..., 1] = values[64:128].reshape(8, 8)
    t.data[..., 2] = values[128:192].reshape(8, 8)
    assert t.to_pixels().to_tiles() == t
    t.data[..., 0] = values[192:256].reshape(8, 8)
    assert t.to_pixels().to_tiles() == t


def test_to_tiles_averages_in_linear_light(geo):
    p = PixelFrame.black()
    half = geo.leds_per_tile // 2
    p.tile(0)[:half] = 255
    assert p.to_tiles()[0, 0].tolist() == [186, 186, 186]  # not 128


def test_to_tiles_of_black_is_exactly_zero():
    assert not PixelFrame.black().to_tiles().data.any()


def test_to_tiles_averages_channels_independently(geo):
    p = PixelFrame.black()
    half = geo.leds_per_tile // 2
    p.tile(0)[:half] = (255, 0, 0)
    p.tile(0)[half:] = (0, 255, 0)
    assert p.to_tiles()[0, 0].tolist() == [186, 186, 0]


# ---- the grid view ---------------------------------------------------------------


def test_grid_zeroes_dark_cells_and_preserves_lit_ones(geo, noise):
    img = noise.grid
    assert img.shape == (geo.height, geo.width, 3)
    assert img.dtype == np.uint8
    assert not img[~geo.lit_mask].any()
    ys, xs = geo.led_to_cell[..., 0], geo.led_to_cell[..., 1]
    assert (img[ys, xs] == noise.data).all()
    # spot-check chain order against the documented LED-0 placement
    assert (img[1, 0] == noise.data[0, 0]).all()


def test_grid_is_a_fresh_array_each_call(noise):
    a = noise.grid
    b = noise.grid
    assert a is not b
    a[:] = 0
    assert noise.grid.any()


def test_from_grid_round_trips_any_pixel_frame(noise):
    assert PixelFrame.from_grid(noise.grid) == noise


def test_from_grid_drops_dark_cells_unless_strict(geo):
    img = np.full((geo.height, geo.width, 3), 7, dtype=np.uint8)
    p = PixelFrame.from_grid(img)
    assert (p.data == 7).all()
    assert not p.grid[~geo.lit_mask].any()
    with pytest.raises(ValueError, match="dark cell"):
        PixelFrame.from_grid(img, strict=True)
    # strict passes when only lit cells are written
    img[~geo.lit_mask] = 0
    assert PixelFrame.from_grid(img, strict=True) == p


def test_from_grid_does_not_alias_the_image(geo):
    img = np.zeros((geo.height, geo.width, 3), dtype=np.uint8)
    p = PixelFrame.from_grid(img)
    img[geo.lit_mask] = 255
    assert not p.data.any()


def test_from_grid_rejects_wrong_shape_and_dtype(geo):
    with pytest.raises(ValueError):
        PixelFrame.from_grid(np.zeros((geo.height, geo.width), dtype=np.uint8))
    with pytest.raises(TypeError):
        PixelFrame.from_grid(np.zeros((geo.height, geo.width, 3), dtype=np.float32))


def test_grid_view_and_commit_avoid_the_copy(geo, noise):
    p = noise.copy()
    view = p.grid_view()
    assert view is p.grid_view()  # same buffer on every call
    assert PixelFrame.from_grid(view) == p  # seeded from the frame
    view[:] = 0
    view[1, 0] = (9, 8, 7)  # tile 0, LED 0
    view[0, 0] = (1, 1, 1)  # a dark corner - ignored on commit
    assert p == noise  # nothing lands until commit()
    p.commit()
    assert p.data[0, 0].tolist() == [9, 8, 7]
    assert p.data.sum() == 9 + 8 + 7
    assert not p.grid[0, 0].any()


def test_commit_without_a_view_raises():
    with pytest.raises(ValueError):
        PixelFrame.black().commit()


# ---- helpers ------------------------------------------------------------------


def test_gain_saturates_rather_than_wrapping():
    p = PixelFrame.black()
    p.data[:] = 200
    assert (p.gain(2.0).data == 255).all()
    assert (p.gain(0.5).data == 100).all()


def test_gain_of_one_is_the_identity(noise):
    assert noise.gain(1.0) == noise


def test_gain_rejects_negative(noise):
    with pytest.raises(ValueError):
        noise.gain(-0.1)


def test_blend_endpoints_are_exact(noise):
    other = noise.gain(0.3)
    assert blend(noise, other, 0.0) == noise
    assert blend(noise, other, 1.0) == other


def test_blend_mixes_in_linear_light():
    black = PixelFrame.black()
    white = PixelFrame.black()
    white.data[:] = 255
    assert (blend(black, white, 0.5).data == 186).all()  # not 128


def test_blend_works_on_tile_frames_and_rejects_mixed_types():
    a = TileFrame.black()
    b = TileFrame.black()
    b.data[:] = 255
    assert (blend(a, b, 0.5).data == 186).all()
    with pytest.raises(TypeError):
        blend(a, PixelFrame.black(), 0.5)
    with pytest.raises(ValueError):
        blend(a, b, 1.5)


def test_helpers_return_new_frames_and_leave_inputs_untouched(noise):
    before = noise.data.copy()
    other = noise.gain(0.5)
    c = noise.copy()
    g = noise.gain(2.0)
    m = blend(noise, other, 0.5)
    for out in (c, g, m):
        assert out is not noise
        assert not np.may_share_memory(out.data, noise.data)
    assert (noise.data == before).all()
    assert (other.data == np.rint(before * 0.5)).all()


def test_like_produces_a_black_frame_of_the_same_type(noise):
    t = TileFrame.black()
    t.data[:] = 5
    assert type(PixelFrame.like(noise)) is PixelFrame
    assert type(TileFrame.like(noise)) is PixelFrame  # `like` follows the argument
    assert type(PixelFrame.like(t)) is TileFrame
    assert not PixelFrame.like(noise).data.any()
    assert PixelFrame.like(noise).geometry is noise.geometry


def test_equality_compares_type_and_data(noise):
    assert noise == noise.copy()
    assert noise != noise.gain(0.5)
    assert TileFrame.black() != PixelFrame.black()
    assert noise != "not a frame"


# ---- mutation and ownership -------------------------------------------------------


def test_frozen_frame_rejects_writes_but_copies_are_writable(noise):
    prev = noise.freeze()
    assert prev is noise and prev.frozen
    with pytest.raises(ValueError):
        prev.data[0, 0] = 1
    with pytest.raises(ValueError):
        prev.tile(0)[:] = 1
    with pytest.raises(ValueError):
        prev[0, 0] = (1, 1, 1)
    fresh = prev.copy()
    assert not fresh.frozen
    fresh.data[0, 0] = 1
    assert prev.data[0, 0, 0] != 1


def test_returning_the_previous_frame_unmodified_raises(noise):
    prev = noise.freeze()
    with pytest.raises(FrameOwnershipError, match="previous.copy"):
        check_ownership(prev, prev)


def test_returning_a_frame_that_aliases_the_previous_buffer_raises(noise):
    prev = noise.freeze()
    alias = PixelFrame(prev.data, prev.geometry)
    with pytest.raises(FrameOwnershipError, match="aliases"):
        check_ownership(prev, alias)


def test_the_documented_idiom_passes_ownership(noise):
    prev = noise.freeze()
    frame = prev.copy().gain(0.88)
    check_ownership(prev, frame)  # no raise
    check_ownership(prev, PixelFrame.like(prev))
