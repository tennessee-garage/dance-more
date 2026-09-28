import numpy as np
import pytest

from df2_pi.engine.overlays import Overlays, check_strobe_max, color_matrix
from df2_pi.pixels import PixelFrame, TileFrame, default_geometry

FPS = 30.0


def tiles(rgb=(200, 100, 50)) -> TileFrame:
    frame = TileFrame.black(default_geometry())
    frame.data[:] = rgb
    return frame


def test_nothing_active_returns_the_frame_itself():
    frame = tiles()
    assert Overlays().apply(frame, 0.0) is frame


@pytest.mark.parametrize(
    "setup",
    [
        lambda o: o.set_hue_shift(0.25),
        lambda o: o.set_saturation(0.0),
        lambda o: o.set_tint((255, 0, 0), 0.5),
        lambda o: o.bump(1.0, 0.25, 0.0),
        lambda o: o.set_strobe(5.0),
        lambda o: o.set_freeze(True),
    ],
)
def test_every_overlay_keeps_the_frame_format(setup):
    for frame in (tiles(), PixelFrame(np.full((64, 60, 3), 120, dtype=np.uint8))):
        o = Overlays()
        setup(o)
        assert type(o.apply(frame, 0.0)) is type(frame)


# ---- colour correction ---------------------------------------------------------------------------


def test_identity_matrix_at_no_shift_and_full_saturation():
    assert np.allclose(color_matrix(0.0, 1.0), np.eye(3), atol=1e-3)


def test_a_third_of_a_turn_takes_red_toward_green():
    o = Overlays()
    o.set_hue_shift(1 / 3)
    r, g, b = o.apply(tiles((255, 0, 0)), 0.0).data[0, 0]
    assert g > 80 and r == 0 and b == 0


def test_zero_saturation_is_grey_and_a_full_turn_wraps():
    o = Overlays()
    o.set_saturation(0.0)
    r, g, b = (int(v) for v in o.apply(tiles((200, 40, 90)), 0.0).data[0, 0])
    assert abs(r - g) <= 1 and abs(g - b) <= 1
    o.reset()
    o.set_hue_shift(1.0)
    assert o.hue_shift == 0.0


# ---- tint ------------------------------------------------------------------------------------------


def test_tint_keeps_black_black_and_turns_white_into_the_tint():
    o = Overlays()
    o.set_tint((255, 0, 0), 1.0)
    assert o.apply(tiles((0, 0, 0)), 0.0).data[0, 0].tolist() == [0, 0, 0]
    assert o.apply(tiles((255, 255, 255)), 0.0).data[0, 0].tolist() == [255, 0, 0]


def test_half_a_tint_is_between():
    o = Overlays()
    o.set_tint((0, 0, 255), 0.5)
    r, g, b = (int(v) for v in o.apply(tiles((255, 255, 255)), 0.0).data[0, 0])
    assert 0 < r < 255 and b == 255


# ---- bump ------------------------------------------------------------------------------------------


def test_a_bump_flashes_white_and_fades_over_its_decay():
    o = Overlays()
    o.bump(1.0, 0.3, t=10.0)
    black = tiles((0, 0, 0))
    assert o.apply(black, 10.0).data[0, 0].tolist() == [255, 255, 255]
    middle = int(o.apply(black, 10.15).data[0, 0, 0])
    assert 0 < middle < 255
    assert o.apply(black, 10.3) is black  # faded out: nothing active


def test_a_weaker_bump_does_not_cut_a_stronger_one_short():
    o = Overlays()
    o.bump(1.0, 1.0, t=0.0)
    o.bump(0.2, 1.0, t=0.1)
    assert o._bump_now(0.1) == pytest.approx(0.9)


# ---- strobe -----------------------------------------------------------------------------------------


def lit(o: Overlays, n: int) -> list[int]:
    frame = tiles((255, 255, 255))
    return [int(o.apply(frame, k / FPS).data.any()) for k in range(n)]


def test_strobe_shows_one_frame_per_period():
    o = Overlays()
    o.set_strobe(10.0)
    assert lit(o, 9) == [1, 0, 0] * 3


def test_strobe_is_held_to_the_cap():
    o = Overlays(strobe_max_hz=5.0)
    o.set_strobe(30.0)
    assert lit(o, 12) == [1, 0, 0, 0, 0, 0] * 2
    o.set_strobe_max(0.0)
    assert lit(o, 3) == [1, 1, 1]


def test_the_cap_has_its_own_ceiling():
    assert check_strobe_max(15) == 15.0
    for bad in (-1, 16):
        with pytest.raises(ValueError):
            check_strobe_max(bad)


# ---- freeze ------------------------------------------------------------------------------------------


def test_freeze_holds_the_picture_and_later_controls_still_act_on_it():
    o = Overlays()
    o.set_freeze(True)
    first = tiles((10, 20, 30))
    assert o.apply(first, 0.0) is first
    assert o.apply(tiles((200, 0, 0)), 0.1) is first
    o.set_saturation(0.0)
    held = o.apply(tiles((200, 0, 0)), 0.2).data[0, 0]
    assert abs(int(held[0]) - int(held[2])) <= 1  # the held picture, desaturated
    o.set_freeze(False)
    live = tiles((0, 200, 0))
    assert o.apply(live, 0.3).data[0, 0, 1] > 0


def test_reset_clears_every_control_but_keeps_the_cap():
    o = Overlays(strobe_max_hz=6.0)
    o.set_speed(2.0)
    o.set_freeze(True)
    o.set_tint((1, 2, 3), 0.5)
    o.set_strobe(3.0)
    o.reset()
    state = o.state()
    assert (state.speed, state.frozen, state.tint_amount, state.strobe_hz, state.strobe_max_hz) == (1.0, False, 0.0, 0.0, 6.0)
