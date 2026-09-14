import numpy as np
import pytest

from df2_pi.encode import DECODE_LUT, GAMMA, from_linear, to_linear

ALL_BYTES = np.arange(256, dtype=np.uint8)


def test_decode_lut_spans_zero_to_one_and_is_strictly_increasing():
    assert DECODE_LUT.shape == (256,)
    assert DECODE_LUT.dtype == np.float32
    assert DECODE_LUT[0] == 0.0
    assert DECODE_LUT[255] == 1.0
    assert (np.diff(DECODE_LUT) > 0).all()


def test_decode_is_a_power_law_not_srgb():
    # 128 is not half as bright as 255 - that is the whole reason this
    # module exists. Check the exact exponent rather than "it curves".
    assert DECODE_LUT[128] == pytest.approx((128 / 255) ** GAMMA, rel=1e-6)
    assert DECODE_LUT[128] < 0.25


def test_from_linear_round_trips_every_byte_value():
    # A LUT construction invariant the frame ops depend on: a uniform tile
    # must reduce to its own colour, and blend(a, b, 0) must be exactly a.
    assert (from_linear(to_linear(ALL_BYTES)) == ALL_BYTES).all()


def test_from_linear_saturates_rather_than_wrapping():
    out = from_linear(np.array([-0.5, 0.0, 1.0, 1.5, 100.0]))
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 0, 255, 255, 255]


def test_half_light_encodes_to_186():
    # The number the frame tests lean on: the linear midpoint between
    # black and full white, re-encoded.
    assert from_linear(np.array(0.5)) == 186


def test_to_linear_preserves_shape_and_rejects_non_uint8():
    data = np.zeros((4, 5, 3), dtype=np.uint8)
    assert to_linear(data).shape == (4, 5, 3)
    with pytest.raises(TypeError):
        to_linear(data.astype(np.float32))
