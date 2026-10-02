import math

import numpy as np
import pytest

from df2_pi.tempo import SHAPES, lfo, pulse


@pytest.mark.parametrize(
    "shape, quarter_points",
    [
        ("sine", [0.0, 0.5, 1.0, 0.5]),
        ("tri", [0.0, 0.5, 1.0, 0.5]),
        ("saw", [0.0, 0.25, 0.5, 0.75]),
        ("ramp_down", [1.0, 0.75, 0.5, 0.25]),
        ("square", [1.0, 1.0, 0.0, 0.0]),
    ],
)
def test_each_shape_over_one_beat(shape, quarter_points):
    assert [lfo(t, shape=shape) for t in (0.0, 0.25, 0.5, 0.75)] == pytest.approx(quarter_points)
    assert lfo(1.0, shape=shape) == pytest.approx(lfo(0.0, shape=shape))  # periodic


def test_rate_is_cycles_per_beat_and_offset_shifts_in_cycles():
    assert lfo(2.0, rate=0.25, shape="saw") == pytest.approx(0.5)  # once a bar: halfway at beat 2
    assert lfo(0.25, rate=2.0, shape="saw") == pytest.approx(0.5)  # twice a beat
    assert lfo(0.0, shape="saw", offset=0.25) == pytest.approx(0.25)
    assert lfo(-0.25, shape="saw") == pytest.approx(0.75)  # before zero wraps, not negative


def test_every_shape_stays_in_0_to_1():
    t = np.linspace(-3, 7, 997)
    for shape in SHAPES:
        values = lfo(t, rate=1.3, shape=shape, offset=0.17)
        assert values.min() >= 0.0 and values.max() <= 1.0


def test_arrays_in_arrays_out_scalars_in_floats_out():
    offsets = np.array([[0.0, 0.25], [0.5, 0.75]])
    out = lfo(0.0 - offsets, shape="saw")
    assert isinstance(out, np.ndarray) and out.shape == (2, 2)
    np.testing.assert_allclose(out, [[0.0, 0.75], [0.5, 0.25]])
    assert isinstance(lfo(0.3), float) and isinstance(pulse(0.3), float)
    assert pulse(np.array([0.0, 1.0, 2.5])).shape == (3,)
    per_tile = lfo(2.0, shape="saw", offset=-np.array([0.0, 0.25, 0.5]))  # one time, an offset per tile
    np.testing.assert_allclose(per_tile, [0.0, 0.75, 0.5])


def test_pulse_is_one_on_the_beat_and_decays_by_its_time_constant():
    assert pulse(3.0) == pytest.approx(1.0)
    assert pulse(3.25, decay=0.25) == pytest.approx(math.exp(-1))
    assert pulse(3.5, decay=0.25) == pytest.approx(math.exp(-2))
    assert pulse(1.0, rate=0.5) == pytest.approx(math.exp(-4))  # every other beat: beat 1 is a beat after the last pulse
    assert pulse(2.0, rate=0.5) == pytest.approx(1.0)


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        lfo(0.0, shape="wobble")
    with pytest.raises(ValueError):
        pulse(0.0, decay=0)
    with pytest.raises(ValueError):
        pulse(0.0, rate=0)
