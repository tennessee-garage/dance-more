import math

import pytest

from df2_pi.animation import MACROS, ROLES, AnimationMeta, Param
from df2_pi.animation.meta import check_control_target, macro_number


# ---- from_unit / to_unit ------------------------------------------------------------------------


def test_linear_float_maps_across_min_to_max():
    p = Param(float, default=1.0, min=0.5, max=2.5)
    assert [p.from_unit(u) for u in (0.0, 0.5, 1.0)] == [0.5, 1.5, 2.5]


def test_int_rounds_to_the_nearest_step():
    p = Param(int, default=3, min=1, max=12)
    assert [p.from_unit(u) for u in (0.0, 0.5, 1.0)] == [1, 6, 12]  # 6.5 rounds to even
    assert isinstance(p.from_unit(0.3), int)


def test_log_curve_puts_the_geometric_mean_at_the_middle():
    p = Param(float, default=1.0, min=0.1, max=10.0, curve="log")
    assert p.from_unit(0.0) == pytest.approx(0.1)
    assert p.from_unit(0.5) == pytest.approx(1.0)
    assert p.from_unit(1.0) == pytest.approx(10.0)


def test_out_of_range_units_are_clamped():
    p = Param(float, default=1.0, min=0.0, max=2.0)
    assert (p.from_unit(-3), p.from_unit(7)) == (0.0, 2.0)


def test_choices_split_the_range_into_equal_slots():
    p = Param(str, default="a", choices=["a", "b", "c", "d"])
    assert [p.from_unit(u) for u in (0.0, 0.24, 0.25, 0.74, 0.75, 1.0)] == ["a", "a", "b", "c", "d", "d"]
    assert [p.to_unit(c) for c in "abcd"] == [0.125, 0.375, 0.625, 0.875]


def test_bool_switches_at_the_middle():
    p = Param(bool, default=False)
    assert (p.from_unit(0.49), p.from_unit(0.5)) == (False, True)
    assert (p.to_unit(False), p.to_unit(True)) == (0.0, 1.0)


@pytest.mark.parametrize(
    "param",
    [
        Param(float, default=1.0, min=0.5, max=2.5),
        Param(float, default=1.0, min=0.1, max=10.0, curve="log"),
        Param(str, default="b", choices=["a", "b", "c"]),
    ],
)
def test_to_unit_inverts_from_unit(param):
    for u in (0.0, 0.1, 0.5, 0.9, 1.0):
        value = param.from_unit(u)
        back = param.from_unit(param.to_unit(value))
        if param.choices:
            assert back == value
        else:
            assert back == pytest.approx(value)


@pytest.mark.parametrize("param", [Param(str, default="x"), Param(float, default=1.0, min=0.0), Param(int, default=1)])
def test_a_param_without_a_range_cannot_be_mapped(param):
    assert not param.mappable
    with pytest.raises(ValueError):
        param.from_unit(0.5)
    with pytest.raises(ValueError):
        param.to_unit(1)


# ---- validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(role="loudness"),
        dict(macro=0),
        dict(macro=MACROS + 1),
        dict(macro=True),
        dict(curve="exp"),
    ],
)
def test_bad_role_macro_or_curve_is_refused(kwargs):
    with pytest.raises(ValueError):
        Param(float, default=1.0, min=0.1, max=2.0, **kwargs)


def test_a_log_curve_needs_a_positive_minimum():
    with pytest.raises(ValueError, match="log"):
        Param(float, default=1.0, min=0.0, max=2.0, curve="log")
    with pytest.raises(ValueError, match="log"):
        Param(float, default=1.0, min=0.1, curve="log")


def test_a_role_or_macro_needs_a_mappable_param():
    with pytest.raises(ValueError, match="role or macro"):
        Param(str, default="x", role="variation")
    with pytest.raises(ValueError, match="role or macro"):
        Param(float, default=1.0, macro=1)


# ---- roles and macros on an animation ---------------------------------------------------------


def meta(**params) -> AnimationMeta:
    return AnimationMeta(name="T", params=params)


def test_a_param_named_after_a_role_has_that_role():
    m = meta(speed=Param(float, default=1.0, min=0.1, max=4.0), rate=Param(float, default=1.0, min=0.1, max=4.0, role="density"))
    assert m.roles == {"speed": "speed", "density": "rate"}
    assert m.control("speed") == "speed" and m.control("density") == "rate"
    assert m.control("hue") is None


def test_an_explicit_role_overrides_the_name():
    m = meta(speed=Param(float, default=1.0, min=0.1, max=4.0, role="intensity"))
    assert m.roles == {"intensity": "speed"}


def test_a_param_named_after_a_role_but_unmappable_has_no_role():
    assert meta(hue=Param(float, default=0.5)).roles == {}


def test_two_params_cannot_share_a_role_or_a_macro():
    with pytest.raises(ValueError, match="role 'speed'"):
        meta(speed=Param(float, default=1.0, min=0.1, max=4.0), pace=Param(float, default=1.0, min=0.1, max=4.0, role="speed"))
    with pytest.raises(ValueError, match="macro 2"):
        meta(a=Param(int, default=1, min=0, max=9, macro=2), b=Param(int, default=1, min=0, max=9, macro=2))


def test_macros_resolve_by_number():
    m = meta(tail=Param(int, default=40, min=2, max=200, macro=1), speed=Param(float, default=1.0, min=0.1, max=4.0, macro=2))
    assert (m.control("macro1"), m.control("macro2"), m.control("macro3")) == ("tail", "speed", None)


def test_sync_and_triggers_are_validated():
    assert AnimationMeta(name="T", sync="beat", triggers=True).sync == "beat"
    with pytest.raises(ValueError):
        AnimationMeta(name="T", sync="bar")
    with pytest.raises(TypeError):
        AnimationMeta(name="T", triggers="yes")


def test_control_targets_are_roles_or_macros():
    for target in (*ROLES, "macro1", f"macro{MACROS}"):
        assert check_control_target(target) == target
    for bad in ("macro0", f"macro{MACROS + 1}", "macro", "tempo", ""):
        with pytest.raises(ValueError):
            check_control_target(bad)
    assert macro_number("macro3") == 3
    assert math.isclose(Param(float, default=1.0, min=0.1, max=10.0, curve="log").to_unit(1.0), 0.5)
