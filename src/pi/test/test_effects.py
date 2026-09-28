import pytest

from df2_pi.effects import CHASE, FADE, HUE_SPLIT, Effect


def test_rotation_turns_hue_split_offset_round_the_ring():
    effect = Effect(HUE_SPLIT, (20, 50, 0, 0))
    assert effect.rotated(15, 60) == Effect(HUE_SPLIT, (20, 5, 0, 0))
    assert effect.rotated(45, 60).rotated(15, 60) == effect


@pytest.mark.parametrize("effect", [Effect.NONE, Effect(FADE, (230, 0, 0, 0)), Effect(CHASE, (0, 255, 10, 14))])
def test_effects_without_positional_params_are_unchanged_by_rotation(effect):
    assert effect.rotated(15, 60) is effect


def test_a_shift_of_a_whole_ring_or_an_invalid_offset_is_left_alone():
    assert Effect(HUE_SPLIT, (20, 50, 0, 0)).rotated(60, 60) == Effect(HUE_SPLIT, (20, 50, 0, 0))
    # >= 60 is invalid on the wire and the tile drops it; rotation must not make it valid
    assert Effect(HUE_SPLIT, (20, 70, 0, 0)).rotated(15, 60) == Effect(HUE_SPLIT, (20, 70, 0, 0))
