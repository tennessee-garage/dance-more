from unittest.mock import MagicMock

import numpy as np
import pytest

from df2_pi.animation import AnimationRegistry
from df2_pi.interfacing.external import EXTERNAL_ID, ExternalSource, decode, external_animation
from df2_pi.interfacing.receiver import ExternalFrame, Layout
from df2_pi.pixels import PixelFrame, default_geometry

GEO = default_geometry()


def frame_of(layout: Layout, pixels: np.ndarray, t: float = 0.0, seq: int = 1) -> ExternalFrame:
    """An ExternalFrame whose pixels (n, 3) are packed 170 to a universe."""
    channels = np.zeros((layout.universes, 512), dtype=np.uint8)
    flat = np.zeros((layout.universes * 170, 3), dtype=np.uint8)
    flat[: len(pixels)] = pixels
    channels[:, :510] = flat.reshape(layout.universes, 510)
    return ExternalFrame(channels, layout, "artnet", "10.0.0.9", t, seq)


# ---- decoding --------------------------------------------------------------------------------


def test_tile_mode_is_raster_order_from_the_top_left():
    px = np.zeros((64, 3), dtype=np.uint8)
    px[0] = (255, 0, 0)  # first pixel: top-left of the picture
    out = decode(frame_of(Layout("tile"), px), GEO)
    # The top-left tile in the canonical view is the far row (7), column 0.
    top_left = 7 * 8 + 0
    assert out.data[top_left].tolist() == [[255, 0, 0]] * 60
    assert not out.data[0].any()  # tile 0, nearest the Pi, is the BOTTOM-left


def test_raw_mode_is_chain_order():
    px = np.arange(GEO.led_count * 3, dtype=np.uint32).reshape(-1, 3) % 251
    out = decode(frame_of(Layout("raw"), px.astype(np.uint8)), GEO)
    assert np.array_equal(out.data.reshape(-1, 3), px.astype(np.uint8))


def test_grid_mode_samples_the_image_at_each_led():
    layout = Layout("grid", 2, 2)  # four big quadrants
    px = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)], dtype=np.uint8)
    out = decode(frame_of(layout, px), GEO)
    # Tile 7*8 (top-left of the view) sits in the red quadrant; tile 7 (bottom-right) in white.
    corner = GEO.leds_per_side  # an LED near the top-left corner of the tile, on its top edge
    assert tuple(out.data[7 * 8, corner]) == (255, 0, 0)
    assert tuple(out.data[7, corner]) == (255, 255, 255)
    assert tuple(out.data[0, corner]) == (0, 0, 255)  # bottom-left: blue


def test_a_uniform_grid_decodes_uniform():
    layout = Layout("grid", 34, 34)
    px = np.full((34 * 34, 3), (10, 200, 90), dtype=np.uint8)
    out = decode(frame_of(layout, px), GEO)
    assert (out.data == (10, 200, 90)).all()


# ---- the External animation ------------------------------------------------------------------


def test_the_external_animation_shows_the_latest_frame_and_is_a_builtin(tmp_path):
    receiver = MagicMock()
    receiver.latest.return_value = None
    definition = external_animation(receiver)
    run = definition.start(GEO)
    assert not run.render().frame.data.any()  # nothing received yet: dark
    px = np.full((64, 3), (0, 0, 180), dtype=np.uint8)
    receiver.latest.return_value = frame_of(Layout("tile"), px)
    first = run.render().frame
    second = run.render().frame  # the same frame held, never the object handed back before
    assert first == second and first is not second and (first.data == (0, 0, 180)).all()

    (tmp_path / "external.py").write_text("x = 1\n")
    registry = AnimationRegistry.discover(tmp_path)
    registry.add_builtin(definition)
    assert registry.get(EXTERNAL_ID) is definition and registry.is_builtin(EXTERNAL_ID)
    (tmp_path / "external.py").write_text("from df2_pi.animation import animation\n")
    registry.reload()
    assert registry.get(EXTERNAL_ID) is definition
    assert "reserved for a built-in" in registry.errors[EXTERNAL_ID].message


# ---- the source control ----------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def rig():
    clock = Clock()
    receiver = MagicMock()
    receiver.latest.return_value = None
    runner = MagicMock()
    source = ExternalSource(runner, receiver, source="external", mix=0.4, timeout_s=2.0, now=clock)
    return source, runner, receiver, clock


def arrive(receiver, clock):
    receiver.latest.return_value = frame_of(Layout("tile"), np.zeros((64, 3), dtype=np.uint8), t=clock.t)


def test_signal_takes_over_and_silence_gives_the_floor_back(rig):
    source, runner, receiver, clock = rig
    source.poll()
    runner.play_animation.assert_not_called()  # no signal yet
    arrive(receiver, clock)
    source.poll()
    runner.play_animation.assert_called_once_with(EXTERNAL_ID, hold=None)
    clock.t += 1.9
    source.poll()
    runner.end_one_off.assert_not_called()  # within the timeout
    clock.t += 0.2
    source.poll()
    runner.end_one_off.assert_called_once_with(EXTERNAL_ID)
    assert source.state()["applied"] == "internal" and not source.state()["live"]


def test_mix_runs_external_as_the_layer_and_the_amount_follows(rig):
    source, runner, receiver, clock = rig
    arrive(receiver, clock)
    source.set_source("mix")
    runner.set_layer.assert_called_once_with(EXTERNAL_ID, mode="mix", amount=0.4)
    source.set_mix(0.8)
    runner.set_layer_blend.assert_called_once_with(amount=0.8)
    source.set_source("external")
    runner.clear_layer.assert_called_once_with(EXTERNAL_ID)
    runner.play_animation.assert_called_once_with(EXTERNAL_ID, hold=None)


def test_internal_ignores_the_signal(rig):
    source, runner, receiver, clock = rig
    source.set_source("internal")
    arrive(receiver, clock)
    source.poll()
    runner.play_animation.assert_not_called()
    runner.set_layer.assert_not_called()


def test_bad_values_are_refused(rig):
    source, *_ = rig
    for call in (lambda: source.set_source("both"), lambda: source.set_mix(1.5), lambda: source.set_timeout(0.0)):
        with pytest.raises(ValueError):
            call()
