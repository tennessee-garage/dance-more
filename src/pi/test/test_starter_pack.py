"""The starter pack is the driver's acceptance test: every animation loads,
renders headless, honours its declared format, and reproduces from a seed."""

import os
import time

import numpy as np
import pytest

from df2_pi.animation import AnimationRegistry, default_animations_dir
from df2_pi.pixels import PixelFrame, TileFrame, default_geometry
from df2_pi.playlists import PlaylistStore

FRAMES = 90
PACK = {"solid", "rainbow_sweep", "checkerboard", "plasma", "ripple", "lightning", "chase", "seams"}


@pytest.fixture(scope="module")
def registry() -> AnimationRegistry:
    return AnimationRegistry.discover(default_animations_dir())


def test_every_animation_in_the_pack_loads_with_no_errors(registry):
    assert registry.errors == {}
    assert set(registry.animations) == PACK


def test_the_template_is_skipped_by_discovery(registry):
    assert (default_animations_dir() / "_template.py").exists()
    assert "_template" not in registry.animations and "_template" not in registry.errors


@pytest.mark.parametrize("animation_id", sorted(PACK))
def test_each_renders_90_frames_in_its_declared_format_and_never_writes_a_dark_cell(registry, animation_id):
    definition = registry[animation_id]
    expected = TileFrame if definition.meta.format == "tile" else PixelFrame
    geo = default_geometry()
    run = definition.start(seed=1)
    lit_total = 0
    for _ in range(FRAMES):
        rendered = run.render()
        frame = rendered.frame
        assert isinstance(frame, expected)
        assert frame.data.dtype == np.uint8 and frame.data.shape == expected.shape_for(geo)
        assert frame.frozen
        if isinstance(frame, PixelFrame):
            assert not frame.grid[~geo.lit_mask].any()  # a PixelFrame cannot address a dark cell
        lit_total += int(frame.data.any(axis=-1).sum())
    assert lit_total > 0  # it drew something


@pytest.mark.parametrize("animation_id", sorted(PACK))
def test_each_is_reproducible_frame_for_frame_from_the_same_seed(registry, animation_id):
    definition = registry[animation_id]
    a = definition.start(seed=42)
    b = definition.start(seed=42)
    for _ in range(FRAMES):
        assert a.render().frame == b.render().frame


def test_the_random_ones_differ_across_seeds(registry):
    for animation_id in ("lightning", "ripple", "checkerboard"):
        definition = registry[animation_id]
        a = definition.start(seed=1)
        b = definition.start(seed=2)
        assert any(a.render().frame != b.render().frame for _ in range(FRAMES)), animation_id


def test_params_are_honoured(registry):
    solid = registry["solid"]
    red = solid.start(params={"hue": 0.0}).render().frame
    assert red[0, 0].tolist() == [255, 0, 0]
    with pytest.raises(ValueError):
        solid.start(params={"hue": 2.0})
    chase = registry["chase"].start(params={"comets": 1, "tail": 5})
    frame = chase.render().frame
    assert int(frame.data.any(axis=-1).sum()) == 5


def test_seams_light_both_halves_of_every_seam_identically(registry):
    geo = default_geometry()
    frame = registry["seams"].start(params={"base": 20}).render().frame
    for a, b in geo.seams:
        np.testing.assert_array_equal(frame.flat[a.flat_leds], frame.flat[b.flat_leds])
    assert all(frame.flat[e.flat_leds].max() >= 20 for e in (edge for pair in geo.seams for edge in pair))
    assert not any(frame.flat[e.flat_leds].any() for e in geo.edges if e.outer)  # outer edges are not seams


def test_chase_stays_on_the_floor_ring(registry):
    geo = default_geometry()
    run = registry["chase"].start(seed=0)
    ring = set(geo.floor_ring.tolist())
    for _ in range(30):
        lit = np.flatnonzero(run.render().frame.flat.any(axis=-1))
        assert set(lit.tolist()) <= ring


def test_lightning_bolts_run_along_edges(registry):
    geo = default_geometry()
    run = registry["lightning"].start(seed=3, params={"rate": 5.0, "decay": 0.3})
    struck = False
    for _ in range(FRAMES):
        frame = run.render().frame
        lit = np.flatnonzero(frame.flat.max(axis=-1) > 150)  # the bolt itself, not afterglow
        if len(lit) >= 15:
            struck = True
            # every fully lit LED belongs to an edge that is entirely lit: bolts are whole edges
            lit_set = set(lit.tolist())
            covered = {e.index for e in geo.edges if set(e.flat_leds.tolist()) <= lit_set}
            assert covered, "a strike should light whole edges"
    assert struck


def test_seeding_a_fresh_database_gets_the_whole_pack(registry):
    store = PlaylistStore(":memory:", registry=registry)
    pl = store.seed_default()
    assert [e.animation_id for e in pl.entries] == sorted(PACK)
    assert store.startup_playlist() == pl
    resolved = store.resolve(pl)
    assert all(r.playable for r in resolved.entries)
    store.close()


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("DF2_SLOW_TESTS"), reason="set DF2_SLOW_TESTS=1 to run")
@pytest.mark.parametrize("animation_id", sorted(PACK))
def test_each_stays_inside_the_frame_budget(registry, animation_id):
    # A generous ceiling: this catches an accidental O(n^2), it is not a benchmark.
    run = registry[animation_id].start(seed=1)
    times = []
    for _ in range(FRAMES):
        t0 = time.perf_counter()
        run.render()
        times.append(time.perf_counter() - t0)
    assert np.percentile(times, 95) < 0.010, f"{animation_id} p95 {np.percentile(times, 95) * 1000:.1f} ms"
