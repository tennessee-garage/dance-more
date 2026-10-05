"""The starter pack is the driver's acceptance test: every animation loads,
renders headless, honours its declared format, and reproduces from a seed."""

import os
import sys
import time

import numpy as np
import pytest

from df2_pi.animation import AnimationRegistry, default_animations_dir
from df2_pi.animation.loader import MODULE_PREFIX
from df2_pi.edges import Axis
from df2_pi.pixels import PixelFrame, TileFrame, default_geometry
from df2_pi.playlists import PlaylistStore

FRAMES = 90
PACK = {"solid", "rainbow_sweep", "checkerboard", "plasma", "ripple", "lightning", "chase", "seams", "comet_squares", "vortex", "twin_peaks", "spiral", "stardust", "stripes", "waves", "video", "waterline", "comet_train", "switchyard"}


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


@pytest.mark.parametrize("animation_id", sorted(PACK))
def test_each_puts_a_param_under_macro_1(registry, animation_id):
    """So a knob bound to macro 1 does something whatever is playing."""
    assert registry[animation_id].meta.control("macro1") is not None


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


def test_comet_squares_land_as_whole_uniform_tiles(registry):
    """Each side only fills in at full strength once its comet lands, so a
    formed square is one tile's 60 LEDs at a single colour."""
    geo = default_geometry()
    run = registry["comet_squares"].start(seed=0, params={"squares": 1, "speed": 600.0})
    for _ in range(FRAMES):
        frame = run.render().frame
        for tile in range(geo.tiles):
            ring = frame.flat[tile * geo.leds_per_tile + geo.tile_ring(tile)]
            if ring.max() == 255 and (ring == ring[0]).all():
                return
    pytest.fail("no square formed")


def test_vortex_rings_are_concentric_closed_loops(registry):
    geo = default_geometry()
    rings = sys.modules[MODULE_PREFIX + "vortex"]._rings(geo)
    np.testing.assert_array_equal(rings[0], geo.floor_ring)
    assert [len(r) for r in rings] == [480, 360, 360, 240, 240, 120, 120, 120]
    every = np.concatenate(rings)
    assert len(set(every.tolist())) == len(every)  # no LED on two rings
    cells = geo.led_positions.reshape(-1, 2)
    for ring in rings:
        steps = np.linalg.norm(np.diff(cells[np.append(ring, ring[0])], axis=0), axis=1)
        assert steps.max() <= 3.0 + 1e-9  # neighbours, or across a pair of dark corner cells


def test_vortex_lights_only_its_rings(registry):
    geo = default_geometry()
    on_rings = set(np.concatenate(sys.modules[MODULE_PREFIX + "vortex"]._rings(geo)).tolist())
    run = registry["vortex"].start(seed=0)
    for _ in range(FRAMES):
        assert set(np.flatnonzero(run.render().frame.flat.any(axis=-1)).tolist()) <= on_rings


def test_twin_peaks_curtains_are_red_and_never_black(registry):
    run = registry["twin_peaks"].start(seed=0, params={"flutter": 1.0, "flicker": 0.0})
    for _ in range(FRAMES):
        data = run.render().frame.data.astype(int)
        assert data[..., 0].min() > 0  # every LED lit: the floor shows black poorly
        assert (data[..., 0] > data[..., 1]).all() and (data[..., 0] > data[..., 2]).all()


def test_spiral_moves_each_colour_one_tile_outward_per_step(registry):
    spiral = sys.modules[MODULE_PREFIX + "spiral"]
    path = list(zip(spiral.PATH_ROWS, spiral.PATH_COLS))
    assert len(set(path)) == 64 and path[0] == (4, 3)  # every tile once, from the middle
    run = registry["spiral"].start(params={"speed": 30.0})  # one step per frame at 30 fps
    frames = [run.render().frame for _ in range(80)]
    for n in range(64, 80):
        for k in range(64):
            assert frames[n][path[k]].tolist() == frames[n - k][path[0]].tolist()


def test_checkerboard_flips_on_the_beat_when_there_is_one(registry):
    from df2_pi.animation import BeatInfo

    run = registry["checkerboard"].start(seed=0, params={"interval": 4.0})  # far slower than the beat
    tiles = []
    for frame in range(12):
        beat = BeatInfo(tempo=120.0, phase=0.0, beat=frame // 3, bar_phase=0.0, beats_per_bar=4, downbeat=False)
        tiles.append(run.render(beat=beat).frame[0, 0].tolist())
    changes = [i for i in range(1, 12) if tiles[i] != tiles[i - 1]]
    assert changes == [3, 6, 9]  # a flip on each new beat, and only then


def test_stardust_never_goes_dark_and_a_nova_glows_then_scatters(registry):
    geo = default_geometry()
    run = registry["stardust"].start(seed=4, params={"novae": 12.0, "stars": 0})
    glowed = scattered = False
    for _ in range(600):
        level = run.render().frame.data.max(axis=-1)  # (64, 60)
        assert level.min() > 0  # the floor shows black poorly: the sky is always there
        whole = [t for t in range(geo.tiles) if level[t].min() > 100]  # a tile glowing edge to edge
        glowed = glowed or bool(whole)
        bright = level > 100
        partial = [t for t in range(geo.tiles) if 0 < bright[t].sum() < geo.leds_per_tile]
        scattered = scattered or (glowed and bool(partial))  # specks: some of a tile's LEDs, not all
    assert glowed and scattered


def test_stripes_fade_as_one_over_n_in_light_either_side_of_the_peak(registry):
    from df2_pi.gamma import to_linear

    run = registry["stripes"].start(seed=2, params={"palette": "rygw", "length": 20})
    checked = 0
    for _ in range(600):
        light = to_linear(run.render().frame.data).max(axis=-1)  # (8, 8) linear light, brightest channel
        for row in light:
            peak = int(row.argmax())
            if row[peak] > 0.98 and 1 <= peak <= 6:  # the peak itself is on the row, with tiles both sides
                assert row[peak - 1] == pytest.approx(0.5, abs=0.02) and row[peak + 1] == pytest.approx(0.5, abs=0.02)
                checked += 1
    assert checked > 10


def test_ripple_drops_a_ring_where_a_trigger_says(registry):
    from df2_pi.animation import Trigger

    run = registry["ripple"].start(seed=0, params={"rate": 0.1})
    run.render()
    frame = run.render(triggers=(Trigger(slot=15, velocity=1.0, age_s=0.0),)).frame  # slot 15: the far corner of the grid
    lit = frame.grid.max(axis=-1) > 120
    ys, xs = np.nonzero(lit)
    assert len(ys) and ys.mean() > 136 * 0.6 and xs.mean() > 136 * 0.6


def test_rainbow_sweep_pumps_on_the_beat_and_follows_beat_time(registry):
    from df2_pi.animation import BeatInfo

    run = registry["rainbow_sweep"].start(params={"pump": 0.5})
    on = run.render(beat=BeatInfo(tempo=120.0, phase=0.0, beat=4, bar_phase=0.0, beats_per_bar=4, downbeat=True)).frame
    off = run.render(beat=BeatInfo(tempo=120.0, phase=0.9, beat=4, bar_phase=0.225, beats_per_bar=4, downbeat=False)).frame
    assert on.data.max() == 255 and off.data.max() < 160  # full on the beat, dipped by pump just before the next


def test_waves_follow_their_phase_map(registry):
    geo = default_geometry()
    level = lambda frame: frame.data.max(axis=-1).astype(int)  # (64, 60)

    rows = level(registry["waves"].start(params={"map": "rows", "shape": "saw"}).render().frame).reshape(8, 8, 60)
    assert all(len(np.unique(row)) == 1 for row in rows)  # a row is one level...
    assert len({int(row.flat[0]) for row in rows}) == 8  # ...and every row a different one

    together = level(registry["waves"].start(params={"map": "radial", "spread": 0.0}).render().frame)
    assert len(np.unique(together)) == 1  # spread 0: the whole floor in phase

    ring = level(registry["waves"].start(params={"map": "perimeter", "shape": "saw"}).render().frame)
    assert (ring == ring[0]).all() and len(np.unique(ring[0])) > 30  # every tile the same ring, varying round it


def test_palette_opt_ins_follow_the_floor_palette_or_a_named_one(registry):
    from df2_pi.palette import BUILTIN, Palette

    mine = Palette(["00ff00", "0000ff"], "mine")
    heads = lambda frame: {tuple(c) for c in frame.flat[frame.flat.max(axis=-1) == 255].tolist()}

    floor = registry["chase"].start(params={"comets": 2}).render(palette=mine).frame
    assert heads(floor) == {(0, 255, 0), (0, 0, 255)}  # comet k of n at k/n round the floor's palette
    named = registry["chase"].start(params={"comets": 2, "palette": "fire"}).render(palette=mine).frame
    assert heads(named) == {tuple(BUILTIN["fire"].at(0.0).tolist()), tuple(BUILTIN["fire"].at(0.5).tolist())}

    waves = registry["waves"].start(params={"spread": 0.0, "hue_spread": 0.0, "hue": 0.0, "shape": "square"}).render(palette=mine).frame  # square: full at beat 0
    assert {tuple(c) for c in waves.flat.tolist()} == {(0, 255, 0)}  # the floor palette's first stop, everywhere


def test_waterline_draws_a_surface_over_fading_depths(registry):
    run = registry["waterline"].start(seed=1)
    for _ in range(90):
        frame = run.render().frame
    level = frame.data.max(axis=-1).astype(int)  # (rows, cols), row 0 the bottom
    heights = run.state["y"]
    for col, y in enumerate(heights):
        top = int(np.floor(y)) + 2
        assert (level[top:, col] == 0).all()  # above the surface: dark
        assert level[0, col] == 0  # the bottom row: faded to black
        assert level[int(np.round(y)), col] > level[0, col]


def test_waterline_cohesion_holds_neighbours_together(registry):
    def spread(cohesion):
        run = registry["waterline"].start(seed=3, params={"cohesion": cohesion})
        gaps = []
        for _ in range(900):
            run.render()
            gaps.append(np.abs(np.diff(run.state["y"])).max())
        return np.percentile(gaps, 90)

    loose, tight = spread(0.0), spread(1.0)
    assert tight < 0.6 and loose > 2 * tight


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


def test_lightning_comes_down_and_forks_late(registry):
    import random

    lightning = sys.modules[MODULE_PREFIX + "lightning"]
    geo = default_geometry()
    rng = random.Random(11)
    fork_points = []
    for _ in range(300):
        trunk, forks = lightning.strike(geo, rng, {"length": 14, "turn_bias": 0.35, "forks": 2})
        rows = [geo.tile_rows] + [end[0] for _, end in trunk]
        assert all(b <= a for a, b in zip(rows, rows[1:]))  # never climbs
        assert rows[1] == geo.tile_rows - 1  # leaves the top wall heading straight down
        corners = [end for _, end in trunk]
        for fork in forks:
            (edge, forward), _ = fork[0]
            origin = edge.junctions[0] if forward else edge.junctions[1]
            fork_points.append((corners.index(origin) + 1) / len(trunk))
            fork_rows = [origin[0]] + [end[0] for _, end in fork]
            assert all(b <= a for a, b in zip(fork_rows, fork_rows[1:]))
    assert min(fork_points) > 0.2  # nothing forks in the top fifth of a bolt
    assert np.median(fork_points) > 0.65  # and most fork in its last third


def test_comet_train_rests_with_one_comet_on_every_edge_running_one_way(registry):
    """Between pulses (a pulse is 15 frames at the fallback 120 bpm), with
    a whole tile side per pulse, each edge running the comets' way holds
    one: a near-white head at its leading end and `tail` LEDs fading behind
    it; the other way is dark."""
    geo = default_geometry()
    run = registry["comet_train"].start(seed=0, params={"tail": 6, "turns": 0.5, "step": 15})
    axes = set()
    for f in range(150):
        frame = run.render().frame
        if f % 15 != 14:
            continue
        move = run.state["move"]
        axes.add(move["axis"])
        for edge in geo.edges:
            leds = edge.flat_leds if move["dir"] > 0 else edge.flat_leds[::-1]  # head last
            level = frame.flat[leds].max(axis=-1).astype(int)
            if edge.axis is not move["axis"]:
                assert not level.any()
                continue
            assert frame.flat[leds[-1]].min() > 200  # the head, near white
            assert (np.diff(level[-7:-1]) > 0).all()  # the tail brightening towards it
            assert not level[:-7].any()  # and nothing further back
    assert len(axes) == 2  # it turned


def test_comet_train_turns_every_comet_onto_its_own_edge(registry):
    """A turn maps lanes to lanes: no two comets land on the same edge, and
    every edge the comets now run along is filled."""
    run = registry["comet_train"].start(seed=1, params={"turns": 1.0, "step": 15})
    axes = []
    for f in range(15 * 6):
        run.render()
        if f % 15 == 0:
            move = run.state["move"]
            lanes, edges = move["shape"]
            landing = [(j, f) for _, _, j, f in move["routes"] if 0 <= j < lanes and 0 <= f < edges]
            assert len(landing) == len(set(landing)) == 128
            axes.append(move["axis"])
    assert all(a is not b for a, b in zip(axes, axes[1:]))  # every pulse a turn


def test_comet_train_steps_a_few_leds_a_pulse(registry):
    """Each pulse moves every head three LEDs along its line; new heads
    only appear where comets come in, at the start of the line."""
    geo = default_geometry()
    run = registry["comet_train"].start(seed=2, params={"turns": 0.0, "step": 3})
    heads = []
    for f in range(15 * 6):
        frame = run.render().frame
        if f % 15 == 14:
            heads.append(np.flatnonzero(frame.flat.min(axis=-1) > 200))
    axis, d = run.state["axis"], run.state["dir"]
    for lane in range(2 * (geo.tile_rows if axis.value == "x" else geo.tile_cols)):
        rail = geo.rails(axis, lane)
        if d < 0:
            rail = rail[::-1]  # in the way the comets run
        along = [set(np.flatnonzero(np.isin(rail, h)).tolist()) for h in heads]
        for before, after in zip(along, along[1:]):
            moved = {i + 3 for i in before if i + 3 < len(rail)}
            assert moved <= after and all(i < 3 for i in after - moved)


def test_comet_train_turns_round_the_corner_led_by_led(registry):
    """A turn is not one jump: the new way fills in over most of a pulse."""
    run = registry["comet_train"].start(seed=3, params={"turns": 1.0, "step": 15})
    run.render()  # the first pulse sets off
    move = run.state["move"]
    assert move["turn"]
    geo = default_geometry()
    new = np.concatenate([e.flat_leds for e in geo.edges if e.axis is move["axis"]])
    counts = [int(run.render().frame.flat[new].any(axis=-1).sum()) for _ in range(14)]
    assert all(0 < b - a <= 2 * 128 for a, b in zip(counts, counts[1:12]))  # an LED or so per comet per frame
    assert counts[-1] == 128 * 15  # then every comet is round


def test_switchyard_throws_switches_then_takes_them_out_never_doubling_up_a_lane(registry):
    run = registry["switchyard"].start(seed=4, params={"interval": 10.0, "switches": 2})
    phases = []
    for _ in range(30 * 60):
        run.render()
        state = run.state
        if not phases or phases[-1] != (state["phase"], len(state["saved"])):
            phases.append((state["phase"], len(state["saved"])))
        lanes = [key for key, _ in state["segment"]["keys"] if key is not None]
        assert len(lanes) == len(set(lanes))  # no two comets heading for one lane
        assert set(lanes) == {(link, side) for link in state["flows"] for side in (1, -1)}  # and none left empty
    assert phases == [("train", 0), ("build", 1), ("build", 2), ("unwind", 1), ("train", 0)]
    assert state["routes"] == {}
    assert set(state["flows"].values()) in ({1}, {-1}) and len({link[0] for link in state["flows"]}) == 1  # uniform again


def test_a_switch_jogs_the_lines_it_crosses_and_feeds_the_lines_it_starved(registry):
    """Comets running east, switched north at the corner (3, 4): the column
    above swaps at every corner, and the edges east of the switch are fed
    from a link of comets born at the corner below."""
    import random

    sy = sys.modules[MODULE_PREFIX + "switchyard"]
    lattice = sy.Lattice(default_geometry())
    state = {"flows": {link: 1 for link in lattice.links(Axis.X)}, "routes": {}, "fronts": []}
    sy.switch(state, lattice, (3, 4), "W", "N")
    while state["fronts"]:
        j, come = state["fronts"].pop()
        state["fronts"] += sy._arrive(state, lattice, j, come, random.Random(0))

    assert state["routes"][((3, 4), "W")] == "N"
    assert state["flows"][("v", 4, 2)] == 1 and state["routes"][((3, 4), "S")] == "E"  # the feed, from (2, 4)
    for r in range(4, lattice.rows + 1):  # every line above jogs north at column line 4
        assert state["routes"][((r, 4), "W")] == "N" and state["routes"][((r, 4), "S")] == "E"
    assert all(j[0] >= 3 for j, _ in state["routes"])  # the lines below carry straight on
    vertical = {link for link in state["flows"] if link[0] == "v"}
    assert vertical == {("v", 4, r) for r in range(2, lattice.rows)}


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
