import os
import textwrap
from pathlib import Path

import numpy as np
import pytest

from df2_pi.animation import (
    AnimationError,
    AnimationMeta,
    AnimationRegistry,
    Effect,
    LoadError,
    Param,
    animation,
    default_animations_dir,
    load_animation_file,
)
from df2_pi.effects import FADE
from df2_pi.pixels import PixelFrame, TileFrame, default_geometry

# ---- fixture files ------------------------------------------------------------------

SOLID = '''
"""A solid colour, brighter every frame."""
from df2_pi.animation import animation, Param
from df2_pi.pixels import TileFrame

@animation(
    name="Solid",
    description="One colour everywhere.",
    author="test",
    format="tile",
    tags=["test", "flat"],
    params={
        "level": Param(int, default=10, min=0, max=255, label="Level"),
        "mode": Param(str, default="up", choices=["up", "down"]),
        "wobble": Param(bool, default=False),
    },
    period=2.0,
    preview_hint="loop",
    energy="low",
)
def render(previous, ctx):
    frame = TileFrame.black(ctx.geometry)
    ctx.state["calls"] = ctx.state.get("calls", 0) + 1
    v = min(255, ctx.params["level"] + ctx.frame)
    frame.data[:] = (v, 0, 0)
    return frame
'''

PIXEL_DECAY = '''
from df2_pi.animation import animation
from df2_pi.pixels import PixelFrame

@animation(name="Decay", format="pixel")
def render(previous, ctx):
    frame = previous.copy().gain(0.5)
    if ctx.frame == 0:
        frame.data[:] = 200
    return frame
'''

NO_DECORATOR = '''
"""A helper module, not an animation."""
def helper():
    return 1
'''

BROKEN_SYNTAX = '''
from df2_pi.animation import animation

@animation(name="Broken"
def render(previous, ctx):
    return previous
'''

RAISES_ON_IMPORT = '''
from df2_pi.animation import animation
raise RuntimeError("boom at import")
'''

BAD_SIGNATURE = '''
from df2_pi.animation import animation

@animation(name="OneArg")
def render(previous):
    return previous
'''

TWO_ANIMATIONS = '''
from df2_pi.animation import animation

@animation(name="A")
def a(previous, ctx):
    return previous

@animation(name="B")
def b(previous, ctx):
    return previous
'''

WRONG_FORMAT = '''
from df2_pi.animation import animation
from df2_pi.pixels import PixelFrame

@animation(name="Liar", format="tile")
def render(previous, ctx):
    return PixelFrame.black(ctx.geometry)
'''

RETURNS_PREVIOUS = '''
from df2_pi.animation import animation

@animation(name="Lazy", format="tile")
def render(previous, ctx):
    return previous
'''

RAISES_IN_RENDER = '''
from df2_pi.animation import animation

@animation(name="Crash", format="tile")
def render(previous, ctx):
    raise ZeroDivisionError("oops")
'''

BAD_PARAM_DEFAULT = '''
from df2_pi.animation import animation, Param

@animation(name="OutOfRange", params={"x": Param(float, default=9.0, min=0.0, max=1.0)})
def render(previous, ctx):
    return previous
'''

EFFECT_ALL = '''
from df2_pi.animation import animation, Effect, FADE
from df2_pi.pixels import TileFrame

@animation(name="Trail", format="tile", effect=Effect(FADE, (230, 0, 0, 0)))
def render(previous, ctx):
    return TileFrame.black(ctx.geometry)
'''

EFFECT_SOME = '''
from df2_pi.animation import animation, Effect, FADE
from df2_pi.pixels import TileFrame

@animation(name="Corners", format="tile", effect={0: Effect(FADE, (200, 0, 0, 0)), 63: Effect.NONE})
def render(previous, ctx):
    return TileFrame.black(ctx.geometry)
'''

SENDS_EFFECT = '''
from df2_pi.animation import animation, Effect, FADE
from df2_pi.pixels import TileFrame

@animation(name="Sender", format="tile")
def render(previous, ctx):
    if ctx.frame == 2:
        ctx.send_effect(5, Effect(FADE, (100, 0, 0, 0)))
    if ctx.frame == 3:
        ctx.send_effect(5, Effect.NONE)
    if ctx.frame == 4:
        ctx.send_effect_all(Effect(FADE))
    if ctx.frame == 5:
        ctx.send_effect(64, Effect.NONE)
    return TileFrame.black(ctx.geometry)
'''

RANDOM_WALK = '''
from df2_pi.animation import animation
from df2_pi.pixels import PixelFrame

@animation(name="Walk", format="pixel")
def render(previous, ctx):
    frame = PixelFrame.black(ctx.geometry)
    g = ctx.geometry.edge_graph()
    path = g.walk(g.random_edge(ctx.np_rng), 6, ctx.np_rng)
    frame.flat[path.leds()] = (ctx.rng.randrange(256), 0, 0)
    return frame
'''


def write(directory: Path, name: str, source: str, mtime: float | None = None) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(source))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def anim_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animations"
    d.mkdir()
    write(d, "solid.py", SOLID)
    write(d, "decay.py", PIXEL_DECAY)
    write(d, "_helpers.py", NO_DECORATOR)
    write(d, "plain.py", NO_DECORATOR)
    write(d, "broken.py", BROKEN_SYNTAX)
    write(d, "explodes.py", RAISES_ON_IMPORT)
    write(d, "onearg.py", BAD_SIGNATURE)
    write(d, "twins.py", TWO_ANIMATIONS)
    write(d, "outofrange.py", BAD_PARAM_DEFAULT)
    return d


# ---- discovery ----------------------------------------------------------------------


def test_discovers_valid_animations_and_isolates_the_broken_ones(anim_dir):
    registry = AnimationRegistry.discover(anim_dir)
    assert set(registry.animations) == {"solid", "decay"}
    assert set(registry.errors) == {"broken", "explodes", "onearg", "twins", "outofrange"}
    assert "plain" not in registry.errors  # no decorator: skipped, not an error
    assert "_helpers" not in registry.animations and "_helpers" not in registry.errors
    assert len(registry) == 2 and "solid" in registry and "broken" not in registry
    assert {d.id for d in registry} == {"solid", "decay"}


def test_metadata_and_param_specs_survive_intact(anim_dir):
    solid = AnimationRegistry.discover(anim_dir)["solid"]
    assert solid.id == "solid" and solid.path == anim_dir / "solid.py"
    meta = solid.meta
    assert isinstance(meta, AnimationMeta)
    assert (meta.name, meta.author, meta.format) == ("Solid", "test", "tile")
    assert meta.description == "One colour everywhere."
    assert meta.tags == ("test", "flat")
    assert meta.period == 2.0
    assert meta.effect is None
    level = meta.params["level"]
    assert isinstance(level, Param)
    assert (level.type, level.default, level.min, level.max, level.label) == (int, 10, 0, 255, "Level")
    assert meta.params["mode"].choices == ("up", "down")
    assert meta.params["wobble"].default is False
    assert solid.frame_type is TileFrame
    assert solid.render.__doc__ is None and callable(solid.render)


def test_unknown_kwargs_land_in_extra(anim_dir):
    meta = AnimationRegistry.discover(anim_dir)["solid"].meta
    assert meta.extra == {"preview_hint": "loop", "energy": "low"}


def test_errors_carry_stage_message_and_traceback(anim_dir):
    errors = AnimationRegistry.discover(anim_dir).errors
    broken = errors["broken"]
    assert isinstance(broken, LoadError)
    assert broken.stage == "syntax" and broken.path == anim_dir / "broken.py"
    assert "SyntaxError" in broken.message and "line" in broken.message
    assert "broken.py" in str(broken)
    explodes = errors["explodes"]
    assert explodes.stage == "import"
    assert "boom at import" in explodes.message
    assert "RuntimeError" in explodes.traceback and "explodes.py" in explodes.traceback
    assert errors["onearg"].stage == "validate" and "(previous, ctx)" in errors["onearg"].message
    assert errors["twins"].stage == "validate" and "more than one" in errors["twins"].message
    assert errors["outofrange"].stage == "import"  # the decorator raised at import time
    assert "above the maximum" in errors["outofrange"].message


def test_getitem_on_a_failed_animation_says_why(anim_dir):
    registry = AnimationRegistry.discover(anim_dir)
    with pytest.raises(KeyError, match="failed to load"):
        registry["broken"]
    with pytest.raises(KeyError, match="no animation 'nope'"):
        registry["nope"]
    assert registry.get("nope") is None


def test_a_module_without_a_decorator_is_not_an_animation(tmp_path):
    assert load_animation_file(write(tmp_path, "plain.py", NO_DECORATOR)) is None


def test_non_identifier_filenames_are_rejected(tmp_path):
    path = write(tmp_path, "bad-name.py", SOLID)
    with pytest.raises(LoadError, match="not a valid animation id"):
        load_animation_file(path)


def test_missing_directory_loads_nothing(tmp_path):
    registry = AnimationRegistry.discover(tmp_path / "missing")
    assert not registry.animations and not registry.errors


def test_duplicate_ids_across_directories_are_reported_not_shadowed(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write(a, "solid.py", SOLID)
    write(b, "solid.py", PIXEL_DECAY)
    registry = AnimationRegistry.discover([a, b])
    assert registry["solid"].path == a / "solid.py"  # first directory wins
    assert registry.errors["solid"].stage == "validate"
    assert "duplicate" in registry.errors["solid"].message


# ---- reload -----------------------------------------------------------------------


def test_reload_picks_up_an_edit_a_new_file_and_a_removal(anim_dir):
    registry = AnimationRegistry.discover(anim_dir)
    assert registry.reload() == []  # nothing changed

    write(anim_dir, "solid.py", SOLID.replace('name="Solid"', 'name="Solid v2"'), mtime=2e9)
    assert registry.reload() == ["solid"]
    assert registry["solid"].meta.name == "Solid v2"

    write(anim_dir, "fresh.py", PIXEL_DECAY)
    assert registry.reload() == ["fresh"]
    assert registry["fresh"].meta.name == "Decay"

    (anim_dir / "fresh.py").unlink()
    assert registry.reload() == ["fresh"]
    assert "fresh" not in registry


def test_failed_reload_keeps_the_last_good_version_live(anim_dir):
    registry = AnimationRegistry.discover(anim_dir)
    good = registry["solid"]
    write(anim_dir, "solid.py", BROKEN_SYNTAX, mtime=2e9)
    assert registry.reload() == ["solid"]
    assert registry["solid"] is good  # still live
    assert registry.errors["solid"].stage == "syntax"  # but the problem is visible

    write(anim_dir, "solid.py", SOLID, mtime=2e9 + 1)
    registry.reload()
    assert registry["solid"] is not good and registry["solid"].meta.name == "Solid"
    assert "solid" not in registry.errors  # fixed: error cleared


def test_a_broken_file_that_gets_fixed_starts_loading(anim_dir):
    registry = AnimationRegistry.discover(anim_dir)
    assert "broken" not in registry
    write(anim_dir, "broken.py", SOLID, mtime=2e9)
    registry.reload()
    assert "broken" in registry and "broken" not in registry.errors


# ---- running --------------------------------------------------------------------


def test_run_context_frame_time_state_and_previous(anim_dir):
    run = AnimationRegistry.discover(anim_dir)["solid"].start(fps=30.0)
    assert run.frame == 0 and run.state == {}
    assert isinstance(run.previous, TileFrame) and run.previous.frozen
    assert not run.previous.data.any()

    out = run.render()
    assert isinstance(out.frame, TileFrame) and out.frame.frozen
    assert out.frame[0, 0].tolist() == [10, 0, 0]  # level + frame 0
    assert out.effects == {}
    assert run.frame == 1 and run.state == {"calls": 1}
    assert run.previous is out.frame

    out = run.render()
    assert out.frame[0, 0].tolist() == [11, 0, 0]
    assert run.state == {"calls": 2}


def test_t_defaults_to_frame_over_fps_and_can_be_supplied(tmp_path):
    seen = []
    write(
        tmp_path,
        "clock.py",
        '''
        from df2_pi.animation import animation
        from df2_pi.pixels import TileFrame
        SEEN = []
        @animation(name="Clock")
        def render(previous, ctx):
            SEEN.append((ctx.frame, ctx.t, ctx.dt, ctx.fps))
            return TileFrame.black(ctx.geometry)
        ''',
    )
    definition = load_animation_file(tmp_path / "clock.py")
    import sys

    seen = sys.modules["df2_animations.clock"].SEEN
    run = definition.start(fps=10.0)
    run.render()
    run.render()
    run.render(t=7.5)
    assert seen == [(0, 0.0, 0.1, 10.0), (1, 0.1, 0.1, 10.0), (2, 7.5, 0.1, 10.0)]


def test_previous_frame_is_read_only_and_evolving_a_copy_works(anim_dir):
    run = AnimationRegistry.discover(anim_dir)["decay"].start()
    first = run.render().frame
    assert (first.data == 200).all()
    second = run.render().frame
    assert (second.data == 100).all()
    assert (first.data == 200).all()  # untouched
    with pytest.raises(ValueError):
        first.data[0, 0] = 0


def test_params_are_resolved_through_their_specs(anim_dir):
    definition = AnimationRegistry.discover(anim_dir)["solid"]
    assert definition.meta.defaults() == {"level": 10, "mode": "up", "wobble": False}
    run = definition.start(params={"level": "42", "mode": "down", "wobble": "yes"})
    assert run.params == {"level": 42, "mode": "down", "wobble": True}
    assert run.render().frame[0, 0].tolist() == [42, 0, 0]
    run.set_params({"level": 100})
    assert run.render().frame[3, 3].tolist() == [101, 0, 0]

    with pytest.raises(ValueError, match="unknown parameter 'lvl'"):
        definition.start(params={"lvl": 1})
    with pytest.raises(ValueError, match="above the maximum"):
        definition.start(params={"level": 300})
    with pytest.raises(ValueError, match="not one of"):
        definition.start(params={"mode": "sideways"})
    with pytest.raises(ValueError):
        definition.start(params={"level": "lots"})


def test_param_coerce_rules():
    p = Param(float, default=1.0, min=0.0, max=2.0)
    assert p.coerce(1) == 1.0 and isinstance(p.coerce(1), float)
    with pytest.raises(TypeError):
        p.coerce(True)  # a bool is not a number here
    with pytest.raises(ValueError):
        Param(int, default=5, choices=[1, 2, 3])
    with pytest.raises(TypeError):
        Param("float", default=1.0)
    b = Param(bool, default=True)
    assert b.coerce("off") is False and b.coerce(False) is False
    with pytest.raises(TypeError):
        b.coerce(1)


def test_same_seed_reproduces_and_different_seeds_differ(tmp_path):
    write(tmp_path, "walk.py", RANDOM_WALK)
    definition = load_animation_file(tmp_path / "walk.py")
    a = [definition.start(seed=5).render().frame for _ in range(2)]
    assert a[0] == a[1]
    b = definition.start(seed=6).render().frame
    assert b != a[0]
    run = definition.start()
    again = definition.start(seed=run.seed)
    assert run.render().frame == again.render().frame


def test_wrong_frame_format_is_rejected_naming_the_file(tmp_path):
    write(tmp_path, "liar.py", WRONG_FORMAT)
    run = load_animation_file(tmp_path / "liar.py").start()
    with pytest.raises(AnimationError, match=r"liar\.py: render\(\) returned PixelFrame.*TileFrame"):
        run.render()


def test_returning_previous_is_rejected_naming_the_file(tmp_path):
    write(tmp_path, "lazy.py", RETURNS_PREVIOUS)
    run = load_animation_file(tmp_path / "lazy.py").start()
    with pytest.raises(AnimationError, match=r"lazy\.py: .*previous\.copy\(\)"):
        run.render()


def test_render_raising_is_wrapped_naming_the_file(tmp_path):
    write(tmp_path, "crash.py", RAISES_IN_RENDER)
    run = load_animation_file(tmp_path / "crash.py").start()
    with pytest.raises(AnimationError, match=r"crash\.py: render\(\) raised on frame 0") as info:
        run.render()
    assert isinstance(info.value.__cause__, ZeroDivisionError)


def test_geometry_is_honoured(tmp_path):
    from df2_pi.geometry import FloorGeometry

    write(tmp_path, "decay.py", PIXEL_DECAY)
    small = FloorGeometry(tile_grid=(2, 2), leds_per_side=3)
    run = load_animation_file(tmp_path / "decay.py").start(small)
    assert run.render().frame.data.shape == (4, 12, 3)


# ---- effects ----------------------------------------------------------------------


def test_effect_metadata_writes_every_tile_on_frame_0_only(tmp_path):
    write(tmp_path, "trail.py", EFFECT_ALL)
    run = load_animation_file(tmp_path / "trail.py").start()
    geo = default_geometry()
    first = run.render().effects
    assert set(first) == set(range(geo.tiles))
    assert all(e == Effect(FADE, (230, 0, 0, 0)) for e in first.values())
    assert bytes(first[0]) == bytes((FADE, 230, 0, 0, 0))
    for _ in range(3):
        assert run.render().effects == {}


def test_effect_metadata_mapping_form_writes_only_those_tiles(tmp_path):
    write(tmp_path, "corners.py", EFFECT_SOME)
    run = load_animation_file(tmp_path / "corners.py").start()
    first = run.render().effects
    assert first == {0: Effect(FADE, (200, 0, 0, 0)), 63: Effect.NONE}
    assert run.render().effects == {}


def test_send_effect_emits_once_and_none_is_an_id_zero_entry(tmp_path):
    write(tmp_path, "sender.py", SENDS_EFFECT)
    run = load_animation_file(tmp_path / "sender.py").start()
    assert run.render().effects == {}  # frame 0
    assert run.render().effects == {}  # frame 1
    assert run.render().effects == {5: Effect(FADE, (100, 0, 0, 0))}  # frame 2
    clear = run.render().effects  # frame 3
    assert clear == {5: Effect.NONE} and bytes(clear[5]) == b"\x00\x00\x00\x00\x00"
    everyone = run.render().effects  # frame 4
    assert len(everyone) == 64 and all(e == Effect(FADE) for e in everyone.values())
    with pytest.raises(AnimationError, match="tile must be 0..63"):
        run.render()  # frame 5 sends to tile 64


def test_effect_ids_and_params_are_validated_at_the_call_site():
    with pytest.raises(ValueError, match="bits 7:5 reserved"):
        Effect(32)
    with pytest.raises(ValueError):
        Effect(0x81)
    with pytest.raises(ValueError):
        Effect(1, (0, 0, 0, 256))
    with pytest.raises(ValueError):
        Effect(1, (0, 0, 0))
    assert Effect(31, (255, 255, 255, 255)) is not None


def test_decorator_validates_metadata_eagerly():
    with pytest.raises(ValueError, match="format must be one of"):
        animation(name="x", format="voxel")
    with pytest.raises(ValueError, match="non-empty"):
        animation(name="  ")
    with pytest.raises(TypeError, match="tags"):
        animation(name="x", tags="ambient")
    with pytest.raises(TypeError, match="must be a Param"):
        animation(name="x", params={"speed": 1.0})
    with pytest.raises(ValueError, match="identifiers"):
        animation(name="x", params={"bad name": Param(int, default=1)})
    with pytest.raises(TypeError, match="Effect"):
        animation(name="x", effect="fade")
    with pytest.raises(ValueError, match="period"):
        animation(name="x", period=0)
    with pytest.raises(TypeError, match="decorate a function"):
        animation(name="x")("not callable")


# ---- the shipped animations directory -------------------------------------------------


def test_the_reference_animation_loads_and_runs():
    directory = default_animations_dir()
    assert directory.is_dir(), directory
    registry = AnimationRegistry.discover(directory)
    assert registry.errors == {}
    assert "rainbow_sweep" in registry
    run = registry["rainbow_sweep"].start(seed=0)
    frames = [run.render().frame for _ in range(3)]
    assert all(isinstance(f, TileFrame) for f in frames)
    assert frames[0] != frames[2]  # it moves
    assert np.all(frames[0].data.max(axis=-1) == 255)  # full value, every tile lit
