import signal
import textwrap
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from df2_pi.animation import AnimationRegistry
from df2_pi.effects import FADE, Effect
from df2_pi.encode import FrameEncoder
from df2_pi.engine import FrameClock
from df2_pi.engine.runner import IDLE, Runner, RunnerState
from df2_pi.output import FanOut, HardwareSink, NullSink
from df2_pi.pixels import PixelFrame, TileFrame
from df2_pi.playlists import PlaylistStore

FPS = 10.0  # coarse, so durations are few ticks
PERIOD = 1.0 / FPS

# Each animation paints its tile (0, 0) with a signature colour and the
# frame number in the green channel, so a probe can tell who rendered.
SOLID = '''
from df2_pi.animation import animation, Param
from df2_pi.pixels import TileFrame

@animation(name="{name}", params={{"level": Param(int, default={level}, min=0, max=255)}})
def render(previous, ctx):
    frame = TileFrame.black(ctx.geometry)
    frame.data[:] = (ctx.params["level"], min(255, ctx.frame), 0)
    return frame
'''

PIXELS = '''
from df2_pi.animation import animation
from df2_pi.pixels import PixelFrame

@animation(name="Pixels", format="pixel")
def render(previous, ctx):
    frame = PixelFrame.black(ctx.geometry)
    frame.data[:] = (0, 0, 200)
    return frame
'''

CRASHES = '''
from df2_pi.animation import animation
from df2_pi.pixels import TileFrame

@animation(name="Crashes")
def render(previous, ctx):
    raise RuntimeError("kaboom")
'''

CRASHES_LATER = '''
from df2_pi.animation import animation
from df2_pi.pixels import TileFrame

@animation(name="Crashes later")
def render(previous, ctx):
    if ctx.frame >= 2:
        raise RuntimeError("kaboom")
    frame = TileFrame.black(ctx.geometry)
    frame.data[:] = (7, ctx.frame, 0)
    return frame
'''

WITH_EFFECT = '''
from df2_pi.animation import animation, Effect, FADE
from df2_pi.pixels import TileFrame

@animation(name="Trail", effect=Effect(FADE, (230, 0, 0, 0)))
def render(previous, ctx):
    if ctx.frame == 1:
        ctx.send_effect(5, Effect(FADE, (100, 0, 0, 0)))
    frame = TileFrame.black(ctx.geometry)
    frame.data[:] = (99, ctx.frame, 0)
    return frame
'''


CLOCKED = '''
from df2_pi.animation import animation
from df2_pi.pixels import TileFrame

@animation(name="Clocked")
def render(previous, ctx):
    frame = TileFrame.black(ctx.geometry)
    frame.data[:] = (min(255, round(ctx.t * 10)), round(ctx.dt * 100), 50)
    return frame
'''


CONTROLLED = '''
from df2_pi.animation import animation, Param
from df2_pi.pixels import TileFrame

@animation(name="Controlled", params={
    "speed": Param(float, default=1.0, min=0.1, max=10.0, curve="log"),
    "tail": Param(int, default=10, min=0, max=100, macro=1),
})
def render(previous, ctx):
    frame = TileFrame.black(ctx.geometry)
    frame.data[:] = (40, ctx.params["tail"], 0)
    return frame
'''


class FakeTime:
    def __init__(self) -> None:
        self.t = 100.0

    def now(self) -> float:
        self.t += 1e-6
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class Probe:
    """A synchronous sink recording what the runner submits, stopping the
    runner after `stop_after` frames (or when `until(runner)` is true)."""

    name = "probe"

    def __init__(self, runner_ref: list, stop_after: int | None = None, until=None) -> None:
        self.frames: list = []
        self.effects: list = []
        self.states: list[RunnerState] = []
        self._runner_ref = runner_ref
        self.stop_after = stop_after
        self.until = until
        self.latches = 0

    def submit(self, frame, info, effects=None):
        self.frames.append((info, frame))
        self.effects.append(dict(effects or {}))
        runner = self._runner_ref[0]
        self.states.append(runner.state)
        if self.stop_after is not None and len(self.frames) >= self.stop_after:
            runner.stop()
        if self.until is not None and self.until(runner, len(self.frames)):
            runner.stop()

    def latch(self):
        self.latches += 1

    def close(self):
        pass

    def levels(self) -> list[int]:
        """Signature colour of tile (0, 0) per frame: who rendered it."""
        return [int(f.grid[1, 0, 0]) if isinstance(f, PixelFrame) else int(f.data[0, 0, 0]) for _, f in self.frames]


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    d = tmp_path / "animations"
    d.mkdir()
    for name, level in (("a", 10), ("b", 20), ("c", 30)):
        (d / f"{name}.py").write_text(textwrap.dedent(SOLID.format(name=name.upper(), level=level)))
    (d / "pixels.py").write_text(textwrap.dedent(PIXELS))
    (d / "crashes.py").write_text(textwrap.dedent(CRASHES))
    (d / "crashes_later.py").write_text(textwrap.dedent(CRASHES_LATER))
    (d / "trail.py").write_text(textwrap.dedent(WITH_EFFECT))
    (d / "clocked.py").write_text(textwrap.dedent(CLOCKED))
    (d / "controlled.py").write_text(textwrap.dedent(CONTROLLED))
    (d / "broken.py").write_text("def render(:\n")
    return AnimationRegistry.discover(d)


@pytest.fixture
def store(registry) -> PlaylistStore:
    s = PlaylistStore(":memory:", registry=registry)
    yield s
    s.close()


def make_runner(registry, store, stop_after=None, until=None, **kw):
    fake = FakeTime()
    clock = FrameClock(fps=FPS, spin_margin=0.0, now=fake.now, sleep=fake.sleep)
    ref: list = []
    probe = Probe(ref, stop_after, until)
    fanout = FanOut([probe])
    runner = Runner(registry, fanout, store=store, clock=clock, **kw)
    ref.append(runner)
    return runner, probe, fake


def playlist(store, *entries, **kw):
    pl = store.create_playlist(kw.pop("name", "P"), **kw)
    for animation_id, duration in entries:
        store.add_entry(pl, animation_id, duration_s=duration)
    return store.resolve(pl.id)


# ---- advancing ----------------------------------------------------------------------------


def test_the_state_names_the_playing_entry_by_id(registry, store):
    """entry_index is a position in the runner's loaded copy; entry_id is
    the entry itself, so a page can find it after the playlist is
    reordered in the database."""
    runner, probe, _ = make_runner(registry, store, stop_after=6)
    resolved = playlist(store, ("a", 3 * PERIOD), ("b", 5 * PERIOD), loop=False)
    first, second = (e.entry.id for e in resolved.entries)
    store.move_entry(second, 0)  # reordered after the runner resolved it: it plays its own copy
    runner.load_playlist(resolved)
    runner.run()
    states = probe.states
    assert (states[0].entry_id, states[0].entry_index) == (first, 0)
    assert (states[3].entry_id, states[3].entry_index) == (second, 1)


def test_the_idle_animation_and_a_one_off_have_no_entry(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=3)
    runner.play_animation("a")
    runner.run()
    assert all(s.entry_id is None and s.one_off for s in probe.states)


def test_advances_after_exactly_duration_worth_of_ticks(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=12)
    runner.load_playlist(playlist(store, ("a", 3 * PERIOD), ("b", 5 * PERIOD), loop=False))
    runner.run()
    assert probe.levels() == [10] * 3 + [20] * 5 + [20] * 4  # ends holding B's last frame
    assert probe.latches == len(probe.frames) + 1  # every frame, plus the shutdown latch
    states = probe.states
    assert states[0].animation == ("a", "A") and states[0].entry_index == 0
    assert states[3].animation == ("b", "B") and states[3].entry_index == 1
    assert states[3].elapsed_s == pytest.approx(0.0) and states[3].remaining_s == pytest.approx(5 * PERIOD)
    assert states[8].playing is False  # ended
    assert [s.frame for s in states] == list(range(12))


def test_loop_wraps(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=9)
    runner.load_playlist(playlist(store, ("a", 2 * PERIOD), ("b", 2 * PERIOD), loop=True))
    runner.run()
    assert probe.levels() == [10, 10, 20, 20, 10, 10, 20, 20, 10]
    assert all(s.playing for s in probe.states)


def test_each_entry_starts_from_its_own_frame_zero(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=6)
    runner.load_playlist(playlist(store, ("a", 3 * PERIOD), ("b", 3 * PERIOD)))
    runner.run()
    frames = [int(f.data[0, 0, 1]) for _, f in probe.frames]  # ctx.frame in green
    assert frames == [0, 1, 2, 0, 1, 2]


def test_shuffle_visits_every_entry_once_before_repeating(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=9, seed=3)
    runner.load_playlist(playlist(store, ("a", PERIOD), ("b", PERIOD), ("c", PERIOD), loop=True, shuffle=True))
    runner.run()
    levels = probe.levels()
    assert sorted(levels[0:3]) == [10, 20, 30]
    assert sorted(levels[3:6]) == [10, 20, 30]
    assert sorted(levels[6:9]) == [10, 20, 30]
    # and it really is shuffled at some point across a few seeds
    orders = set()
    for seed in range(6):
        runner, probe, _ = make_runner(registry, store, stop_after=3, seed=seed)
        runner.load_playlist(playlist(store, ("a", PERIOD), ("b", PERIOD), ("c", PERIOD), name=f"S{seed}", shuffle=True))
        runner.run()
        orders.add(tuple(probe.levels()))
    assert len(orders) > 1


def test_unresolved_and_disabled_entries_are_skipped_without_error(registry, store):
    pl = store.create_playlist("P")
    store.add_entry(pl, "a", duration_s=PERIOD)
    store.add_entry(pl, "vanished", duration_s=PERIOD)
    store.add_entry(pl, "broken", duration_s=PERIOD)
    store.add_entry(pl, "b", duration_s=PERIOD, enabled=False)
    store.add_entry(pl, "c", duration_s=PERIOD)
    runner, probe, _ = make_runner(registry, store, stop_after=4)
    runner.load_playlist(store.resolve(pl.id))
    runner.run()
    assert probe.levels() == [10, 30, 10, 30]
    assert probe.states[1].entry_index == 4
    assert probe.states[0].entry_count == 5
    assert "broken" in probe.states[0].load_errors


def test_nothing_loaded_plays_the_idle_animation(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=3)
    runner.run()
    assert all(s.animation == ("_idle", "Idle") for s in probe.states)
    assert all(f.data[..., 2].min() > 0 for _, f in probe.frames)  # dim blue, never black
    assert probe.states[0].playing is False and probe.states[0].playlist is None


def test_a_playlist_with_nothing_playable_falls_back_to_idle(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=2)
    runner.load_playlist(playlist(store, ("vanished", PERIOD)))
    runner.run()
    assert probe.states[-1].animation[0] == "_idle"
    assert probe.states[-1].playlist == (1, "P")


# ---- transitions -----------------------------------------------------------------------------


def test_cut_transition_is_clean_whatever_the_animation_is_doing(registry, store):
    # A cut: the last frame of A and the first of B are both complete
    # frames from their own animation; nothing is blended or held over.
    runner, probe, _ = make_runner(registry, store, stop_after=4)
    runner.load_playlist(playlist(store, ("a", 2 * PERIOD), ("b", 2 * PERIOD)))
    runner.run()
    assert probe.levels() == [10, 10, 20, 20]
    for _, f in probe.frames:
        assert len(set(f.data[..., 0].ravel().tolist())) == 1  # uniform: one animation's output


def test_crossfade_blends_monotonically_from_outgoing_to_incoming(registry, store):
    fade = 4 * PERIOD
    runner, probe, _ = make_runner(registry, store, stop_after=12)
    runner.load_playlist(playlist(store, ("a", 8 * PERIOD), ("b", 8 * PERIOD), crossfade_s=fade, loop=False))
    runner.run()
    levels = probe.levels()
    assert levels[:4] == [10] * 4
    ramp = levels[4:9]  # the overlap: A (10) -> B (20)
    assert ramp[0] == 10 and ramp[-1] == 20
    assert all(a <= b for a, b in zip(ramp, ramp[1:]))
    assert any(10 < v < 20 for v in ramp)
    assert levels[9:] == [20] * 3
    assert probe.states[5].animation == ("b", "B")  # the incoming is "current" during the fade
    assert probe.states[5].entry_index == 1


def test_crossfade_between_tile_and_pixel_formats(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=8)
    runner.load_playlist(playlist(store, ("a", 4 * PERIOD), ("pixels", 4 * PERIOD), crossfade_s=2 * PERIOD, loop=False))
    runner.run()
    assert all(isinstance(f, PixelFrame) for _, f in probe.frames[2:])
    reds = probe.levels()  # A is (10, n, 0), pixels is (0, 0, 200)
    assert reds[0] == 10 and reds[-1] == 0
    blues = [int(f.grid[1, 0, 2]) for _, f in probe.frames[2:]]
    assert blues[0] < blues[-1] == 200 and all(a <= b for a, b in zip(blues, blues[1:]))


# ---- effects -----------------------------------------------------------------------------------


def test_effects_travel_with_their_frame_and_are_cleared_on_change(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=5)
    runner.load_playlist(playlist(store, ("trail", 3 * PERIOD), ("a", 3 * PERIOD)))
    runner.run()
    assert set(probe.effects[0]) == set(range(64))  # effect= metadata on frame 0
    assert probe.effects[0][0] == Effect(FADE, (230, 0, 0, 0))
    assert probe.effects[1] == {5: Effect(FADE, (100, 0, 0, 0))}
    assert probe.effects[2] == {}
    # the change to A clears everything trail set, with A's first frame
    assert probe.effects[3] == {tile: Effect.NONE for tile in range(64)}
    assert probe.levels()[3] == 10
    assert probe.effects[4] == {}


# ---- failure isolation ----------------------------------------------------------------------


def test_a_raising_animation_is_skipped_the_previous_frame_held_and_three_strikes_disable(registry, store, caplog):
    runner, probe, _ = make_runner(registry, store, stop_after=12)
    runner.load_playlist(playlist(store, ("a", 2 * PERIOD), ("crashes", 5 * PERIOD), loop=True))
    with caplog.at_level("ERROR"):
        runner.run()
    levels = probe.levels()
    # A A [crash: hold A, advance] A A [crash: hold] A A [crash: hold -> disabled] A A A A ...
    assert levels[:2] == [10, 10]
    assert levels[2] == 10  # held, not black, and still latched
    assert probe.latches == len(probe.frames) + 1
    assert levels[3:5] == [10, 10]
    assert levels[5] == 10
    assert levels[8] == 10
    assert all(v == 10 for v in levels[9:])  # crashes is disabled: only A plays
    disabled_states = [s for s in probe.states if s.disabled_entries]
    assert disabled_states and disabled_states[0].disabled_entries == (2,)
    assert "kaboom" in caplog.text and "crashes" in caplog.text
    assert "disabled after 3" in caplog.text
    log = store.play_log()
    assert [e.outcome for e in log if e.animation_id == "crashes"][:3] == ["error"] * 3


def test_a_success_resets_the_strike_count(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=20)
    runner.load_playlist(playlist(store, ("crashes_later", 5 * PERIOD), ("a", PERIOD), loop=True))
    runner.run()
    assert not any(s.disabled_entries for s in probe.states)
    assert 7 in probe.levels() and 10 in probe.levels()


# ---- control API --------------------------------------------------------------------------------


def test_pause_holds_the_frame_and_stops_render_and_advancing(registry, store):
    calls = []

    def until(runner, n):
        if n == 2:
            runner.pause()
        if n == 6:
            runner.resume()
        return n >= 10

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 3 * PERIOD), ("b", 3 * PERIOD)))
    runner.run()
    greens = [int(f.data[0, 0, 1]) for _, f in probe.frames]  # ctx.frame: only moves when rendering
    levels = probe.levels()
    assert levels[:2] == [10, 10]
    assert greens[:2] == [0, 1]
    assert [(l, g) for l, g in zip(levels[2:6], greens[2:6])] == [(10, 1)] * 4  # held: same frame, no render
    assert all(s.paused for s in probe.states[2:6])
    assert [(l, g) for l, g in zip(levels[6:8], greens[6:8])] == [(10, 2), (20, 0)]  # resumes where it was
    assert probe.latches == len(probe.frames) + 1  # every tick still latched
    assert probe.frames[3][1] is probe.frames[2][1]  # literally the same frame object


def test_next_from_another_thread_takes_effect_on_the_following_tick(registry, store):
    gate = threading.Event()
    done = threading.Event()

    def until(runner, n):
        if n == 3:
            gate.set()
            done.wait(1.0)  # another thread queues next() while this frame is in flight
        return n >= 6

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 100), ("b", 100), ("c", 100)))

    def other():
        gate.wait(1.0)
        runner.next()
        done.set()

    threading.Thread(target=other).start()
    runner.run()
    assert probe.levels() == [10, 10, 10, 20, 20, 20]  # frame 3 was mid-flight: unaffected


def test_previous_goto_and_restart(registry, store):
    def until(runner, n):
        if n == 2:
            runner.goto(2)
        if n == 4:
            runner.previous()
        if n == 6:
            runner.restart()
        return n >= 8

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 100), ("b", 100), ("c", 100)))
    runner.run()
    greens = [int(f.data[0, 0, 1]) for _, f in probe.frames]
    assert probe.levels() == [10, 10, 30, 30, 20, 20, 20, 20]
    assert greens == [0, 1, 0, 1, 0, 1, 0, 1]  # restart at frame 6 went back to B's frame 0
    assert [e.outcome for e in store.play_log()][-2:] == ["skipped", "skipped"]


def test_play_after_the_end_restarts_from_the_top(registry, store):
    def until(runner, n):
        if n == 5:
            runner.play()
        return n >= 8

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", PERIOD), ("b", PERIOD), loop=False))
    runner.run()
    assert probe.levels() == [10, 20, 20, 20, 20, 10, 20, 20]


def test_load_playlist_swaps_without_stopping(registry, store):
    second = playlist(store, ("c", 100), name="Second")

    def until(runner, n):
        if n == 2:
            runner.load_playlist(second)
        return n >= 4

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 100)))
    runner.run()
    assert probe.levels() == [10, 10, 30, 30]
    assert probe.states[-1].playlist == (second.playlist.id, "Second")
    assert probe.latches == len(probe.frames) + 1


def test_load_playlist_by_id_needs_the_store(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=2)
    pl = playlist(store, ("b", 100))
    runner.load_playlist("P")
    runner.run()
    assert probe.levels() == [20, 20]
    bare = Runner(registry, FanOut([NullSink()]))
    with pytest.raises(ValueError):
        bare.load_playlist(pl.playlist.id)


def test_play_animation_one_off_then_back_to_the_playlist(registry, store):
    def until(runner, n):
        if n == 2:
            runner.play_animation("c", {"level": 77}, hold=2 * PERIOD)
        return n >= 7

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 100)))
    runner.run()
    assert probe.levels() == [10, 10, 77, 77, 10, 10, 10]
    assert probe.states[2].one_off and probe.states[2].animation == ("c", "C")
    assert probe.states[2].params == {"level": 77}
    assert not probe.states[4].one_off
    with pytest.raises(KeyError):
        runner.play_animation("nope")
    with pytest.raises(ValueError):
        runner.play_animation("c", {"level": 999})


def test_set_params_live_tunes_the_running_animation(registry, store):
    def until(runner, n):
        if n == 2:
            runner.set_params(level=200)
        return n >= 4

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 100)))
    runner.run()
    assert probe.levels() == [10, 10, 200, 200]
    assert probe.states[-1].params == {"level": 200}


def test_set_brightness_reaches_the_encoder_within_one_frame(registry, store):
    floor = MagicMock()
    hw = HardwareSink(floor)
    seen = []

    def until(runner, n):
        seen.append(hw.encoder.brightness)
        if n == 2:
            runner.set_brightness(40)
        return n >= 4

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.fanout.attach(hw)
    runner.load_playlist(playlist(store, ("a", 100)))
    runner.run()
    assert seen == [255, 255, 40, 40]
    assert probe.states[-1].brightness == 40
    with pytest.raises(ValueError):
        runner.set_brightness(300)


def test_blackout_keeps_playback_going_underneath(registry, store):
    floor = MagicMock()
    hw = HardwareSink(floor, FrameEncoder(gamma=1.0))  # level 10 must not encode to black

    def until(runner, n):
        if n == 2:
            runner.blackout()
        if n == 4:
            runner.unblackout()
        return n >= 6

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.fanout.attach(hw)
    runner.load_playlist(playlist(store, ("a", 100)))
    runner.run()
    greens = [int(f.data[0, 0, 1]) for _, f in probe.frames]
    assert greens == [0, 1, 2, 3, 4, 5]  # observers keep seeing live content
    assert [s.blacked_out for s in probe.states] == [False, False, True, True, False, False]
    # BLACKOUT once on blackout(), once at shutdown
    assert floor.blackout.call_count == 2
    black = hw.encoder.blackout_payload()
    sent = [call.args[0] for call in floor.send_rows.call_args_list]
    assert [p[0] == black for p in sent] == [False, False, True, True, False, False]


# ---- shutdown ------------------------------------------------------------------------------------


def test_stop_blacks_out_and_latches_before_closing(registry, store):
    floor = MagicMock()
    hw = HardwareSink(floor)
    runner, probe, _ = make_runner(registry, store, stop_after=3)
    runner.fanout.attach(hw)
    runner.load_playlist(playlist(store, ("a", 100)))
    runner.run()
    calls = [c[0] for c in floor.method_calls]
    assert calls[-2:] == ["blackout", "latch"]
    assert runner.fanout.sinks == ()  # closed
    assert not runner.clock.running


def test_a_simulated_sigterm_stops_the_same_way(registry, store):
    floor = MagicMock()
    hw = HardwareSink(floor)

    def until(runner, n):
        if n == 2:
            runner._on_signal(signal.SIGTERM, None)  # what the handler does
        return False

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.fanout.attach(hw)
    runner.load_playlist(playlist(store, ("a", 100)))
    runner.run()
    assert len(probe.frames) == 2  # the frame in flight completed, then stopped
    calls = [c[0] for c in floor.method_calls]
    assert calls[-2:] == ["blackout", "latch"]


def test_run_on_a_thread_and_state_from_outside(registry, store):
    runner, probe, fake = make_runner(registry, store, stop_after=50)
    runner.load_playlist(playlist(store, ("a", 100)))
    thread = runner.start()
    runner.join(5.0)
    assert not thread.is_alive()
    state = runner.state
    assert isinstance(state, RunnerState)
    assert state.frame == 49 and state.animation == ("a", "A")
    assert state.timing.frames == 49 and state.sinks["probe"]["attached"]


def test_persistent_dropping_raises_a_warning(registry, store):
    def until(runner, n):
        if 3 <= n <= 6:
            fake.t += 2.5 * PERIOD  # this frame overruns
        return n >= 10

    runner, probe, fake = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 100)))
    runner.run()
    assert runner.clock.dropped >= 3
    assert any("dropping frames" in w for w in probe.states[-1].warnings)
    assert not any("dropping" in w for w in probe.states[0].warnings)


# ---- rotation ---------------------------------------------------------------------------------


def test_every_sink_gets_the_rotated_frame_and_the_state_reports_it(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=2, rotation=90)
    runner.play_animation("pixels")
    runner.run()
    geo = runner.geometry
    rendered = PixelFrame.black(geo)
    rendered.data[:] = (0, 0, 200)
    frame = probe.frames[0][1]
    assert frame == rendered.rotated(1)
    assert probe.states[0].rotation == 90


def test_a_tile_frame_stays_a_tile_frame_when_rotated(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=2, rotation=270)
    runner.play_animation("a")
    runner.run()
    assert all(type(f) is TileFrame for _, f in probe.frames)


def test_effect_writes_land_on_the_rotated_tile(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=3, rotation=90)
    runner.play_animation("trail")
    runner.run()
    dest = runner.geometry.rotation(1).tile_dest
    assert set(probe.effects[0]) == set(range(64))
    assert probe.effects[1] == {int(dest[5]): Effect(FADE, (100, 0, 0, 0))}


def test_changing_the_rotation_moves_the_effect_registers_already_set(registry, store):
    def turn_after_two(runner, n):
        if n == 2:
            runner.set_rotation(90)
        return n >= 4

    runner, probe, _ = make_runner(registry, store, until=turn_after_two)
    runner.play_animation("trail")  # frame 0: FADE 230 everywhere; frame 1: FADE 100 on tile 5
    runner.run()
    dest = int(runner.geometry.rotation(1).tile_dest[5])
    assert probe.effects[1] == {5: Effect(FADE, (100, 0, 0, 0))}
    # Every tile still has an effect, so none is cleared: tile 5's old spot
    # takes the uniform value back, and its new spot takes tile 5's.
    assert probe.effects[2] == {5: Effect(FADE, (230, 0, 0, 0)), dest: Effect(FADE, (100, 0, 0, 0))}
    assert probe.effects[3] == {}
    assert [s.rotation for s in probe.states] == [0, 0, 90, 90]


def test_changing_the_rotation_clears_the_tiles_an_effect_leaves(registry, store):
    def turn_after_one(runner, n):
        if n == 1:
            runner.set_rotation(180)
        return n >= 3

    runner, probe, _ = make_runner(registry, store, until=turn_after_one)
    runner.play_animation("a")
    runner._registers = {5: Effect(FADE, (100, 0, 0, 0))}  # as if written earlier
    runner.run()
    dest = int(runner.geometry.rotation(2).tile_dest[5])
    assert probe.effects[1] == {5: Effect.NONE, dest: Effect(FADE, (100, 0, 0, 0))}


def test_rotation_must_be_a_right_angle(registry, store):
    with pytest.raises(ValueError):
        make_runner(registry, store, rotation=45)
    runner, _, _ = make_runner(registry, store)
    with pytest.raises(ValueError):
        runner.set_rotation(-90)


# ---- show controls -------------------------------------------------------------------------------


def clock_readings(probe) -> list[tuple[int, int]]:
    """(ctx.t in tenths, ctx.dt in hundredths) per frame, from Clocked."""
    return [(int(f.data[0, 0, 0]), int(f.data[0, 0, 1])) for _, f in probe.frames]


def test_speed_scales_the_animations_clock_without_a_jump(registry, store):
    def faster_after_three(runner, n):
        if n == 3:
            runner.set_speed(2.0)
        return n >= 6

    runner, probe, _ = make_runner(registry, store, until=faster_after_three)
    runner.play_animation("clocked")
    runner.run()
    # 10 fps: t steps 0.1 a frame, then 0.2 from the change on; dt follows.
    assert clock_readings(probe) == [(0, 10), (1, 10), (2, 10), (4, 20), (6, 20), (8, 20)]
    assert probe.states[-1].show.speed == 2.0


def test_hold_stops_the_countdown_while_the_animation_plays_on(registry, store):
    def until(runner, n):
        if n == 1:
            runner.hold()
        if n == 8:
            runner.hold(False)
        return n >= 11

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 3 * PERIOD), ("b", 3 * PERIOD)))
    runner.run()
    greens = [int(f.data[0, 0, 1]) for _, f in probe.frames]
    assert probe.levels()[:10] == [10] * 9 + [20]  # A's 3 periods: 1 before the hold, 2 after it
    assert greens[:9] == list(range(9))  # A rendered every frame while held
    held = probe.states[1:8]
    assert all(s.timer_held for s in held) and not probe.states[8].timer_held
    assert [s.remaining_s for s in held] == pytest.approx([2 * PERIOD] * 7)


def test_a_hold_carries_over_a_skip_and_the_next_entry_waits_whole(registry, store):
    def until(runner, n):
        if n == 1:
            runner.hold()
        if n == 3:
            runner.next()
        return n >= 8

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 3 * PERIOD), ("b", 3 * PERIOD)))
    runner.run()
    assert probe.levels()[3:] == [20] * 5
    assert [s.remaining_s for s in probe.states[3:]] == pytest.approx([3 * PERIOD] * 5)


def test_a_held_countdown_stands_still_through_a_pause(registry, store):
    def until(runner, n):
        if n == 1:
            runner.hold()
        if n == 3:
            runner.pause()
        if n == 6:
            runner.resume()
        if n == 8:
            runner.hold(False)
        return n >= 11

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 3 * PERIOD), ("b", 3 * PERIOD)))
    runner.run()
    assert [s.remaining_s for s in probe.states[1:8]] == pytest.approx([2 * PERIOD] * 7)
    assert probe.levels()[:10] == [10] * 9 + [20]


def test_a_paused_countdown_reads_as_it_stood(registry, store):
    def until(runner, n):
        if n == 2:
            runner.pause()
        if n == 6:
            runner.resume()
        return n >= 8

    runner, probe, _ = make_runner(registry, store, until=until)
    runner.load_playlist(playlist(store, ("a", 5 * PERIOD)))
    runner.run()
    assert [s.remaining_s for s in probe.states[:8]] == pytest.approx([5 * PERIOD, 4 * PERIOD] + [3 * PERIOD] * 5 + [2 * PERIOD])


def test_speed_zero_stops_the_clock_and_a_pause_does_not_advance_it(registry, store):
    def script(runner, n):
        if n == 2:
            runner.set_speed(0.0)
        if n == 4:
            runner.set_speed(1.0)
            runner.pause()
        if n == 6:
            runner.resume()
        return n >= 8

    runner, probe, _ = make_runner(registry, store, until=script)
    runner.play_animation("clocked")
    runner.run()
    t = [r[0] for r in clock_readings(probe)]
    assert t[:4] == [0, 1, 1, 1]  # stopped at speed 0
    assert t[6:] == [2, 3]  # resumed from where it held, not ahead by the pause


def test_entry_durations_stay_on_the_wall_clock(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=6)
    runner.set_speed(4.0)
    runner.load_playlist(playlist(store, ("a", 3 * PERIOD), ("b", 3 * PERIOD), loop=False))
    runner.run()
    assert probe.levels() == [10, 10, 10, 20, 20, 20]


def test_show_controls_act_on_the_output_and_are_reported(registry, store):
    def script(runner, n):
        if n == 1:
            runner.set_tint(255, 0, 0, 1.0)
            runner.set_strobe(5.0)  # 10 fps: lit every other frame
        return n >= 5

    runner, probe, _ = make_runner(registry, store, until=script)
    runner.play_animation("a")  # (10, frame, 0) everywhere
    runner.run()
    lit = [bool(f.data.any()) for _, f in probe.frames]
    assert lit == [True, True, False, True, False]
    r, g, b = probe.frames[1][1].data[0, 0]
    assert r > 0 and g == 0 and b == 0  # colourised red
    show = probe.states[-1].show
    assert (show.tint, show.tint_amount, show.strobe_hz) == ((255, 0, 0), 1.0, 5.0)


def test_a_bump_starts_at_the_frame_it_is_applied_on(registry, store):
    def script(runner, n):
        if n == 2:
            runner.bump(1.0, 2 * PERIOD)
        return n >= 5

    runner, probe, _ = make_runner(registry, store, until=script)
    runner.play_animation("a")
    runner.run()
    reds = [int(f.data[0, 0, 0]) for _, f in probe.frames]
    assert reds[2] == 255 and 10 < reds[3] < 255 and reds[4] == 10


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.set_speed(4.5),
        lambda r: r.set_speed(float("nan")),
        lambda r: r.bump(1.5),
        lambda r: r.bump(1.0, 0.0),
        lambda r: r.set_strobe(-1),
        lambda r: r.set_strobe_max(20),
        lambda r: r.set_tint(256, 0, 0, 0.5),
        lambda r: r.set_tint(0, 0, 0, 2.0),
        lambda r: r.set_saturation(3.0),
        lambda r: r.set_hue_shift(float("inf")),
        lambda r: r.set_speed(True),
    ],
)
def test_bad_show_values_are_refused_on_the_callers_thread(registry, store, call):
    runner, _, _ = make_runner(registry, store)
    with pytest.raises(ValueError):
        call(runner)


# ---- external controls ---------------------------------------------------------------------------


def test_a_control_reaches_the_param_with_that_role_or_macro(registry, store):
    def controls(runner, n):
        if n == 1:
            runner.set_control("speed", 0.5)
            runner.set_control("macro1", 0.25)
        return n >= 3

    runner, probe, _ = make_runner(registry, store, until=controls)
    runner.play_animation("controlled")
    runner.run()
    assert probe.states[0].params == {"speed": 1.0, "tail": 10}
    assert probe.states[1].params["speed"] == pytest.approx(1.0)  # log: the middle of 0.1..10
    assert probe.states[1].params["tail"] == 25
    assert int(probe.frames[1][1].data[0, 0, 1]) == 25


def test_a_control_the_animation_does_not_have_is_ignored(registry, store):
    def controls(runner, n):
        if n == 1:
            runner.set_control("hue", 0.9)
            runner.set_control("macro3", 0.9)
        return n >= 3

    runner, probe, _ = make_runner(registry, store, until=controls)
    runner.play_animation("controlled")
    runner.run()
    assert probe.states[-1].params == {"speed": 1.0, "tail": 10}


def test_a_bad_control_target_or_value_is_refused_on_the_callers_thread(registry, store):
    runner, _, _ = make_runner(registry, store)
    for target, value in (("tempo", 0.5), ("macro9", 0.5), ("speed", float("nan"))):
        with pytest.raises(ValueError):
            runner.set_control(target, value)


# ---- layers -------------------------------------------------------------------------------------


def reds(probe) -> list[int]:
    return [int(f.data[0, 0, 0]) for _, f in probe.frames]


def test_a_layer_is_composited_over_what_plays_and_reported(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=3)
    runner.play_animation("a")  # red 10
    runner.set_layer("b", mode="mix")  # red 20
    runner.run()
    assert reds(probe) == [20, 20, 20]
    layer = probe.states[-1].layer
    assert (layer.animation, layer.mode, layer.amount, layer.params) == (("b", "B"), "mix", 1.0, {"level": 20})


def test_an_add_layer_does_not_pile_up_frame_after_frame(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=5)
    runner.play_animation("a")
    runner.set_layer("a", mode="add")
    runner.run()
    assert len(set(reds(probe))) == 1 and reds(probe)[0] > 10


def test_a_layer_keeps_its_own_clock_through_a_base_transition(registry, store):
    runner, probe, _ = make_runner(registry, store, stop_after=6)
    runner.load_playlist(playlist(store, ("a", 3 * PERIOD), ("b", 3 * PERIOD)))
    runner.set_layer("clocked", mode="mix")
    runner.run()
    assert [t for t, _ in clock_readings(probe)] == [0, 1, 2, 3, 4, 5]  # never restarted


def test_blend_changes_do_not_restart_the_layer_and_clear_removes_it(registry, store):
    def script(runner, n):
        if n == 2:
            runner.set_layer_blend(amount=0.0)
        if n == 4:
            runner.clear_layer()
        return n >= 5

    runner, probe, _ = make_runner(registry, store, until=script)
    runner.play_animation("a")
    runner.set_layer("b", mode="mix")
    runner.run()
    assert reds(probe) == [20, 20, 10, 10, 10]
    assert probe.states[3].layer.amount == 0.0 and probe.states[4].layer is None


def test_a_layers_effects_are_dropped_and_logged_once(registry, store, caplog):
    runner, probe, _ = make_runner(registry, store, stop_after=3)
    runner.play_animation("a")
    runner.set_layer("trail")
    with caplog.at_level("WARNING"):
        runner.run()
    assert probe.effects == [{}, {}, {}]
    assert sum("writes tile effects" in r.message for r in caplog.records) == 1


def test_a_failing_layer_is_removed_and_the_base_plays_on(registry, store, caplog):
    runner, probe, _ = make_runner(registry, store, stop_after=3)
    runner.play_animation("a")
    runner.set_layer("crashes")
    with caplog.at_level("ERROR"):
        runner.run()
    assert reds(probe) == [10, 10, 10]
    assert probe.states[-1].layer is None
    assert any("layer crashes failed" in r.message for r in caplog.records)


def test_a_pause_holds_the_layered_picture(registry, store):
    def script(runner, n):
        if n == 1:
            runner.pause()
        return n >= 4

    runner, probe, _ = make_runner(registry, store, until=script)
    runner.play_animation("a")
    runner.set_layer("b", mode="mix")
    runner.run()
    assert reds(probe) == [20, 20, 20, 20]


def test_bad_layers_are_refused_on_the_callers_thread(registry, store):
    runner, _, _ = make_runner(registry, store)
    with pytest.raises(KeyError):
        runner.set_layer("nope")
    for kwargs in (dict(mode="screen"), dict(amount=2.0), dict(params={"level": 999})):
        with pytest.raises(ValueError):
            runner.set_layer("a", **kwargs)
    with pytest.raises(ValueError):
        runner.set_layer_blend(amount=-0.5)


def test_end_one_off_and_clear_layer_only_undo_the_named_animation(registry, store):
    def script(runner, n):
        if n == 1:
            runner.end_one_off("b")  # the one-off is a: untouched
            runner.clear_layer("a")  # the layer is c: untouched
        if n == 2:
            runner.end_one_off("a")
            runner.clear_layer("c")
        return n >= 3

    runner, probe, _ = make_runner(registry, store, until=script)
    runner.load_playlist(playlist(store, ("b", 100)))
    runner.play_animation("a")
    runner.set_layer("c", mode="max")
    runner.run()
    states = probe.states
    assert (states[1].animation[0], states[1].layer.animation[0]) == ("a", "c")
    assert (states[2].animation[0], states[2].layer) == ("b", None)
