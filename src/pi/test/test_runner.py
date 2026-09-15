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
