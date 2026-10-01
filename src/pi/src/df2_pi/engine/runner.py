"""`Runner`: walks a playlist, renders, hands frames to the fan-out.

    FrameClock --tick--> Runner --render--> animation --frame--> FanOut --> sinks
                           ^
                  control queue (web / MIDI / CLI)

    runner = Runner(registry, fanout, store=store)
    runner.load_playlist(store.startup_playlist())     # or a ResolvedPlaylist
    runner.run()                                       # the render loop; blocks

Per tick: drain the control queue, advance if the current entry's
duration has elapsed, render, validate, submit to the fan-out, refresh
the state snapshot. Control commands are queued from any thread and
applied ONLY at frame boundaries, so nothing mutates runner state
mid-render and the hot path needs no locks.

Advancing. An entry plays for its explicit `duration_s`; animations
declare none and are stopped by the runner. `loop` wraps; without it
playback ends holding the final frame lit. `shuffle` walks a shuffled
cycle, so nothing repeats until everything has played. TRANSITIONS ARE
THE RUNNER'S: an animation is never told the end is coming, and the cut
or crossfade is applied to the output frame, so the hand-off looks the
same whatever the animation is doing at that instant. With
`crossfade_s > 0` the incoming entry starts early and both are rendered
for the overlap and mixed with `blend()` in linear light - two renders
per frame for the duration of the fade, which the clock's telemetry
shows.

Effects written during render travel with the frame to the encoder and
latch with it. Effects are tile registers and persist on their own; the
runner's only duty is to clear whatever the outgoing animation set when
the animation changes, so the next one does not inherit a fade or hue
shift it never asked for.

Failure isolation. An animation that raises is logged with its id and
traceback, the last good frame is held for that tick (no black, no
skipped latch), and playback advances immediately. Three consecutive
failures disable that entry for the session and flag it in the state.
An overrun is not an error - the clock drops the frame and counts it -
but persistent dropping shows up in `state.warnings`.

Hold. `hold()` stops the countdown of whatever is playing - the
current entry, or a one-off with a hold time - while the animation plays
on: its elapsed and remaining stand still until `hold(False)`, and then
it advances when its remaining reaches zero as usual. A crossfade already
under way finishes first, so a hold never leaves two animations blended.
The hold belongs to the runner, not the entry: skipping while held starts
the next entry with its whole duration, waiting.

Show controls. Speed, freeze, bump, strobe, tint and colour correction
(`overlays.py`) act on whatever is playing, for an operator who does not
know what that is. All but speed transform the output frame, after any
crossfade and before the rotation. Speed scales the time an animation
sees: each playing animation integrates its own clock, `ctx.t` advancing
by the wall step times the speed, and `ctx.dt` scaled to match, so a
speed change never makes an animation jump. Entry durations stay on the
wall clock.

Layers. One animation can run as a LAYER over whatever is playing -
sparkles over a scene, a mask, a second look faded in - composited with
`composite()` (add, max, multiply or mix, faded in by `amount`) before the
show controls. The layer has its own clock, so a base-layer transition
does not restart it, and it follows the show speed like any animation.
Effects are the base layer's: a layer's effect writes are dropped, and
logged once. A layer that raises is logged and removed; the base plays on.

Rotation. The floor-rotation setting turns the picture in 90-degree
steps so the canonical "up" can face whichever way the venue needs. It is
the last thing done to a frame before the fan-out - after crossfades, on
the held frame while paused - so every sink sees what the floor shows, and
no animation ever knows about it. Effect writes move with it: to the
rotated tile, with positional parameters turned round the ring
(`Effect.rotated`). The runner keeps the canonical value of every effect
register it has written so that changing the rotation mid-show can move
the registers already set, clearing the tiles they leave.

Startup and shutdown. With no playlist, or one with nothing playable,
a built-in idle animation runs: a dark floor after boot reads as broken
hardware. `stop()` (and SIGTERM / SIGINT, via `install_signal_handlers()`)
blacks out the floor and latches once more before the fan-out closes;
otherwise the tiles hold the last frame at full power indefinitely.
"""

from __future__ import annotations

import logging
import math
import queue
import random
import signal
import threading
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, TypeVar

from df2_pi.animation.loader import AnimationDef, AnimationError, AnimationRun
from df2_pi.animation.meta import AnimationMeta, check_control_target
from df2_pi.animation.registry import AnimationRegistry
from df2_pi.effects import Effect
from df2_pi.engine.clock import FrameClock, FrameInfo, TelemetrySnapshot
from df2_pi.engine.overlays import (
    DECAY_MAX_S,
    SATURATION_MAX,
    SPEED_MAX,
    Overlays,
    ShowState,
    check_strobe_max,
)
from df2_pi.geometry import ROTATIONS, FloorGeometry, quarter_turns
from df2_pi.pixels import BLEND_MODES, Frame, PixelFrame, TileFrame, blend, composite, default_geometry

if TYPE_CHECKING:
    from df2_pi.output.fanout import FanOut
    # annotation-only: playlists.store imports engine.overlays, and a runtime
    # import here closes a cycle through engine/__init__
    from df2_pi.playlists.store import PlaylistStore, ResolvedEntry, ResolvedPlaylist

log = logging.getLogger(__name__)

T = TypeVar("T")

MAX_STRIKES = 3
EPSILON = 1e-6  # for "duration elapsed" tests; t accumulates float error
DROP_WARNING_THRESHOLD = 3  # dropped frames within the telemetry window


# ---- the built-in idle animation -----------------------------------------------------------


def _idle_render(previous: TileFrame, ctx) -> TileFrame:
    """A slow, dim blue breathe: unmistakably alive, unmistakably not a show."""
    frame = TileFrame.black(ctx.geometry)
    level = 0.5 + 0.5 * math.sin(ctx.t * 2 * math.pi / 4.0)
    frame.data[:] = (0, int(4 + 10 * level), int(12 + 28 * level))
    return frame


IDLE = AnimationDef(
    id="_idle",
    path=Path("<built-in>"),
    meta=AnimationMeta(name="Idle", description="Built-in: nothing is loaded.", format="tile"),
    render=_idle_render,
)


# ---- state ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerState:
    """Immutable snapshot, replaced each frame; safe to read from any thread."""

    playing: bool
    paused: bool
    blacked_out: bool
    brightness: int
    playlist: tuple[int, str] | None
    entry_index: int | None
    entry_count: int
    animation: tuple[str, str] | None
    params: Mapping[str, Any]
    elapsed_s: float
    remaining_s: float | None
    frame: int
    fps: float
    timing: TelemetrySnapshot
    sinks: Mapping[str, Any]
    load_errors: Mapping[str, str]
    disabled_entries: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    one_off: bool = False
    timer_held: bool = False  # the countdown is stopped; what plays, plays until released
    # The playing entry's id: stable when the entries are reordered, which
    # its index (a position in the runner's loaded copy) is not.
    entry_id: int | None = None
    rotation: int = 0  # degrees clockwise: 0, 90, 180 or 270
    show: ShowState = ShowState()
    layer: LayerState | None = None


@dataclass(frozen=True)
class LayerState:
    """The animation layered over what is playing, if any."""

    animation: tuple[str, str]  # (id, name)
    params: Mapping[str, Any]
    mode: str  # one of pixels.BLEND_MODES
    amount: float


@dataclass
class _Layer:
    """An animation running as a layer, and how it combines."""

    playing: _Playing
    mode: str
    amount: float
    warned_effects: bool = False


@dataclass
class _Playing:
    """One entry (or one-off) in progress."""

    definition: AnimationDef
    run: AnimationRun
    started_t: float
    duration_s: float | None  # None: until told otherwise
    entry: ResolvedEntry | None = None
    effects_written: set[int] = field(default_factory=set)
    # The animation's own clock: advances by the wall step times the show speed.
    anim_t: float = 0.0
    last_t: float | None = None

    def advance(self, t: float, speed: float) -> float:
        """The animation time for a frame shown at wall time `t`."""
        if self.last_t is None:
            self.anim_t = self.elapsed(t)
        else:
            self.anim_t += (t - self.last_t) * speed
        self.last_t = t
        return self.anim_t

    def restart_clock(self, t: float) -> None:
        self.started_t = t
        self.anim_t = 0.0
        self.last_t = None

    def elapsed(self, t: float) -> float:
        return t - self.started_t

    def remaining(self, t: float) -> float | None:
        return None if self.duration_s is None else self.duration_s - self.elapsed(t)


# ---- the runner --------------------------------------------------------------------------------


class Runner:
    def __init__(
        self,
        registry: AnimationRegistry,
        fanout: FanOut,
        *,
        store: PlaylistStore | None = None,
        clock: FrameClock | None = None,
        geometry: FloorGeometry | None = None,
        fps: float = 30.0,
        brightness: int = 255,
        rotation: int = 0,
        strobe_max_hz: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.registry = registry
        self.fanout = fanout
        self.store = store
        self.clock = clock if clock is not None else FrameClock(fps=fps)
        self.geometry = geometry if geometry is not None else default_geometry()
        self.fps = self.clock.fps
        self._rng = random.Random(seed)
        self._controls: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()

        self._playlist: ResolvedPlaylist | None = None
        self._cycle: list[int] = []  # indices into playlist.entries, in play order
        self._cycle_pos = 0
        self._current: _Playing | None = None
        self._outgoing: _Playing | None = None  # during a crossfade
        self._one_off: _Playing | None = None
        self._idle: _Playing | None = None
        self._held: Frame | None = None  # the last base frame, never composited
        self._shown: Frame | None = None  # the last composited frame, what a pause holds
        self._layer: _Layer | None = None
        self._playing = False
        self._ended = False  # a non-looping playlist ran out: hold the last frame
        self._paused_at: float | None = None
        self._timer_held = False
        self._timer_t = 0.0  # while held: the wall time the countdown was last held up to
        self._blacked_out = False
        self._brightness = brightness
        self._quarter_turns = quarter_turns(rotation)
        self.overlays = Overlays(strobe_max_hz)
        # Canonical tile -> the effect last written there (NONE is absence),
        # and physical writes owed by a rotation change, sent with the next frame.
        self._registers: dict[int, Effect] = {}
        self._pending_physical: dict[int, Effect] = {}
        self._stopping = False
        self._strikes: dict[int, int] = {}
        self._disabled: set[int] = set()
        self._pending_clear: dict[int, Effect] = {}
        self._drops_at: list[tuple[int, int]] = []  # (frame, dropped) history
        self._thread: threading.Thread | None = None
        self._state = self._snapshot(None)

    # ---- control API (any thread) --------------------------------------------------------

    def _enqueue(self, fn: Callable[[], None]) -> None:
        self._controls.put(fn)

    def play(self) -> None:
        """Start (or restart from the top if the playlist had ended)."""
        self._enqueue(self._do_play)

    def pause(self) -> None:
        """Hold the current frame lit; nothing advances."""
        self._enqueue(self._do_pause)

    def resume(self) -> None:
        self._enqueue(self._do_resume)

    def hold(self, on: bool = True) -> None:
        """Stop the countdown, so what is playing plays until released (on),
        or let it run on from where it stood (off)."""
        on = bool(on)
        self._enqueue(lambda: self._do_hold(on))

    def next(self) -> None:
        self._enqueue(lambda: self._do_skip(+1))

    def previous(self) -> None:
        self._enqueue(lambda: self._do_skip(-1))

    def goto(self, index: int) -> None:
        """Jump to playlist entry `index` (an index into the playlist's
        entries; an unplayable one moves on to the next playable)."""
        self._enqueue(lambda: self._do_goto(index))

    def restart(self) -> None:
        """Restart the current entry from its first frame."""
        self._enqueue(self._do_restart)

    def load_playlist(self, playlist: ResolvedPlaylist | int | str | None) -> None:
        """Swap playlists without stopping the clock. An id or name needs
        the store; None unloads (the idle animation plays)."""
        if isinstance(playlist, (int, str)):
            if self.store is None:
                raise ValueError("load_playlist by id/name needs a PlaylistStore")
            playlist = self.store.resolve(playlist, self.registry)
        self._enqueue(lambda: self._do_load(playlist))

    def play_animation(self, animation_id: str, params: Mapping[str, Any] | None = None, hold: float | None = None) -> None:
        """One-off preview of an animation, `hold` seconds (None: until
        `next()`), then back to the playlist entry that was playing."""
        definition = self.registry.get(animation_id)
        if definition is None:
            raise KeyError(f"no animation {animation_id!r}")
        resolved = definition.meta.resolve_params(params)  # validate on the caller's thread
        self._enqueue(lambda: self._do_play_animation(definition, resolved, hold))

    def set_layer(
        self,
        animation_id: str,
        params: Mapping[str, Any] | None = None,
        mode: str = "add",
        amount: float = 1.0,
    ) -> None:
        """Run `animation_id` as a layer over whatever plays, replacing any
        layer there is. Starts from its first frame."""
        definition = self.registry.get(animation_id)
        if definition is None:
            raise KeyError(f"no animation {animation_id!r}")
        resolved = definition.meta.resolve_params(params)  # validate on the caller's thread
        _check_blend(mode, amount)
        amount = float(amount)
        self._enqueue(lambda: self._do_set_layer(definition, resolved, mode, amount))

    def set_layer_blend(self, mode: str | None = None, amount: float | None = None) -> None:
        """Change how the layer combines without restarting it - what a
        fader binds to. A no-op when there is no layer."""
        _check_blend("add" if mode is None else mode, 0.0 if amount is None else amount)
        amount = None if amount is None else float(amount)
        self._enqueue(lambda: self._do_set_layer_blend(mode, amount))

    def clear_layer(self, animation_id: str | None = None) -> None:
        """Remove the layer - only if it is `animation_id`, when given, so a
        caller can undo its own layer without removing someone else's."""
        self._enqueue(lambda: self._do_clear_layer(animation_id))

    def end_one_off(self, animation_id: str) -> None:
        """End the one-off if it is `animation_id`, returning to the
        playlist; anything else playing is left alone."""
        self._enqueue(lambda: self._do_end_one_off(animation_id))

    def set_params(self, **params: Any) -> None:
        """Live-tune the running animation by param name."""
        self._enqueue(lambda: self._do_set_params(params))

    def set_control(self, target: str, unit: float) -> None:
        """What a MIDI knob, DMX channel or OSC fader binds to: `target` is
        a role ("speed") or a macro ("macro1"), `unit` a 0..1 value. Resolved
        against whatever is playing at the next frame boundary and mapped by
        that param's `from_unit()`; an animation with nothing at `target`
        ignores it. Lands in the live layer, like `set_params()`."""
        check_control_target(target)  # validate on the caller's thread
        unit = float(unit)
        if not math.isfinite(unit):
            raise ValueError(f"control value must be finite, got {unit}")
        self._enqueue(lambda: self._do_set_control(target, unit))

    def set_brightness(self, value: int) -> None:
        if not 0 <= value <= 255:
            raise ValueError(f"brightness must be 0..255, got {value}")
        self._enqueue(lambda: self._do_brightness(int(value)))

    def set_rotation(self, degrees: int) -> None:
        """Turn the picture 0, 90, 180 or 270 degrees clockwise, as a cut
        at the next frame."""
        turns = quarter_turns(degrees)  # validate on the caller's thread
        self._enqueue(lambda: self._do_rotation(turns))

    # ---- show controls: see overlays.py ----

    def set_speed(self, speed: float) -> None:
        """Scale the time animations see, 0..4; 1 is as written."""
        speed = _checked(speed, 0.0, SPEED_MAX, "speed")
        self._enqueue(lambda: self.overlays.set_speed(speed))

    def freeze(self, on: bool = True) -> None:
        """Hold the picture; animations keep running underneath."""
        on = bool(on)
        self._enqueue(lambda: self.overlays.set_freeze(on))

    def bump(self, level: float = 1.0, decay_s: float = 0.25) -> None:
        """A flash toward white by `level`, fading over `decay_s`."""
        level = _checked(level, 0.0, 1.0, "bump level")
        decay_s = _checked(decay_s, 0.01, DECAY_MAX_S, "bump decay")
        self._enqueue(lambda: self.overlays.bump(level, decay_s, self._last_t))

    def set_strobe(self, rate_hz: float) -> None:
        """Shutter the picture at `rate_hz`; 0 is off. Held to the strobe
        cap, however high it is asked for."""
        rate_hz = _checked(rate_hz, 0.0, math.inf, "strobe rate")
        self._enqueue(lambda: self.overlays.set_strobe(rate_hz, self._last_t))

    def set_strobe_max(self, hz: float) -> None:
        """The strobe cap - an admin setting, not a live control."""
        hz = check_strobe_max(hz)
        self._enqueue(lambda: self.overlays.set_strobe_max(hz))

    def set_tint(self, r: int, g: int, b: int, amount: float) -> None:
        """Colourise toward (r, g, b) by `amount`, 0..1; black stays black."""
        rgb = tuple(int(_checked(v, 0, 255, "tint channel")) for v in (r, g, b))
        amount = _checked(amount, 0.0, 1.0, "tint amount")
        self._enqueue(lambda: self.overlays.set_tint(rgb, amount))

    def set_hue_shift(self, turns: float) -> None:
        """Rotate every hue by `turns` (1.0 is all the way round)."""
        turns = _checked(turns, -math.inf, math.inf, "hue shift")
        self._enqueue(lambda: self.overlays.set_hue_shift(turns))

    def set_saturation(self, k: float) -> None:
        """Scale saturation, 0 (grey) .. 2; 1 is unchanged."""
        k = _checked(k, 0.0, SATURATION_MAX, "saturation")
        self._enqueue(lambda: self.overlays.set_saturation(k))

    def reset_show(self) -> None:
        """Every show control back to where it does nothing."""
        self._enqueue(self.overlays.reset)

    def blackout(self) -> None:
        """Broadcast blackout; playback continues underneath."""
        self._enqueue(lambda: self._do_blackout(True))

    def unblackout(self) -> None:
        self._enqueue(lambda: self._do_blackout(False))

    def stop(self) -> None:
        """Clean shutdown: blackout, a final latch, close the fan-out."""
        self._enqueue(self._do_stop)

    def call(self, fn: Callable[[], T]) -> Future[T]:
        """Run `fn` on the render thread at the next frame boundary - after
        a latch, before the next frame goes on the wire - and return a
        future for its result or exception. For work that must not overlap
        a frame on the Row Bus: an admin request that waits for a row's
        reply. It spends that frame's slack, so keep each call to one
        request; a long one makes the clock drop frames."""
        future: Future[T] = Future()

        def run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn())
            except BaseException as exc:
                future.set_exception(exc)

        self._enqueue(run)
        return future

    @property
    def state(self) -> RunnerState:
        return self._state

    # ---- running (the render thread) -----------------------------------------------------

    def run(self) -> None:
        """The render loop. Blocks until `stop()`."""
        if self._current is None and self._one_off is None and self._playlist is None:
            self._ensure_idle()
        try:
            for tick in self.clock.run(latch=self.fanout.latch):
                self._last_t = tick.t  # what control handlers see as "now"
                self._drain_controls()
                if self._stopping:
                    self.clock.stop()
                    break
                frame, effects = self._produce(tick)
                frame = self._show(frame, tick.t)
                frame, effects = self._orient(frame, effects)
                self._state = self._snapshot(tick)
                self.fanout.submit(frame, tick, effects)
                self.clock.mark("submit")
        finally:
            self._shutdown()

    def start(self) -> threading.Thread:
        """`run()` on a daemon thread; returns it."""
        self._thread = threading.Thread(target=self.run, name="render", daemon=True)
        self._thread.start()
        return self._thread

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        """True while `start()`'s thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def install_signal_handlers(self) -> None:
        """SIGTERM and SIGINT become `stop()`. Main thread only."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum: int, frame: Any) -> None:
        log.info("signal %s: stopping", signal.Signals(signum).name)
        self.stop()

    # ---- the loop body --------------------------------------------------------------------

    def _drain_controls(self) -> None:
        while True:
            try:
                fn = self._controls.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception:
                log.exception("control command failed")

    def _produce(self, tick: FrameInfo) -> tuple[Frame, dict[int, Effect]]:
        """The frame for this tick and the effect writes to send with it."""
        if self._paused_at is not None:
            return self._shown if self._shown is not None else self._held_frame(), {}
        base, effects = self._produce_base(tick)
        frame = self._with_layer(base, tick.t)
        self._shown = frame
        return frame, effects

    def _produce_base(self, tick: FrameInfo) -> tuple[Frame, dict[int, Effect]]:
        """What is playing, before any layer: `_held` is only ever this."""
        t = tick.t
        self._hold_countdown(t)
        if self._one_off is not None:
            remaining = self._one_off.remaining(t)
            if remaining is not None and remaining <= EPSILON:
                self._end_one_off()
        if self._one_off is None and self._playing and self._current is not None:
            self._advance_if_due(t)

        playing = self._one_off or self._current
        if playing is None:
            if self._ended:
                return self._held_frame(), dict(self._pending_clear_and_reset())
            playing = self._ensure_idle(t)
        effects: dict[int, Effect] = dict(self._pending_clear)
        self._pending_clear = {}

        frame = self._render(playing, t, effects)
        if frame is None:
            return self._held_frame(), effects

        if self._outgoing is not None and playing is self._current:
            frame = self._crossfade(frame, t, effects)

        self._held = frame
        return frame, effects

    def _hold_countdown(self, t: float) -> None:
        """While held, move the start of the running countdown along with
        the clock, so its elapsed and remaining stand still. Not while a
        crossfade is under way: that finishes, and the hold takes over after."""
        if not self._timer_held:
            return
        gap, self._timer_t = t - self._timer_t, t
        held = self._one_off if self._one_off is not None else self._current if self._outgoing is None else None
        if held is not None:
            # never past now: an entry started this tick has elapsed nothing yet, not less than nothing
            held.started_t = min(held.started_t + gap, t)

    def _with_layer(self, base: Frame, t: float) -> Frame:
        layer = self._layer
        if layer is None:
            return base
        speed = self.overlays.speed
        playing = layer.playing
        try:
            rendered = playing.run.render(t=playing.advance(t, speed), dt=playing.run.dt * speed)
        except AnimationError as exc:
            log.error("layer %s failed and was removed:\n%s", playing.definition.id, "".join(traceback.format_exception(exc)))
            self._layer = None
            return base
        if rendered.effects and not layer.warned_effects:
            layer.warned_effects = True
            log.warning("layer %s writes tile effects; effects belong to the base layer, so they are dropped", playing.definition.id)
        frame = composite(base, rendered.frame, layer.mode, layer.amount)
        self.clock.mark("layer")
        return frame

    def _show(self, frame: Frame, t: float) -> Frame:
        shown = self.overlays.apply(frame, t)
        if shown is not frame:
            self.clock.mark("overlays")
        return shown

    def _orient(self, frame: Frame, effects: dict[int, Effect]) -> tuple[Frame, dict[int, Effect]]:
        """Canonical -> floor: apply the rotation to this frame and its
        effect writes, plus any writes a rotation change is owed."""
        for tile, effect in effects.items():
            if effect == Effect.NONE:
                self._registers.pop(tile, None)
            else:
                self._registers[tile] = effect
        physical, self._pending_physical = self._pending_physical, {}
        if self._quarter_turns == 0:
            if not physical:
                return frame, effects
            physical.update(effects)
            return frame, physical
        physical.update(self._rotate_effects(effects, self._quarter_turns))
        frame = frame.rotated(self._quarter_turns)
        self.clock.mark("rotate")
        return frame, physical

    def _rotate_effects(self, effects: Mapping[int, Effect], turns: int) -> dict[int, Effect]:
        if turns == 0:
            return dict(effects)
        rotation = self.geometry.rotation(turns)
        n = self.geometry.leds_per_tile
        return {
            int(rotation.tile_dest[tile]): effect.rotated(rotation.led_shift, n)
            for tile, effect in effects.items()
        }

    def _render(self, playing: _Playing, t: float, effects: dict[int, Effect]) -> Frame | None:
        """Render one animation for this tick; None if it raised (and the
        failure has been handled)."""
        speed = self.overlays.speed
        try:
            rendered = playing.run.render(t=playing.advance(t, speed), dt=playing.run.dt * speed)
        except AnimationError as exc:
            self._failed(playing, exc)
            return None
        if rendered.effects:
            effects.update(rendered.effects)
            playing.effects_written.update(rendered.effects)
        if playing.entry is not None:
            self._strikes.pop(playing.entry.entry.id, None)  # consecutive means consecutive
        self.clock.mark("render")
        return rendered.frame

    def _crossfade(self, incoming: Frame, t: float, effects: dict[int, Effect]) -> Frame:
        outgoing = self._outgoing
        assert outgoing is not None and self._current is not None
        fade = self._playlist.playlist.crossfade_s if self._playlist else 0.0
        u = 1.0 if fade <= 0 else min(1.0, max(0.0, self._current.elapsed(t) / fade))
        old = self._render(outgoing, t, effects)
        if old is None or u >= 1.0:
            self._finish_outgoing(effects)
            return incoming
        a, b = old, incoming
        if type(a) is not type(b):  # a TileFrame blends into a PixelFrame as its expansion
            a = a.to_pixels() if isinstance(a, TileFrame) else a
            b = b.to_pixels() if isinstance(b, TileFrame) else b
        return blend(a, b, u)

    def _finish_outgoing(self, effects: dict[int, Effect]) -> None:
        outgoing = self._outgoing
        self._outgoing = None
        if outgoing is None:
            return
        for tile in outgoing.effects_written:
            effects.setdefault(tile, Effect.NONE)
        self._log_play(outgoing, "completed")

    def _pending_clear_and_reset(self) -> dict[int, Effect]:
        pending, self._pending_clear = self._pending_clear, {}
        return pending

    def _held_frame(self) -> Frame:
        if self._held is None:
            self._held = TileFrame.black(self.geometry).freeze()
        return self._held

    # ---- advancing ---------------------------------------------------------------------------

    def _advance_if_due(self, t: float) -> None:
        current = self._current
        assert current is not None
        remaining = current.remaining(t)
        if remaining is None:
            return
        fade = self._playlist.playlist.crossfade_s if self._playlist else 0.0
        if current.run.frame == 0:
            return  # every entry shows at least one frame, however short its duration
        if remaining <= EPSILON:
            if self._outgoing is None:
                # A cut (or a fade that never got a partner): the entry is over.
                self._retire_current("completed")
                self._start_next(t)
            return
        if fade > 0 and remaining <= fade + EPSILON and self._outgoing is None:
            pos = self._next_pos()
            if pos is not None:
                # Start the incoming early; the outgoing keeps rendering for the overlap.
                self._outgoing = current
                self._current = None
                self._start_at(pos, t)

    def _retire_current(self, outcome: str) -> None:
        current = self._current
        self._current = None
        if current is None:
            return
        for tile in current.effects_written:
            self._pending_clear.setdefault(tile, Effect.NONE)
        self._log_play(current, outcome)

    def _next_pos(self) -> int | None:
        """The cycle position after the current one; None at the end of a
        non-looping playlist."""
        if self._playlist is None or not self._cycle:
            return None
        pos = self._cycle_pos + 1
        if pos < len(self._cycle):
            return pos
        if not self._playlist.playlist.loop:
            return None
        if self._playlist.playlist.shuffle:
            self._reshuffle()
        return 0

    def _start_next(self, t: float) -> None:
        """Start the entry after the current one, or end playback."""
        if self._current is not None or self._outgoing is not None:
            return
        pos = self._next_pos()
        if pos is None:
            self._playing = False  # ended: hold the final frame
            self._ended = True
            return
        self._start_at(pos, t)

    def _start_at(self, pos: int, t: float) -> None:
        """Start the playable entry at cycle position `pos`, skipping past
        disabled ones; ends playback if every entry is disabled."""
        if self._playlist is None or not self._cycle:
            self._playing = False
            return
        for _ in range(len(self._cycle)):
            entry = self._playlist.entries[self._cycle[pos]]
            if entry.entry.id not in self._disabled:
                self._cycle_pos = pos
                self._start_entry(entry, t)
                return
            pos = (pos + 1) % len(self._cycle)
        self._playing = False

    def _start_entry(self, entry: ResolvedEntry, t: float) -> None:
        assert entry.definition is not None
        run = entry.definition.start(self.geometry, entry.params, fps=self.fps, seed=self._rng.getrandbits(32))
        self._current = _Playing(entry.definition, run, t, entry.entry.duration_s, entry)
        self._ended = False
        log.info("playing %s (%s) for %.1f s", entry.definition.id, entry.definition.meta.name, entry.entry.duration_s)

    def _reshuffle(self) -> None:
        self._rng.shuffle(self._cycle)

    def _ensure_idle(self, t: float = 0.0) -> _Playing:
        if self._idle is None:
            self._idle = _Playing(IDLE, IDLE.start(self.geometry, fps=self.fps, seed=0), t, None)
        return self._idle

    def _failed(self, playing: _Playing, exc: AnimationError) -> None:
        definition = playing.definition
        log.error("animation %s failed:\n%s", definition.id, "".join(traceback.format_exception(exc)))
        if playing is self._one_off:
            self._end_one_off(outcome="error")
            return
        if playing is self._idle:
            return
        if playing is self._outgoing:
            self._outgoing = None
            return
        entry = playing.entry
        if entry is not None:
            eid = entry.entry.id
            self._strikes[eid] = self._strikes.get(eid, 0) + 1
            if self._strikes[eid] >= MAX_STRIKES:
                self._disabled.add(eid)
                log.error("entry %d (%s) disabled after %d consecutive failures", eid, definition.id, MAX_STRIKES)
        self._retire_current("error")
        # advance immediately: the next tick renders the next entry
        self._start_next(self._last_t)

    def _log_play(self, playing: _Playing, outcome: str) -> None:
        if self.store is None or playing.entry is None:
            return
        try:
            self.store.log_play(
                playing.definition.id,
                outcome,
                playlist_id=self._playlist.playlist.id if self._playlist else None,
                duration_s=playing.elapsed(self._last_t),
            )
        except Exception:
            log.exception("play log write failed")

    # ---- control handlers (render thread) ----------------------------------------------------

    def _do_play(self) -> None:
        if self._paused_at is not None:
            self._do_resume()
            return
        if self._playing or self._playlist is None:
            return
        self._playing = True
        if self._current is None:
            if self._playlist.playlist.shuffle:
                self._reshuffle()
            self._start_at(0, self._last_t)

    def _do_pause(self) -> None:
        if self._paused_at is None:
            self._hold_countdown(self._last_t)  # bring a held countdown up to the moment it stops
            self._paused_at = self._last_t

    def _do_resume(self) -> None:
        if self._paused_at is None:
            return
        gap = self._last_t - self._paused_at
        self._paused_at = None
        self._timer_t += gap  # the pause has already been made up below
        layered = self._layer.playing if self._layer is not None else None
        for playing in (self._current, self._outgoing, self._one_off, layered):
            if playing is not None:
                playing.started_t += gap
                if playing.last_t is not None:
                    playing.last_t += gap

    def _do_hold(self, on: bool) -> None:
        if on and not self._timer_held:
            self._timer_t = self._last_t
        self._timer_held = on

    def _do_skip(self, direction: int) -> None:
        if self._one_off is not None:
            self._end_one_off(outcome="skipped")
            return
        if self._playlist is None or not self._cycle:
            return
        self._outgoing = None
        self._retire_current("skipped")
        self._playing = True
        if direction > 0:
            self._start_next(self._last_t)
        else:
            self._start_at((self._cycle_pos - 1) % len(self._cycle), self._last_t)

    def _do_goto(self, index: int) -> None:
        if self._playlist is None or not self._cycle:
            return
        if index not in self._cycle:
            later = [i for i in self._cycle if i > index]
            if not later:
                return
            index = min(later)
        self._outgoing = None
        self._retire_current("skipped")
        self._playing = True
        self._start_at(self._cycle.index(index), self._last_t)

    def _do_restart(self) -> None:
        playing = self._one_off or self._current
        if playing is None:
            return
        params = playing.run.params
        playing.run = playing.definition.start(self.geometry, params, fps=self.fps, seed=self._rng.getrandbits(32))
        playing.restart_clock(self._last_t)

    def _do_load(self, playlist: ResolvedPlaylist | None) -> None:
        self._outgoing = None
        self._retire_current("skipped")
        self._playlist = playlist
        self._disabled.clear()
        self._strikes.clear()
        self._cycle = [] if playlist is None else [
            i for i, entry in enumerate(playlist.entries) if entry.playable
        ]
        if playlist is not None and playlist.playlist.shuffle:
            self._reshuffle()
        for entry in (playlist.entries if playlist else ()):
            if entry.unresolved:
                log.warning("playlist %r: entry %d (%s) unresolved: %s", playlist.playlist.name, entry.entry.position, entry.entry.animation_id, entry.error)
        self._playing = bool(self._cycle)
        self._ended = False
        if self._playing:
            self._start_at(0, self._last_t)
        else:
            log.warning("no playable entries; the idle animation will play")

    def _do_play_animation(self, definition: AnimationDef, params: dict[str, Any], hold: float | None) -> None:
        if self._one_off is not None:
            self._end_one_off(outcome="skipped")
        run = definition.start(self.geometry, params, fps=self.fps, seed=self._rng.getrandbits(32))
        self._one_off = _Playing(definition, run, self._last_t, hold)
        if self._current is not None:
            for tile in self._current.effects_written:
                self._pending_clear.setdefault(tile, Effect.NONE)

    def _end_one_off(self, outcome: str = "completed") -> None:
        one_off = self._one_off
        self._one_off = None
        if one_off is None:
            return
        for tile in one_off.effects_written:
            self._pending_clear.setdefault(tile, Effect.NONE)
        # back to the playlist: restart the entry that was playing
        if self._current is not None:
            self._do_restart_current()

    def _do_restart_current(self) -> None:
        current = self._current
        if current is None:
            return
        current.run = current.definition.start(self.geometry, current.run.params, fps=self.fps, seed=self._rng.getrandbits(32))
        current.restart_clock(self._last_t)
        current.effects_written.clear()

    def _do_end_one_off(self, animation_id: str) -> None:
        if self._one_off is not None and self._one_off.definition.id == animation_id:
            self._end_one_off(outcome="completed")

    def _do_set_params(self, params: dict[str, Any]) -> None:
        playing = self._one_off or self._current
        if playing is None:
            return
        merged = dict(playing.run.params)
        merged.update(params)
        try:
            playing.run.set_params(merged)
        except ValueError as exc:
            log.warning("set_params rejected: %s", exc)

    def _do_set_control(self, target: str, unit: float) -> None:
        playing = self._one_off or self._current
        if playing is None:
            return
        meta = playing.definition.meta
        key = meta.control(target)
        if key is None:
            return
        self._do_set_params({key: meta.params[key].from_unit(unit)})

    def _do_set_layer(self, definition: AnimationDef, params: dict[str, Any], mode: str, amount: float) -> None:
        run = definition.start(self.geometry, params, fps=self.fps, seed=self._rng.getrandbits(32))
        self._layer = _Layer(_Playing(definition, run, self._last_t, None), mode, amount)

    def _do_set_layer_blend(self, mode: str | None, amount: float | None) -> None:
        if self._layer is None:
            return
        if mode is not None:
            self._layer.mode = mode
        if amount is not None:
            self._layer.amount = amount

    def _do_clear_layer(self, animation_id: str | None) -> None:
        if self._layer is not None and animation_id in (None, self._layer.playing.definition.id):
            self._layer = None

    def _do_brightness(self, value: int) -> None:
        self._brightness = value
        self.fanout.set_brightness(value)

    def _do_rotation(self, turns: int) -> None:
        old, self._quarter_turns = self._quarter_turns, turns
        if old == turns or not self._registers:
            return
        # Move every register that is set: clear the tiles it leaves, write
        # the tiles it lands on. Rides with the next frame, like any effect write.
        before = self._rotate_effects(self._registers, old)
        after = self._rotate_effects(self._registers, turns)
        owed = {tile: Effect.NONE for tile in before if tile not in after}
        owed.update({tile: effect for tile, effect in after.items() if before.get(tile) != effect})
        self._pending_physical.update(owed)

    def _do_blackout(self, on: bool) -> None:
        self._blacked_out = on
        if on:
            self.fanout.blackout()
        else:
            self.fanout.unblackout()

    def _do_stop(self) -> None:
        self._stopping = True

    def _shutdown(self) -> None:
        try:
            self.fanout.blackout()
            self.fanout.latch()
        except Exception:
            log.exception("blackout on shutdown failed")
        finally:
            self.fanout.close()

    # ---- snapshot --------------------------------------------------------------------------

    _last_t: float = 0.0

    def _snapshot(self, tick: FrameInfo | None) -> RunnerState:
        t = tick.t if tick is not None else self._last_t
        playing = self._one_off or self._current or self._idle
        playlist = self._playlist.playlist if self._playlist else None
        entry_index = entry_id = None
        if self._current is not None and self._current.entry is not None and self._playlist is not None:
            entry_index = self._playlist.entries.index(self._current.entry)
            entry_id = self._current.entry.entry.id
        # A paused countdown reads as it stood when the pause began; resuming shifts it on to match.
        counted_t = self._paused_at if self._paused_at is not None else t
        remaining = playing.remaining(counted_t) if playing else None
        warnings: list[str] = []
        if tick is not None:
            self._drops_at.append((tick.n, self.clock.dropped))
            window = self.clock.window
            self._drops_at = [(n, d) for n, d in self._drops_at if n >= tick.n - window]
            recent = self.clock.dropped - self._drops_at[0][1]
            if recent >= DROP_WARNING_THRESHOLD:
                warnings.append(f"dropping frames: {recent} in the last {window} frames")
        for sink_name, entry in self.fanout.state()["sinks"].items():
            if entry.get("attached") and entry.get("healthy") is False:
                warnings.append(f"sink {sink_name} is unhealthy")
            if not entry.get("attached"):
                warnings.append(f"sink {sink_name} detached: {entry.get('reason')}")
        return RunnerState(
            playing=self._playing,
            paused=self._paused_at is not None,
            blacked_out=self._blacked_out,
            brightness=self._brightness,
            playlist=(playlist.id, playlist.name) if playlist else None,
            entry_index=entry_index,
            entry_count=len(self._playlist.entries) if self._playlist else 0,
            animation=(playing.definition.id, playing.definition.meta.name) if playing else None,
            params=dict(playing.run.params) if playing else {},
            elapsed_s=max(0.0, playing.elapsed(counted_t)) if playing else 0.0,
            remaining_s=None if remaining is None else max(0.0, remaining),
            frame=tick.n if tick is not None else 0,
            fps=self.fps,
            timing=self.clock.telemetry(),
            sinks=self.fanout.state()["sinks"],
            load_errors={k: v.message for k, v in self.registry.errors.items()},
            disabled_entries=tuple(sorted(self._disabled)),
            warnings=tuple(warnings),
            one_off=self._one_off is not None,
            timer_held=self._timer_held,
            entry_id=entry_id,
            rotation=ROTATIONS[self._quarter_turns],
            show=self.overlays.state(),
            layer=self._layer_state(),
        )

    def _layer_state(self) -> LayerState | None:
        layer = self._layer
        if layer is None:
            return None
        definition = layer.playing.definition
        return LayerState(
            animation=(definition.id, definition.meta.name),
            params=dict(layer.playing.run.params),
            mode=layer.mode,
            amount=layer.amount,
        )


def _checked(value: float, low: float, high: float, what: str) -> float:
    """`value` as a float if it is a finite number in low..high (bounds
    may be infinite); ValueError otherwise. Validates on the caller's
    thread, so a bad value never reaches the control queue."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None
    if math.isnan(number) or not low <= number <= high or math.isinf(number):
        raise ValueError(f"{what} must be in {low}..{high}, got {value!r}")
    return number


def _check_blend(mode: str, amount: float) -> None:
    """ValueError unless `mode` is a blend mode and `amount` is in 0..1."""
    if mode not in BLEND_MODES:
        raise ValueError(f"blend mode must be one of {BLEND_MODES}, got {mode!r}")
    _checked(amount, 0.0, 1.0, "layer amount")
