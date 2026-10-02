"""The floor's controls as addresses: one vocabulary for every adapter.

    controls = FloorControls(runner, registry, store=store, beat=beat, palettes=palettes, external=external)
    controls.apply("/floor/palette/fire", [1.0])
    controls.apply("/floor/brightness", [0.6])

OSC (osc.py) hands each message straight here, and a MIDI binding (midi.py)
names an address to send its control's value to - so a fader on
`/floor/brightness` behaves the same over OSC and MIDI, and both map values
the way the DMX control block maps a fader (speed 0.5 normal, source in
thirds). The addresses and their values are documented in osc.py and
docs/external-input.md.

PRESS addresses (next, play/<id>, palette/<name>, tap ...) fire on no value
at all, or on a value of 0.5 or more that is a rise from below 0.5 or comes
more than REPRESS_S after the previous high one: a button sending 1 then 0
fires once, a sender that only ever sends 1 fires on every press, and one
repeating 1 every frame while held fires once.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from df2_pi.animation import AnimationRegistry
    from df2_pi.engine.runner import Runner
    from df2_pi.interfacing.beat_service import BeatService
    from df2_pi.interfacing.service import ExternalInput
    from df2_pi.palette import PaletteBook
    from df2_pi.playlists import PlaylistStore

BUMP_DECAY_S = 0.25
REPRESS_S = 0.3  # a high value this long after the last one is a new press, even with no release between


class ControlError(ValueError):
    """An address the floor understood, but values it couldn't act on."""


def speed_from_unit(u: float) -> float:
    """0 stops, 0.5 is normal, 1 is four times - the DMX speed fader's curve."""
    return 2.0 * u if u <= 0.5 else 1.0 + (u - 0.5) * 6.0


def source_from_unit(u: float) -> str:
    return "internal" if u < 1 / 3 else "external" if u < 2 / 3 else "mix"


class FloorControls:
    def __init__(
        self,
        runner: Runner,
        registry: AnimationRegistry,
        *,
        store: PlaylistStore | None = None,
        beat: BeatService | None = None,
        palettes: PaletteBook | None = None,
        external: ExternalInput | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runner = runner
        self.registry = registry
        self.store = store
        self.beat = beat
        self.palettes = palettes
        self.external = external
        self._now = now
        self._lock = threading.Lock()
        self._levels: dict[str, tuple[float, float]] = {}  # press addresses: (last value, when)

    def apply(self, address: str, args: list[Any]) -> None:
        """Act on one /floor/... address with its values. Raises
        ControlError (a ValueError) when it can't."""
        parts = [p for p in address.split("/") if p]
        if not parts or parts[0] != "floor":
            raise ControlError("not a /floor address")
        head, rest = (parts[1] if len(parts) > 1 else ""), parts[2:]
        runner = self.runner

        if head == "brightness":
            runner.set_brightness(round(_unit(args) * 255))
        elif head == "blackout":
            runner.blackout() if _on(args) else runner.unblackout()
        elif head == "speed":
            runner.set_speed(speed_from_unit(_unit(args)))
        elif head == "strobe":
            runner.set_strobe(_unit(args) * runner.state.show.strobe_max_hz)
        elif head == "bump":
            level = _unit(args) if args else 1.0
            if level > 0:
                runner.bump(level, BUMP_DECAY_S)
        elif head == "tint":
            if len(args) < 4:
                raise ControlError("tint takes r g b amount")
            r, g, b, amount = (_clamp(_number(a)) for a in args[:4])
            runner.set_tint(round(r * 255), round(g * 255), round(b * 255), amount)
        elif head == "hue":
            runner.set_hue_shift(_number(args[0]))
        elif head == "saturation":
            runner.set_saturation(_unit(args) * 2.0)
        elif head == "freeze":
            runner.freeze(_on(args))
        elif head == "hold":
            runner.hold(_on(args))
        elif head == "reset":
            if self._pressed(address, args):
                runner.reset_show()
        elif head in ("source", "mix"):
            if self.external is None:
                raise ControlError("external input is not running")
            if head == "mix":
                self.external.source.set_mix(_unit(args))
            else:
                value = args[0] if args else None
                self.external.source.set_source(value if isinstance(value, str) else source_from_unit(_unit(args)))
        elif head in ("next", "previous", "restart"):
            if self._pressed(address, args):
                getattr(runner, head)()
        elif head == "goto":
            if rest:
                if self._pressed(address, args):
                    runner.goto(int(rest[0]))
            else:
                runner.goto(int(_number(args[0])))
        elif head == "playlist":
            if rest:
                if self._pressed(address, args):
                    self._load(rest[0])
            else:
                self._load(args[0])
        elif head == "play":
            if rest:
                if self._pressed(address, args):
                    self._play(rest[0])
            else:
                self._play(str(args[0]))
        elif head == "palette":
            if rest:
                if self._pressed(address, args):
                    self._palette(rest[0])
            else:
                self._palette(args[0])
        elif head == "param" and rest:
            self._param(rest[0], _unit(args))
        elif head == "macro" and rest:
            runner.set_control(f"macro{int(rest[0])}", _unit(args))
        elif head == "trigger":
            if rest:
                slot, velocity = int(rest[0]), (_unit(args) if args else 1.0)
            else:
                slot, velocity = int(_number(args[0])), (_unit(args[1:]) if len(args) > 1 else 1.0)
            if velocity > 0:  # 0 is a note-off or a release, not a hit
                runner.trigger(slot, velocity)
        elif head == "tempo" and rest:
            if self.beat is None:
                raise ControlError("beat sync is not running")
            if rest[0] == "tap" and self._pressed(address, args):
                self.beat.tap()
            elif rest[0] == "resync" and self._pressed(address, args):
                self.beat.resync()
            elif rest[0] == "nudge":
                self.beat.nudge(_number(args[0]))
        else:
            raise ControlError(f"no such address {address}")

    def _pressed(self, address: str, args: list[Any]) -> bool:
        """A press: no value at all, or a high value (0.5+) that rises from
        low or comes more than REPRESS_S after the previous high one."""
        if not args:
            return True
        value, now = _number(args[0]), self._now()
        with self._lock:
            before, when = self._levels.get(address, (0.0, float("-inf")))
            self._levels[address] = (value, now)
        return value >= 0.5 and (before < 0.5 or now - when > REPRESS_S)

    def _load(self, which: Any) -> None:
        if self.store is None:
            raise ControlError("no playlist store")
        key = int(which) if isinstance(which, (int, float)) or (isinstance(which, str) and which.isdigit()) else str(which)
        self.runner.load_playlist(self.store.resolve(key, self.registry))

    def _play(self, animation_id: str) -> None:
        if self.registry.get(animation_id) is None:
            raise ControlError(f"no animation {animation_id!r}")
        self.runner.play_animation(animation_id)

    def _palette(self, which: Any) -> None:
        if self.palettes is None:
            raise ControlError("no palettes")
        if isinstance(which, (int, float)) or (isinstance(which, str) and which.isdigit()):
            names = self.palettes.names()
            index = int(which)
            if not 0 <= index < len(names):
                raise ControlError(f"palette {index}: there are {len(names)}")
            which = names[index]
        self.palettes.activate(str(which))

    def _param(self, name: str, u: float) -> None:
        playing = self.runner.state.animation
        definition = self.registry.get(playing[0]) if playing else None
        if definition is None:
            raise ControlError("nothing with parameters is playing")
        spec = definition.meta.params.get(name)
        if spec is None:
            raise ControlError(f"{definition.id} has no parameter {name!r}")
        self.runner.set_params(**{name: spec.from_unit(u)})


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ControlError(f"expected a number, got {value!r}")


def _clamp(u: float) -> float:
    return min(1.0, max(0.0, u))


def _unit(args: list[Any]) -> float:
    if not args:
        raise ControlError("expected a value")
    return _clamp(_number(args[0]))


def _on(args: list[Any]) -> bool:
    return True if not args else _number(args[0]) >= 0.5
