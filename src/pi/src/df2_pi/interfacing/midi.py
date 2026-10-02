"""MIDI input (#131): controllers, and anything else that sends MIDI, run the floor.

    midi = MidiControl(controls, store=store, beat=beat)
    midi.begin()                  # opens every input port, and keeps watching for new ones
    midi.handle("APC MINI", mido.Message("note_on", note=0, velocity=127))
    midi.learn("/floor/palette/fire")   # the next note or CC moved becomes this binding
    midi.stop()

Transport. `mido` over `python-rtmidi` (ALSA on the Pi, CoreMIDI on a Mac).
Every input port is opened except ALSA's internal "Midi Through", and the
port list is re-read every couple of seconds, so a controller plugged in
mid-show is picked up and one unplugged is let go. A laptop reaches the Pi
over a USB-MIDI interface or network MIDI (RTP-MIDI, `rtpmidid`).

The mapping. A YAML file binds MIDI controls to floor ADDRESSES - the same
`/floor/...` vocabulary OSC uses (controls.py), so a fader on
`/floor/brightness` behaves the same over either:

    program_change: true      # Program Change, with Bank Select CC0/CC32: bank = playlist, program = entry
    clock: true               # MIDI clock / Start / Stop / Song Position to beat sync
    bindings:
      - {note: 0, to: /floor/goto/0}                  # a pad: its velocity, 0..1; release sends 0
      - {cc: 48, to: /floor/macro/1}                  # a fader: 0..127 as 0..1
      - {cc: [7, 39], to: /floor/speed}               # a 14-bit pair: MSB, LSB
      - {note: 64, to: /floor/blackout, toggle: true} # each press flips it
      - {note: 82, to: /floor/speed, value: 0.25}     # a press sends this value instead
      - {note: 3, to: /floor/bump, channel: 10, port: launchpad}   # only this channel (1-16) / port

`port` matches any port whose name contains it, ignoring case. The file
lives beside the database (`$DF2_MIDI_MAP` overrides), not in the package:
`sync-to-pi.sh --delete` would wipe a mapping learned on the Pi. If there is
none, the APC mini default (midi_apc_mini.yaml) is copied there. It is
re-read when it changes, so it can be edited by hand on a running floor.

MIDI learn. `learn(address)` arms it; the next note-on or CC from any port
becomes a binding to that address - replacing any binding that control
already had - and the file is saved.

Bank and program. A Program Change loads playlist BANK (counting from 0 in
name order, like the DMX control block's bank channel) if it isn't loaded,
then goes to entry PROGRAM. Launch quantization (#125) applies.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from df2_pi.interfacing.controls import ControlError, FloorControls

if TYPE_CHECKING:
    from df2_pi.interfacing.beat_service import BeatService
    from df2_pi.playlists import PlaylistStore

log = logging.getLogger(__name__)

DEFAULT_MAP = Path(__file__).with_name("midi_apc_mini.yaml")
POLL_S = 2.0  # how often the port list and the mapping file are re-read
RECENT = 20
IGNORED_PORTS = ("midi through",)
CLOCK_TYPES = ("clock", "start", "continue", "stop", "songpos")


@dataclass(frozen=True)
class Binding:
    kind: str  # "note" or "cc"
    number: int  # the note, or the CC (a 14-bit pair's MSB)
    to: str  # a /floor/... address
    lsb: int | None = None  # a 14-bit pair's LSB controller
    channel: int | None = None  # 1..16; None: any
    port: str | None = None  # matches a port whose name contains it, ignoring case; None: any
    toggle: bool = False  # each press flips it on or off; releases ignored
    value: float | None = None  # a press sends this instead of the control's value; releases ignored

    def __post_init__(self) -> None:
        if self.kind not in ("note", "cc"):
            raise ValueError(f"a binding is a note or a cc, got {self.kind!r}")
        for n in (self.number, self.lsb):
            if n is not None and not 0 <= n <= 127:
                raise ValueError(f"MIDI note and CC numbers are 0..127, got {n}")
        if self.channel is not None and not 1 <= self.channel <= 16:
            raise ValueError(f"channel is 1..16, got {self.channel}")
        if not self.to.startswith("/floor/"):
            raise ValueError(f"a binding goes to a /floor/... address, got {self.to!r}")

    def matches(self, port: str, channel: int) -> bool:
        return (self.channel is None or self.channel == channel) and (self.port is None or self.port.lower() in port.lower())

    def control(self) -> str:
        """How the web UI names the control: "note 0", "cc 7+39 ch 2 (apc)"."""
        text = f"{self.kind} {self.number}" + (f"+{self.lsb}" if self.lsb is not None else "")
        if self.channel is not None:
            text += f" ch {self.channel}"
        if self.port:
            text += f" ({self.port})"
        return text

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {self.kind: [self.number, self.lsb] if self.lsb is not None else self.number, "to": self.to}
        if self.channel is not None:
            out["channel"] = self.channel
        if self.port:
            out["port"] = self.port
        if self.toggle:
            out["toggle"] = True
        if self.value is not None:
            out["value"] = self.value
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Binding:
        kinds = [k for k in ("note", "cc") if k in d]
        if len(kinds) != 1:
            raise ValueError(f"a binding has exactly one of note: or cc:, got {sorted(d)}")
        kind = kinds[0]
        number = d[kind]
        lsb = None
        if isinstance(number, list):
            if kind != "cc" or len(number) != 2:
                raise ValueError("only a cc can be a [MSB, LSB] pair")
            number, lsb = number
        unknown = set(d) - {kind, "to", "channel", "port", "toggle", "value"}
        if unknown:
            raise ValueError(f"unknown binding keys {sorted(unknown)}")
        return cls(
            kind=kind,
            number=int(number),
            to=str(d["to"]),
            lsb=None if lsb is None else int(lsb),
            channel=None if d.get("channel") is None else int(d["channel"]),
            port=None if d.get("port") in (None, "") else str(d["port"]),
            toggle=bool(d.get("toggle", False)),
            value=None if d.get("value") is None else float(d["value"]),
        )


@dataclass
class MidiMap:
    bindings: list[Binding] = field(default_factory=list)
    program_change: bool = True
    clock: bool = True

    @classmethod
    def parse(cls, text: str) -> MidiMap:
        import yaml

        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("a MIDI map is a mapping with a bindings: list")
        bindings = []
        for i, item in enumerate(data.get("bindings") or []):
            try:
                bindings.append(Binding.from_dict(item))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"binding {i + 1}: {exc}") from exc
        return cls(bindings, bool(data.get("program_change", True)), bool(data.get("clock", True)))

    def dump(self) -> str:
        import yaml

        lines = [
            "# Dance floor MIDI mapping. Edit by hand, or with MIDI learn in the web UI's External tab;",
            "# the floor re-reads it when it changes. Addresses: docs/external-input.md#osc.",
            f"program_change: {str(self.program_change).lower()}   # Program Change + Bank Select: bank = playlist, program = entry",
            f"clock: {str(self.clock).lower()}            # MIDI clock to beat sync (beat source: MIDI clock)",
            "bindings:",
        ]
        for b in self.bindings:
            lines.append("  - " + yaml.safe_dump(b.to_dict(), default_flow_style=True, sort_keys=False, width=1000).strip())
        return "\n".join(lines) + "\n"

    def lookup(self, kind: str, number: int, port: str, channel: int) -> list[Binding]:
        return [b for b in self.bindings if b.kind == kind and b.number == number and b.matches(port, channel)]

    def lsb_of(self, number: int, port: str, channel: int) -> list[Binding]:
        """The 14-bit bindings whose LSB controller this is."""
        return [b for b in self.bindings if b.kind == "cc" and b.lsb == number and b.matches(port, channel)]


def default_map_path(store: PlaylistStore | None = None) -> Path:
    env = os.environ.get("DF2_MIDI_MAP")
    if env:
        return Path(env).expanduser()
    if store is not None and store.path != ":memory:":
        return Path(store.path).parent / "midi.yaml"
    from df2_pi.playlists.store import default_db_path

    return default_db_path().parent / "midi.yaml"


class MidiControl:
    def __init__(
        self,
        controls: FloorControls,
        *,
        store: PlaylistStore | None = None,
        beat: BeatService | None = None,
        map_path: Path | None = None,
        lister: Callable[[], list[str]] | None = None,
        opener: Callable[[str, Callable[[Any], None]], Any] | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.controls = controls
        self.store = store
        self.beat = beat
        self.map_path = map_path or default_map_path(store)
        self._lister = lister
        self._opener = opener
        self._now = now
        self._lock = threading.RLock()
        self.map = MidiMap()
        self.map_error: str | None = None
        self._map_mtime: int | None = None
        self._ports: dict[str, Any] = {}
        self._port_errors: dict[str, str] = {}
        self._bank: dict[tuple[str, int], tuple[int, int]] = {}  # (port, channel) -> (bank MSB, LSB)
        self._msb: dict[tuple[str, int, int], int] = {}  # 14-bit pairs: the MSB's value
        self._lsb: dict[tuple[str, int, int], int] = {}  # ... and the LSB's, reset when the MSB moves
        self._toggled: dict[Binding, bool] = {}
        self.learning: dict[str, Any] | None = None
        self.received = 0
        self.clock_received = 0
        self.errors = 0
        self.recent: deque[dict[str, Any]] = deque(maxlen=RECENT)
        self.enabled = True
        if store is not None:
            stored = store.get_bool("midi_enabled")
            self.enabled = True if stored is None else stored
        self._running = False
        self._thread: threading.Thread | None = None
        self.load_map()

    # ---- the mapping file ---------------------------------------------------------------------

    def load_map(self) -> None:
        """Read the mapping file, creating it from the APC mini default if
        there is none. A file that doesn't parse keeps the last good map."""
        path = self.map_path
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(DEFAULT_MAP, path)
                log.info("MIDI: no mapping at %s; copied the APC mini default", path)
            mtime = path.stat().st_mtime_ns
            parsed = MidiMap.parse(path.read_text())
        except (OSError, ValueError) as exc:
            self.map_error = f"{path}: {exc}"
            log.warning("MIDI map: %s", self.map_error)
            return
        with self._lock:
            self.map, self._map_mtime, self.map_error = parsed, mtime, None
            self._toggled.clear()

    def save_map(self) -> None:
        with self._lock:
            text = self.map.dump()
            self.map_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.map_path.with_suffix(".yaml.tmp")
            tmp.write_text(text)
            tmp.replace(self.map_path)
            self._map_mtime = self.map_path.stat().st_mtime_ns

    def reset_to_default(self) -> None:
        shutil.copyfile(DEFAULT_MAP, self.map_path)
        self.load_map()

    def remove_binding(self, index: int) -> None:
        with self._lock:
            del self.map.bindings[index]
            self.save_map()

    def set_options(self, *, program_change: bool | None = None, clock: bool | None = None) -> None:
        with self._lock:
            if program_change is not None:
                self.map.program_change = bool(program_change)
            if clock is not None:
                self.map.clock = bool(clock)
            self.save_map()

    def _reload_if_changed(self) -> None:
        try:
            mtime = self.map_path.stat().st_mtime_ns
        except OSError:
            mtime = None
        if mtime != self._map_mtime:
            self.load_map()

    # ---- learn ------------------------------------------------------------------------------------

    def learn(self, to: str, *, toggle: bool = False, value: float | None = None, this_port_only: bool = False) -> None:
        Binding("note", 0, to, toggle=toggle, value=value)  # validates the address and options
        with self._lock:
            self.learning = {"to": to, "toggle": toggle, "value": value, "this_port_only": this_port_only, "since": time.time()}

    def cancel_learn(self) -> None:
        with self._lock:
            self.learning = None

    def _learn_from(self, port: str, kind: str, number: int, channel: int) -> None:
        learning = self.learning
        assert learning is not None
        binding = Binding(
            kind, number, learning["to"],
            port=_port_label(port) if learning["this_port_only"] else None,
            toggle=learning["toggle"],
            value=learning["value"],
        )
        same = lambda b: b.kind == kind and b.number == number and b.matches(port, channel)  # noqa: E731
        self.map.bindings = [b for b in self.map.bindings if not same(b)] + [binding]
        self.learning = None
        self.save_map()
        log.info("MIDI learn: %s -> %s", binding.control(), binding.to)

    # ---- ports ------------------------------------------------------------------------------------

    def begin(self) -> None:
        if not self.enabled or self._running:
            return
        self._running = True
        self.poll()
        self._thread = threading.Thread(target=self._run, name="midi-ports", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(POLL_S + 1)
            self._thread = None
        with self._lock:
            ports, self._ports = self._ports, {}
        for port in ports.values():
            try:
                port.close()
            except Exception:  # pragma: no cover - a port that vanished mid-close
                pass

    def configure(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if self.store is not None:
            self.store.set_setting("midi_enabled", self.enabled)
        self.stop()
        self.begin()

    def _run(self) -> None:
        while self._running:
            time.sleep(POLL_S)
            try:
                self.poll()
            except Exception:
                log.exception("MIDI port poll failed")

    def poll(self) -> None:
        """Open new ports, let go of vanished ones, and re-read the map if it changed."""
        self._reload_if_changed()
        lister, opener = self._lister, self._opener
        if lister is None or opener is None:
            try:
                import mido
            except ImportError:
                self._port_errors["*"] = 'MIDI needs mido and python-rtmidi: pip install -e ".[midi]"'
                return
            lister = lister or mido.get_input_names
            opener = opener or (lambda name, callback: mido.open_input(name, callback=callback))
        try:
            names = [n for n in lister() if not any(skip in n.lower() for skip in IGNORED_PORTS)]
        except Exception as exc:
            self._port_errors["*"] = f"could not list MIDI ports: {exc}"
            return
        self._port_errors.pop("*", None)
        with self._lock:
            gone = [n for n in self._ports if n not in names]
            for name in gone:
                port = self._ports.pop(name)
                try:
                    port.close()
                except Exception:
                    pass
                log.info("MIDI: %s went away", name)
        for name in names:
            if name in self._ports:
                continue
            try:
                port = opener(name, lambda message, name=name: self.handle(name, message))
            except Exception as exc:
                self._port_errors[name] = str(exc)
                continue
            self._port_errors.pop(name, None)
            with self._lock:
                self._ports[name] = port
            log.info("MIDI: listening to %s", name)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "ports": sorted(self._ports),
                "port_errors": dict(self._port_errors),
                "received": self.received,
                "clock_received": self.clock_received,
                "errors": self.errors,
                "recent": list(reversed(self.recent)),
                "learning": self.learning,
                "map_path": str(self.map_path),
                "map_error": self.map_error,
            }

    def mapping(self) -> dict[str, Any]:
        with self._lock:
            return {
                "program_change": self.map.program_change,
                "clock": self.map.clock,
                "bindings": [{**b.to_dict(), "control": b.control()} for b in self.map.bindings],
            }

    # ---- messages ---------------------------------------------------------------------------------

    def handle(self, port: str, message: Any) -> None:
        """One MIDI message from `port` (a mido Message, or raw bytes).
        Called on the port's thread; never raises."""
        try:
            if isinstance(message, (bytes, bytearray, list, tuple)):
                import mido

                message = mido.Message.from_bytes(list(message))
            kind = message.type
        except Exception as exc:
            with self._lock:
                self.errors += 1
            log.debug("MIDI from %s: %s", port, exc)
            return

        if kind in CLOCK_TYPES:
            with self._lock:
                self.clock_received += 1
            if self.map.clock and self.beat is not None:
                self.beat.midi_message(message.bytes())
            return
        if kind not in ("note_on", "note_off", "control_change", "program_change"):
            return  # aftertouch, pitch bend, sysex...: not bound to anything

        entry = {"t": time.time(), "port": port, "message": _describe(message)}
        with self._lock:
            self.received += 1
            self.recent.append(entry)
            try:
                self._dispatch(port, message, entry)
            except (ControlError, ValueError, TypeError, KeyError, IndexError) as exc:
                self.errors += 1
                entry["error"] = str(exc) or type(exc).__name__

    def _dispatch(self, port: str, message: Any, entry: dict[str, Any]) -> None:
        channel = message.channel + 1
        kind = message.type
        if self.learning is not None and (kind == "control_change" or (kind == "note_on" and message.velocity > 0)):
            number = message.control if kind == "control_change" else message.note
            self._learn_from(port, "cc" if kind == "control_change" else "note", number, channel)
            entry["learned"] = True
            return

        if kind == "program_change":
            if self.map.program_change:
                self._program(port, channel, message.program)
            return

        if kind == "control_change":
            cc, value = message.control, message.value
            if cc in (0, 32):  # Bank Select MSB / LSB, for the next Program Change
                msb, lsb = self._bank.get((port, channel), (0, 0))
                self._bank[(port, channel)] = (value, lsb) if cc == 0 else (msb, value)
            for b in self.map.lsb_of(cc, port, channel):  # the fine half of a 14-bit pair
                self._lsb[(port, channel, b.number)] = value
                self._send(b, (self._msb.get((port, channel, b.number), 0) * 128 + value) / 16383, entry)
            for b in self.map.lookup("cc", cc, port, channel):
                if b.lsb is not None:
                    self._msb[(port, channel, cc)] = value
                    self._lsb[(port, channel, cc)] = 0  # a new coarse value: the fine one follows
                    self._send(b, value * 128 / 16383, entry)
                else:
                    self._send(b, value / 127, entry)
            return

        velocity = message.velocity if kind == "note_on" else 0  # note-on at 0 is a note-off
        for b in self.map.lookup("note", message.note, port, channel):
            self._send(b, velocity / 127, entry)

    def _send(self, binding: Binding, value: float, entry: dict[str, Any]) -> None:
        pressed = value >= 0.5
        if binding.toggle:
            if not pressed:
                return  # releases don't flip it back
            state = not self._toggled.get(binding, False)
            self._toggled[binding] = state
            out = 1.0 if state else 0.0
        elif binding.value is not None:
            if not pressed:
                return
            out = binding.value
        else:
            out = value
        entry.setdefault("sent", []).append(f"{binding.to} {round(out, 4)}")
        self.controls.apply(binding.to, [out])

    def _program(self, port: str, channel: int, program: int) -> None:
        store, runner = self.controls.store, self.controls.runner
        if store is None:
            raise ControlError("no playlist store")
        msb, lsb = self._bank.get((port, channel), (0, 0))
        bank = msb * 128 + lsb
        playlists = store.playlists()
        if bank >= len(playlists):
            raise ControlError(f"bank {bank}: there are {len(playlists)} playlists")
        playlist = playlists[bank]
        if program >= len(playlist.entries):
            raise ControlError(f"program {program}: {playlist.name!r} has {len(playlist.entries)} entries")
        loaded = runner.state.playlist
        if loaded is None or loaded[0] != playlist.id:
            runner.load_playlist(store.resolve(playlist.id, self.controls.registry))
        runner.goto(program)


def _describe(message: Any) -> str:
    ch = f"ch {message.channel + 1}"
    if message.type in ("note_on", "note_off"):
        return f"{message.type.replace('_', ' ')} {message.note} vel {message.velocity} {ch}"
    if message.type == "control_change":
        return f"cc {message.control} = {message.value} {ch}"
    if message.type == "program_change":
        return f"program {message.program} {ch}"
    return str(message)


def _port_label(port: str) -> str:
    """A stable part of a port name to scope a learned binding to: ALSA
    names end in client:port numbers that change when a device is replugged."""
    return port.split(":")[0].strip() or port
