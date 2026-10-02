"""OSC control surface (#132): the floor's controls as OSC addresses, with
state fed back to anyone who subscribes.

    osc = OscControl(runner, registry, store=store, beat=beat, palettes=palettes, external=external)
    osc.begin()           # if the osc_enabled setting is on: a UDP server on osc_port (7000)
    osc.handle("/floor/palette/ocean", 1.0)   # what the server calls, for each message
    osc.configure(enabled=True, port=9000)    # applied now, and stored
    osc.stop()

TouchDesigner (OSC Out CHOP / DAT), TouchOSC-style phone apps and Resolume all
send OSC. Resolume lets you choose the outgoing address of anything it
outputs, so a clip can send `/floor/play/waves` when it launches and a layer's
opacity can send `/floor/mix`: everything a sender only ever triggers has a
form with its target in the ADDRESS (`/floor/palette/ocean`), since the
sender picks the value.

Values. Floats are 0..1 unless noted and map the way the DMX control block
maps a fader (speed: 0 stop, 0.5 normal, 1 four times; source in thirds).
Ints are accepted wherever a float is. On/off addresses take any number,
on at 0.5 and up. PRESS addresses (next, the path forms, tap...) fire on a
message with no value, or on a value of 0.5 or more that is either a rise
from below 0.5 or more than 0.3 s after the last one - so a button that
sends 1 then 0 fires once, a sender that only ever sends 1 fires on every
press, and one that repeats 1 every frame while held fires once.

    /floor/brightness f        /floor/blackout on        /floor/speed f
    /floor/strobe f            /floor/bump [f]           /floor/tint r g b amount
    /floor/hue f (turns)       /floor/saturation f (0.5 normal)
    /floor/freeze on           /floor/hold on            /floor/reset
    /floor/source f|s          /floor/mix f
    /floor/next  /floor/previous  /floor/restart
    /floor/goto i              /floor/goto/<entry>
    /floor/playlist i|s        /floor/playlist/<name>
    /floor/play s              /floor/play/<animation>
    /floor/palette i|s         /floor/palette/<name>
    /floor/param/<name> f      /floor/macro/<n> f
    /floor/trigger i [f]       /floor/trigger/<slot> [f]
    /floor/tempo/tap  /floor/tempo/resync  /floor/tempo/nudge ms
    /floor/subscribe port      (state comes back to the sender's address, that port)

Feedback: a subscriber gets every value once on subscribing and then each
change, polled at 10 Hz: /floor/state/{animation s, playlist s, entry i,
palette s, brightness f, blackout i, held i, tempo f, beat i}. A
subscription lapses after 60 s without a renewal.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from df2_pi.animation import AnimationRegistry
    from df2_pi.engine.runner import Runner
    from df2_pi.interfacing.beat_service import BeatService
    from df2_pi.interfacing.service import ExternalInput
    from df2_pi.palette import PaletteBook
    from df2_pi.playlists import PlaylistStore

log = logging.getLogger(__name__)

DEFAULT_PORT = 7000
SUBSCRIPTION_S = 60.0
FEEDBACK_INTERVAL_S = 0.1
RECENT = 20  # messages kept for the web UI's monitor
BUMP_DECAY_S = 0.25
REPRESS_S = 0.3  # a high value this long after the last one is a new press, even with no release between


class OscError(ValueError):
    """A message the floor understood the address of but not the values."""


def speed_from_unit(u: float) -> float:
    """0 stops, 0.5 is normal, 1 is four times - the DMX speed fader's curve."""
    return 2.0 * u if u <= 0.5 else 1.0 + (u - 0.5) * 6.0


def source_from_unit(u: float) -> str:
    return "internal" if u < 1 / 3 else "external" if u < 2 / 3 else "mix"


class OscControl:
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
        send: Callable[[tuple[str, int], str, Any], None] | None = None,
    ) -> None:
        self.runner = runner
        self.registry = registry
        self.store = store
        self.beat = beat
        self.palettes = palettes
        self.external = external
        self._now = now
        self._send = send or _udp_sender()
        self._lock = threading.Lock()
        self._levels: dict[str, tuple[float, float]] = {}  # press addresses: (last value, when)
        self._subscribers: dict[tuple[str, int], float] = {}  # (host, port) -> expiry
        self._sent: dict[tuple[str, int], dict[str, Any]] = {}  # what each subscriber last got
        self.received = 0
        self.errors = 0
        self.recent: deque[dict[str, Any]] = deque(maxlen=RECENT)
        self.port: int | None = None  # where it is listening; None when it isn't
        self.error: str | None = None
        self.enabled = store.get_bool("osc_enabled") if store is not None else True
        self.configured_port = store.get_int("osc_port") if store is not None else DEFAULT_PORT
        if self.enabled is None:
            self.enabled = True
        if self.configured_port is None or not 1 <= self.configured_port <= 65535:
            self.configured_port = DEFAULT_PORT
        self._server = None
        self._threads: list[threading.Thread] = []
        self._running = False

    # ---- the server ---------------------------------------------------------------------------

    def begin(self) -> None:
        """Start listening if the setting says so."""
        if self.enabled:
            self.start(self.configured_port)

    def settings(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "port": self.configured_port}

    def configure(self, *, enabled: bool | None = None, port: int | None = None) -> dict[str, Any]:
        """Change and store the settings, restarting the server to match."""
        if port is not None and not 1 <= int(port) <= 65535:
            raise ValueError("port must be 1..65535")
        if enabled is not None:
            self.enabled = bool(enabled)
        if port is not None:
            self.configured_port = int(port)
        if self.store is not None:
            self.store.set_setting("osc_enabled", self.enabled)
            self.store.set_setting("osc_port", self.configured_port)
        self.stop()
        self.begin()
        return self.settings()

    def start(self, port: int = DEFAULT_PORT, bind: str = "0.0.0.0") -> None:
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer

        dispatcher = Dispatcher()
        dispatcher.set_default_handler(lambda client, address, *args: self.handle(address, *args, sender=client), needs_reply_address=True)
        try:
            self._server = ThreadingOSCUDPServer((bind, port), dispatcher)
        except OSError as exc:
            self.error = f"could not listen on UDP {port}: {exc}"
            log.warning("OSC: %s", self.error)
            return
        self.port = self._server.server_address[1]
        self.error = None
        self._running = True
        self._threads = [
            threading.Thread(target=self._server.serve_forever, name="osc-server", daemon=True),
            threading.Thread(target=self._feedback_loop, name="osc-feedback", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        log.info("OSC: listening on UDP %d", self.port)

    def stop(self) -> None:
        self._running = False
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        for thread in self._threads:
            thread.join(2.0)
        self._threads = []
        self.port = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            return {
                "listening": self.port is not None,
                "port": self.port,
                "error": self.error,
                "received": self.received,
                "errors": self.errors,
                "subscribers": [f"{host}:{port}" for (host, port), expires in self._subscribers.items() if expires > now],
                "recent": list(reversed(self.recent)),
            }

    # ---- handling -----------------------------------------------------------------------------

    def handle(self, address: str, *args: Any, sender: tuple[str, int] | None = None) -> None:
        """Act on one message. Never raises: a bad message is logged,
        counted and shown in the monitor."""
        entry = {"t": time.time(), "from": sender[0] if sender else None, "address": address, "args": [_plain(a) for a in args]}
        with self._lock:
            self.received += 1
            self.recent.append(entry)
        try:
            self._route(address, list(args), sender)
        except (OscError, ValueError, TypeError, KeyError, IndexError) as exc:
            with self._lock:
                self.errors += 1
            entry["error"] = str(exc) or type(exc).__name__
            log.debug("OSC %s %s: %s", address, args, exc)

    def _route(self, address: str, args: list[Any], sender: tuple[str, int] | None) -> None:
        parts = [p for p in address.split("/") if p]
        if not parts or parts[0] != "floor":
            raise OscError("not a /floor address")
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
                raise OscError("tint takes r g b amount")
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
                raise OscError("external input is not running")
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
                raise OscError("beat sync is not running")
            if rest[0] == "tap" and self._pressed(address, args):
                self.beat.tap()
            elif rest[0] == "resync" and self._pressed(address, args):
                self.beat.resync()
            elif rest[0] == "nudge":
                self.beat.nudge(_number(args[0]))
        elif head == "subscribe":
            if sender is None:
                raise OscError("subscribe needs the sender's address")
            self.subscribe(sender[0], int(_number(args[0])) if args else sender[1])
        else:
            raise OscError(f"no such address {address}")

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
            raise OscError("no playlist store")
        key = int(which) if isinstance(which, (int, float)) or (isinstance(which, str) and which.isdigit()) else str(which)
        self.runner.load_playlist(self.store.resolve(key, self.registry))

    def _play(self, animation_id: str) -> None:
        if self.registry.get(animation_id) is None:
            raise OscError(f"no animation {animation_id!r}")
        self.runner.play_animation(animation_id)

    def _palette(self, which: Any) -> None:
        if self.palettes is None:
            raise OscError("no palettes")
        if isinstance(which, (int, float)) or (isinstance(which, str) and which.isdigit()):
            names = self.palettes.names()
            index = int(which)
            if not 0 <= index < len(names):
                raise OscError(f"palette {index}: there are {len(names)}")
            which = names[index]
        self.palettes.activate(str(which))

    def _param(self, name: str, u: float) -> None:
        playing = self.runner.state.animation
        definition = self.registry.get(playing[0]) if playing else None
        if definition is None:
            raise OscError("nothing with parameters is playing")
        spec = definition.meta.params.get(name)
        if spec is None:
            raise OscError(f"{definition.id} has no parameter {name!r}")
        self.runner.set_params(**{name: spec.from_unit(u)})

    # ---- feedback -----------------------------------------------------------------------------

    def subscribe(self, host: str, port: int) -> None:
        with self._lock:
            self._subscribers[(host, port)] = self._now() + SUBSCRIPTION_S
            self._sent.pop((host, port), None)  # a (re)subscription gets everything again
        self.push()

    def state(self) -> dict[str, Any]:
        s = self.runner.state
        beat = s.beat
        return {
            "animation": s.animation[1] if s.animation else "",
            "playlist": s.playlist[1] if s.playlist else "",
            "entry": -1 if s.entry_index is None else s.entry_index,
            "palette": s.palette or "",
            "brightness": round(s.brightness / 255, 4),
            "blackout": int(s.blacked_out),
            "held": int(s.timer_held),
            "tempo": round(beat.tempo, 3) if beat else 0.0,
            "beat": beat.beat if beat else -1,
        }

    def push(self) -> None:
        """Send each live subscriber whatever has changed since it last heard."""
        now = self._now()
        with self._lock:
            for key in [k for k, expires in self._subscribers.items() if expires <= now]:
                del self._subscribers[key]
                self._sent.pop(key, None)
            subscribers = list(self._subscribers)
        if not subscribers:
            return
        state = self.state()
        for target in subscribers:
            last = self._sent.setdefault(target, {})
            for key, value in state.items():
                if last.get(key) != value:
                    try:
                        self._send(target, f"/floor/state/{key}", value)
                        last[key] = value
                    except OSError as exc:
                        log.debug("OSC feedback to %s: %s", target, exc)

    def _feedback_loop(self) -> None:
        while self._running:
            try:
                self.push()
            except Exception:
                log.exception("OSC feedback failed")
            time.sleep(FEEDBACK_INTERVAL_S)


# ---- values ---------------------------------------------------------------------------------------


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise OscError(f"expected a number, got {value!r}")


def _clamp(u: float) -> float:
    return min(1.0, max(0.0, u))


def _unit(args: list[Any]) -> float:
    if not args:
        raise OscError("expected a value")
    return _clamp(_number(args[0]))


def _on(args: list[Any]) -> bool:
    return True if not args else _number(args[0]) >= 0.5


def _plain(value: Any) -> Any:
    return value if isinstance(value, (int, float, str, bool)) or value is None else repr(value)


def _udp_sender() -> Callable[[tuple[str, int], str, Any], None]:
    clients: dict[tuple[str, int], Any] = {}

    def send(target: tuple[str, int], address: str, value: Any) -> None:
        from pythonosc.udp_client import SimpleUDPClient

        client = clients.get(target)
        if client is None:
            client = clients[target] = SimpleUDPClient(*target)
        client.send_message(address, value)

    return send
