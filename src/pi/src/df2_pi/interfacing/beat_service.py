"""`BeatService`: the beat source and its settings, applied live and stored.

    beat = BeatService(runner, store)     # attaches its BeatClock to the runner
    beat.start()                          # the stored source starts (Link joins the session)
    beat.update(source="link", beats_per_bar=4, launch_quantum="bar")
    beat.tap(); beat.resync(); beat.nudge(10)
    beat.status()
    beat.stop()

Settings live in the `setting` table under `KEYS`, like the external
input's, and `update()` validates the whole change before applying any of
it. The MIDI clock source exists (`self.midi`, fed through `midi_message()`)
but is not offered as a setting until #131 gives the floor a MIDI input to
feed it from.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from df2_pi.engine.runner import LAUNCH_QUANTA
from df2_pi.interfacing.beat import MULTIPLIERS, BeatClock, LinkSource, LinkUnavailable, MidiClock, TapTempo

if TYPE_CHECKING:
    from df2_pi.engine.runner import Runner
    from df2_pi.playlists import PlaylistStore

log = logging.getLogger(__name__)

SOURCES = ("off", "link", "tap")  # "midi" joins with #131's MIDI input

# update() field -> setting key
KEYS = {
    "source": "beat_source",
    "beats_per_bar": "beats_per_bar",
    "multiplier": "beat_multiplier",
    "offset_ms": "beat_offset_ms",
    "launch_quantum": "launch_quantum",
}


class BeatService:
    def __init__(
        self,
        runner: Runner,
        store: PlaylistStore | None,
        *,
        now: Callable[[], float] = time.perf_counter,
        link_factory: Callable[..., LinkSource] = LinkSource,
    ) -> None:
        self.runner = runner
        self.store = store
        self._now = now
        self._link_factory = link_factory
        self._lock = threading.Lock()
        self._settings = self._load()
        s = self._settings
        self.clock = BeatClock(s["beats_per_bar"], s["multiplier"], s["offset_ms"], now=now)
        self.tap_source = TapTempo(now=now)
        self.midi = MidiClock(now=now)
        self._link: LinkSource | None = None
        self._error: str | None = None
        runner.attach_beat(self.clock)
        runner.set_launch_quantum(s["launch_quantum"])

    def start(self) -> None:
        self._activate(self._settings["source"])

    def stop(self) -> None:
        self.clock.set_source(None)
        self._stop_link()

    # ---- live controls ----------------------------------------------------------------

    def tap(self) -> None:
        self.tap_source.tap()

    def resync(self) -> None:
        self.clock.resync()

    def nudge(self, ms: float) -> None:
        self.clock.nudge(ms)

    def midi_message(self, message: bytes | list[int], t: float | None = None) -> None:
        """A MIDI realtime / Song Position message, for #131's input to call."""
        self.midi.handle(message, t)

    # ---- settings ---------------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        return dict(self._settings)

    def update(self, **changes: Any) -> dict[str, Any]:
        """Validate, apply now, and store. Raises ValueError, with nothing
        changed, if any value is bad."""
        unknown = set(changes) - set(KEYS)
        if unknown:
            raise ValueError(f"unknown beat settings {sorted(unknown)}")
        merged = {**self._settings, **changes}
        _validate(merged)
        with self._lock:
            if {"beats_per_bar", "multiplier", "offset_ms"} & set(changes):
                self.clock.configure(beats_per_bar=merged["beats_per_bar"], multiplier=merged["multiplier"], offset_ms=merged["offset_ms"])
                if self._link is not None:
                    self._link.set_quantum(merged["beats_per_bar"])
            if "launch_quantum" in changes:
                self.runner.set_launch_quantum(merged["launch_quantum"])
            self._settings = merged
            if "source" in changes:
                self._activate(merged["source"])
        if self.store is not None:
            for field in changes:
                self.store.set_setting(KEYS[field], merged[field])
        return self.settings()

    def _activate(self, source: str) -> None:
        self._error = None
        if source != "link":
            self._stop_link()
        if source == "off":
            self.clock.set_source(None)
        elif source == "tap":
            self.clock.set_source(self.tap_source)
        elif source == "midi":
            self.clock.set_source(self.midi)
        elif source == "link":
            if self._link is None:
                try:
                    self._link = self._link_factory(quantum=self._settings["beats_per_bar"], now=self._now)
                except LinkUnavailable as exc:
                    log.warning("Ableton Link: %s", exc)
                    self._error = str(exc)
                    self.clock.set_source(None)
                    return
            self.clock.set_source(self._link)

    def _stop_link(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            link.stop()

    def _load(self) -> dict[str, Any]:
        """The stored settings, falling back to the defaults for any value
        that no longer validates, so a bad row never stops the floor."""
        from df2_pi.playlists.store import DEFAULT_SETTINGS

        values = {field: DEFAULT_SETTINGS[key] for field, key in KEYS.items()}
        if self.store is None:
            return values
        readers = {int: self.store.get_int, float: self.store.get_float, str: self.store.get_str}
        for field, key in KEYS.items():
            stored = readers[type(DEFAULT_SETTINGS[key])](key)
            candidate = {**values, field: stored}
            try:
                _validate(candidate)
                values = candidate
            except (TypeError, ValueError) as exc:
                log.warning("setting %r: %s; using %r", key, exc, values[field])
        return values

    # ---- status -----------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        source = self.clock.source
        info = self.clock.last
        detail = source.status() if source is not None else {}
        return {
            "active": info is not None,
            "source": self._settings["source"],
            "error": self._error,
            "tempo": info.tempo if info is not None else detail.get("tempo"),
            "beat": None if info is None else info.beat,
            "phase": None if info is None else info.phase,
            "bar_phase": None if info is None else info.bar_phase,
            "bar_known": self.clock.bar_known,
            "peers": detail.get("peers"),
            "taps": self.tap_source.status()["taps"],
        }


def _validate(s: dict[str, Any]) -> None:
    if s["source"] not in SOURCES and s["source"] != "midi":
        raise ValueError(f"source must be one of {SOURCES}")
    if not isinstance(s["beats_per_bar"], int) or isinstance(s["beats_per_bar"], bool) or not 1 <= s["beats_per_bar"] <= 16:
        raise ValueError("beats_per_bar must be 1..16")
    if float(s["multiplier"]) not in MULTIPLIERS:
        raise ValueError(f"multiplier must be one of {MULTIPLIERS}")
    if not -500.0 <= float(s["offset_ms"]) <= 500.0:
        raise ValueError("offset_ms must be -500..500")
    if s["launch_quantum"] not in LAUNCH_QUANTA:
        raise ValueError(f"launch_quantum must be one of {LAUNCH_QUANTA}")
