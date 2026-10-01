"""The floor as a lighting-desk fixture: a 17-channel control block.

    control = DmxControl(runner, external_source, store)
    control.handle(channels, now)     # every DMX packet for the control universe
    control.poll()                    # periodically: releases after silence

A desk (grandMA, Chamsys, QLC+) or a media server's DMX output patches the
floor as a fixture at a start address and drives it like any other:

    ch  function                     maps to
    1   master dimmer                Runner.set_brightness
    2   strobe rate, 0 = off         Runner.set_strobe, 1..255 -> up to strobe_max_hz
    3   source                       0-84 internal, 85-169 external, 170-255 mix (#123)
    4   internal/external mix        ExternalSource.set_mix
    5   bank                         playlist N, by name order (0-based)
    6   program                      entry N of that playlist (0-based)
    7   speed                        0 stop .. 128 = 1x .. 255 = 4x
    8-11 macro 1-4                   Runner.set_control("macroN")
    12-14 tint red, green, blue      Runner.set_tint
    15  tint amount                  Runner.set_tint
    16  bump                         a flash of value/255 when it rises
    17  hold                         Runner.hold: 0-127 the countdown runs, 128-255 it is held

(`CHANNELS` is the table; the QLC+ fixture in docs/fixtures/ is generated
from it, so the two cannot drift.)

Change detection. DMX resends every channel continuously; a control acts
only when its value changes. Continuous controls (dimmer, strobe, source,
mix, speed, tint) also act on the first packet after silence - a fixture
follows its faders from the moment it is patched. Triggers (bank/program,
macros, bump) never do: a desk that connects with them at 0 must not
reload the playlist or flash the floor. Hold is between the two: it acts
on a first packet only to hold, so a desk that connects with it down
does not release a hold set from the web UI, and after that whenever it
crosses 128.

Release. When the control universe has been silent for the timeout, every
continuous control goes back to its non-DMX value: brightness to the
stored setting (what the web UI last set), strobe off, speed 1, tint off,
source and mix to their stored settings, and a hold the desk was
applying is released, so a dropped desk cannot leave the show stuck on
one entry. What bank/program loaded, and macro values, stay.

Off by default. A media server driving tile mode usually sends the whole
512-channel universe with the unused channels at 0; a control block in
that universe would read master dimmer 0. The operator turns this on.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from df2_pi.engine.runner import Runner
    from df2_pi.interfacing.external import ExternalSource
    from df2_pi.playlists import PlaylistStore

log = logging.getLogger(__name__)

POLL_INTERVAL_S = 0.1


@dataclass(frozen=True)
class Channel:
    name: str
    group: str  # QLC+ channel group
    help: str
    trigger: bool = False  # acts on change only, never on the first packet
    colour: str | None = None


CHANNELS: tuple[Channel, ...] = (
    Channel("Master dimmer", "Intensity", "0 dark .. 255 full"),
    Channel("Strobe", "Shutter", "0 off, 1..255 up to the strobe cap"),
    Channel("Source", "Maintenance", "0-84 internal, 85-169 external, 170-255 mix"),
    Channel("External mix", "Intensity", "External over the playlist, when source is mix"),
    Channel("Bank", "Effect", "Playlist N in name order, from 0", trigger=True),
    Channel("Program", "Effect", "Entry N of the playlist, from 0", trigger=True),
    Channel("Speed", "Speed", "0 stop, 128 = 1x, 255 = 4x"),
    Channel("Macro 1", "Effect", "The playing animation's macro 1", trigger=True),
    Channel("Macro 2", "Effect", "The playing animation's macro 2", trigger=True),
    Channel("Macro 3", "Effect", "The playing animation's macro 3", trigger=True),
    Channel("Macro 4", "Effect", "The playing animation's macro 4", trigger=True),
    Channel("Tint red", "Intensity", "Tint colour", colour="Red"),
    Channel("Tint green", "Intensity", "Tint colour", colour="Green"),
    Channel("Tint blue", "Intensity", "Tint colour", colour="Blue"),
    Channel("Tint amount", "Intensity", "0 off .. 255 fully the tint"),
    Channel("Bump", "Intensity", "A flash of value/255 each time it rises", trigger=True),
    Channel("Hold", "Maintenance", "0-127 the countdown runs, 128-255 the playing entry is held"),
)
WIDTH = len(CHANNELS)
DIMMER, STROBE, SOURCE, MIX, BANK, PROGRAM, SPEED, MACRO1 = 0, 1, 2, 3, 4, 5, 6, 7
TINT = (11, 12, 13, 14)
BUMP = 15
HOLD = 16
HOLD_THRESHOLD = 128
BUMP_DECAY_S = 0.25


def speed_of(value: int) -> float:
    """0 stops, 128 is 1x, 255 is 4x - linear on each side of 128."""
    return value / 128 if value <= 128 else 1.0 + (value - 128) / 127 * 3.0


def source_of(value: int) -> str:
    return "internal" if value < 85 else "external" if value < 170 else "mix"


class DmxControl:
    def __init__(
        self,
        runner: Runner,
        source: ExternalSource | None,
        store: PlaylistStore | None,
        *,
        timeout_s: float = 2.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runner = runner
        self.source = source
        self.store = store
        self.timeout_s = timeout_s
        self._now = now
        self._lock = threading.Lock()
        self._last: bytes | None = None  # the block as last acted on; None while released
        self._last_at: float | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.packets = 0

    # ---- input ----------------------------------------------------------------------------

    def handle(self, block: bytes, now: float | None = None) -> None:
        """One packet's worth of the control block (WIDTH channels, from the
        start address). Called on the receiver's thread."""
        block = bytes(block[:WIDTH]).ljust(WIDTH, b"\x00")
        with self._lock:
            self._last_at = self._now() if now is None else now
            self.packets += 1
            previous, self._last = self._last, block
        self._apply(previous, block)

    def poll(self) -> None:
        """Release after `timeout_s` of silence."""
        with self._lock:
            if self._last is None or self._last_at is None or self._now() - self._last_at < self.timeout_s:
                return
            last, self._last = self._last, None
        log.info("DMX control released after %.1f s of silence", self.timeout_s)
        self._release(last)

    def live(self) -> bool:
        return self._last is not None

    def status(self) -> dict:
        block = self._last
        return {
            "live": block is not None,
            "packets": self.packets,
            "values": None if block is None else list(block),
            "channels": [c.name for c in CHANNELS],
        }

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="dmx-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(2.0)

    def _run(self) -> None:
        while self._running:
            try:
                self.poll()
            except Exception:
                log.exception("DMX control poll failed")
            time.sleep(POLL_INTERVAL_S)

    # ---- acting -----------------------------------------------------------------------------

    def _apply(self, previous: bytes | None, block: bytes) -> None:
        def changed(i: int) -> bool:
            if previous is None:
                return not CHANNELS[i].trigger
            return previous[i] != block[i]

        runner = self.runner
        if changed(DIMMER):
            runner.set_brightness(block[DIMMER])
        if changed(STROBE):
            cap = runner.state.show.strobe_max_hz
            runner.set_strobe(block[STROBE] / 255 * cap)
        if self.source is not None:
            if changed(SOURCE):
                self.source.set_source(source_of(block[SOURCE]))
            if changed(MIX):
                self.source.set_mix(block[MIX] / 255)
        if changed(SPEED):
            runner.set_speed(speed_of(block[SPEED]))
        for n in range(4):
            if changed(MACRO1 + n):
                runner.set_control(f"macro{n + 1}", block[MACRO1 + n] / 255)
        if any(changed(i) for i in TINT):
            r, g, b, amount = (block[i] for i in TINT)
            runner.set_tint(r, g, b, amount / 255)
        if previous is not None and block[BUMP] > previous[BUMP]:
            runner.bump(block[BUMP] / 255, BUMP_DECAY_S)
        held = block[HOLD] >= HOLD_THRESHOLD
        if (held and previous is None) or (previous is not None and held != (previous[HOLD] >= HOLD_THRESHOLD)):
            runner.hold(held)
        if changed(BANK) or changed(PROGRAM):
            self._program(block[BANK], block[PROGRAM], bank_changed=previous is None or previous[BANK] != block[BANK])

    def _program(self, bank: int, program: int, *, bank_changed: bool) -> None:
        if self.store is None:
            return
        playlists = self.store.playlists()
        if bank >= len(playlists):
            log.info("DMX bank %d: there are only %d playlists", bank, len(playlists))
            return
        playlist = playlists[bank]
        if program >= len(playlist.entries):
            log.info("DMX program %d: playlist %r has %d entries", program, playlist.name, len(playlist.entries))
            return
        loaded = self.runner.state.playlist
        if bank_changed or loaded is None or loaded[0] != playlist.id:
            self.runner.load_playlist(self.store.resolve(playlist.id))
        self.runner.goto(program)

    def _release(self, last: bytes) -> None:
        runner = self.runner
        brightness = self.store.get_int("brightness") if self.store is not None else 255
        runner.set_brightness(255 if brightness is None else brightness)
        runner.set_strobe(0.0)
        runner.set_speed(1.0)
        runner.set_tint(255, 255, 255, 0.0)
        if last[HOLD] >= HOLD_THRESHOLD:
            runner.hold(False)
        if self.source is not None and self.store is not None:
            self.source.set_source(self.store.get_str("external_source") or "external")
            self.source.set_mix(self.store.get_float("external_mix") or 0.0)


# ---- the QLC+ fixture definition ------------------------------------------------------------


def qlc_fixture() -> str:
    """A QLC+ fixture definition (.qxf) for the control block, from `CHANNELS`."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE FixtureDefinition>",
        '<FixtureDefinition xmlns="http://www.qlcplus.org/FixtureDefinition">',
        " <Creator>",
        "  <Name>Q Light Controller Plus</Name>",
        "  <Version>4.12.0</Version>",
        "  <Author>df2-pi (generated from interfacing/dmx_control.py)</Author>",
        " </Creator>",
        " <Manufacturer>Tennessee Garage</Manufacturer>",
        " <Model>Dance Floor v2</Model>",
        " <Type>Other</Type>",
    ]
    for ch in CHANNELS:
        lines.append(f' <Channel Name="{escape(ch.name)}">')
        lines.append(f'  <Group Byte="0">{ch.group}</Group>')
        if ch.colour:
            lines.append(f"  <Colour>{ch.colour}</Colour>")
        lines.extend(_capabilities(ch))
        lines.append(" </Channel>")
    lines.append(f' <Mode Name="Control {WIDTH}ch">')
    lines.extend(f'  <Channel Number="{i}">{escape(ch.name)}</Channel>' for i, ch in enumerate(CHANNELS))
    lines.append(" </Mode>")
    lines.extend(
        [
            " <Physical>",
            '  <Bulb Type="LED" Lumens="0" ColourTemperature="0"/>',
            '  <Dimensions Weight="0" Width="3048" Height="3048" Depth="100"/>',
            '  <Lens Name="Other" DegreesMin="0" DegreesMax="0"/>',
            '  <Focus Type="Fixed" PanMax="0" TiltMax="0"/>',
            '  <Technical PowerConsumption="0" DmxConnector="Other"/>',
            " </Physical>",
            "</FixtureDefinition>",
            "",
        ]
    )
    return "\n".join(lines)


def _capabilities(ch: Channel) -> list[str]:
    if ch.name == "Strobe":
        return ['  <Capability Min="0" Max="0">Off</Capability>', '  <Capability Min="1" Max="255">Strobe slow to fast</Capability>']
    if ch.name == "Source":
        return [
            '  <Capability Min="0" Max="84">Internal (playlist only)</Capability>',
            '  <Capability Min="85" Max="169">External (Art-Net takes over)</Capability>',
            '  <Capability Min="170" Max="255">Mix (Art-Net over the playlist)</Capability>',
        ]
    if ch.name == "Hold":
        return [
            '  <Capability Min="0" Max="127">Run (the playlist advances)</Capability>',
            '  <Capability Min="128" Max="255">Hold (the playing entry plays on)</Capability>',
        ]
    if ch.name == "Speed":
        return [
            '  <Capability Min="0" Max="127">Slower (0 stops)</Capability>',
            '  <Capability Min="128" Max="128">Normal</Capability>',
            '  <Capability Min="129" Max="255">Faster (up to 4x)</Capability>',
        ]
    return [f'  <Capability Min="0" Max="255">{escape(ch.help)}</Capability>']
