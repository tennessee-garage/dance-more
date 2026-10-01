"""Beat sync: where the music is, for every frame (#125).

    clock = BeatClock(beats_per_bar=4)
    clock.set_source(TapTempo())        # or LinkSource(quantum=4), MidiClock()
    info = clock.info(tick.deadline)    # once per frame; a BeatInfo, or None

Every source answers one question - what beat is it at time t? - as a
`Reading`: a float position on its own beat count, the tempo, and which
position on that count is a bar line, if it knows. Times are on
`time.perf_counter()`, the frame clock's timebase, so a frame asks about
the moment it will be seen: `tick.deadline`, its latch moment, plus the
`offset_ms` setting for the rest of the way to the LEDs (Row Bus, Tile
Bus, the strip), tuned by eye against the music.

    LinkSource   Ableton Link (via `aalink`), on the local network: tempo,
                 beat and bar from Resolume, TouchDesigner, Ableton, DJ
                 software. Quantum = beats per bar. Inactive with no peers -
                 Link alone just runs at its own default tempo.
    MidiClock    24 PPQN clock plus Start / Continue / Stop / Song Position.
                 Tempo and phase are a least-squares fit over the last four
                 beats of ticks, so jitter averages out. The bar is known
                 only after a Start or Song Position. Fed by `handle()`;
                 the MIDI port that feeds it is #131's.
    TapTempo     The mean interval of the last four taps; the last tap is
                 beat 1 of a bar. One tap after a pause re-phases at the
                 same tempo.

`BeatClock` turns a reading into a `BeatInfo`: the 1/2x / 1x / 2x
multiplier, `nudge(ms)` (shifts time, accumulating) and `resync()` (the
next beat is a downbeat) are applied here, so every source gets them. It
also guarantees `downbeat` fires on exactly one frame per bar, however the
frame times fall: it remembers the last bar it reported and only a later
one counts.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

from df2_pi.animation.context import BeatInfo

log = logging.getLogger(__name__)

MULTIPLIERS = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Reading:
    beat: float  # position on the source's own beat count
    tempo: float  # BPM
    bar_line: float | None  # a position on that count that is a bar line; None: not known


class BeatSource(Protocol):
    name: str

    def read(self, t: float) -> Reading | None:
        """The beat at perf-counter time `t`; None while there is no tempo."""

    def status(self) -> dict: ...

    def stop(self) -> None: ...


# ---- the model ------------------------------------------------------------------------------


class BeatClock:
    """The one beat everything sees. `info()` belongs to the render thread
    (it keeps the downbeat bookkeeping); the rest may be called from any."""

    def __init__(
        self,
        beats_per_bar: int = 4,
        multiplier: float = 1.0,
        offset_ms: float = 0.0,
        now: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._lock = threading.Lock()
        self._now = now
        self._source: BeatSource | None = None
        self.configure(beats_per_bar=beats_per_bar, multiplier=multiplier, offset_ms=offset_ms)
        self._nudge_s = 0.0
        self._resync_line: float | None = None  # in source beats; overrides the source's bar line
        self._source_line: float | None = None  # the source's bar line, last seen
        self._last_bar: int | None = None  # the bar index last reported (or first seen)
        self._bar_known = False
        self.last: BeatInfo | None = None

    @property
    def source(self) -> BeatSource | None:
        return self._source

    @property
    def bar_known(self) -> bool:
        """Whether the last `info()` knew where the bar line is."""
        return self._bar_known

    def set_source(self, source: BeatSource | None) -> None:
        with self._lock:
            self._source = source
            self._nudge_s = 0.0
            self._resync_line = None
            self._source_line = None
            self._last_bar = None

    def configure(self, *, beats_per_bar: int, multiplier: float, offset_ms: float) -> None:
        if not 1 <= int(beats_per_bar) <= 16:
            raise ValueError(f"beats_per_bar must be 1..16, got {beats_per_bar}")
        if multiplier not in MULTIPLIERS:
            raise ValueError(f"multiplier must be one of {MULTIPLIERS}, got {multiplier}")
        if not -500.0 <= float(offset_ms) <= 500.0:
            raise ValueError(f"offset_ms must be -500..500, got {offset_ms}")
        with self._lock:
            self.beats_per_bar = int(beats_per_bar)
            self.multiplier = float(multiplier)
            self.offset_ms = float(offset_ms)
            self._last_bar = None  # bar indices change scale with the multiplier or bar length

    def nudge(self, ms: float) -> None:
        """Shift the beat by `ms` - positive is later on the floor, so the
        lights land after where they were. Accumulates; reset by a source change."""
        with self._lock:
            self._nudge_s -= float(ms) / 1000.0

    def resync(self, t: float | None = None) -> None:
        """Make the next beat a downbeat."""
        t = self._now() if t is None else t
        with self._lock:
            source = self._source
            if source is None:
                return
            reading = source.read(t + self.offset_ms / 1000.0 + self._nudge_s)
            if reading is None:
                return
            m = self.multiplier
            self._resync_line = math.ceil(reading.beat * m) / m
            self._source_line = reading.bar_line
            self._last_bar = None

    def info(self, t: float) -> BeatInfo | None:
        """The beat for a frame seen at perf-counter time `t`. Call once per frame."""
        with self._lock:
            source = self._source
            reading = None if source is None else source.read(t + self.offset_ms / 1000.0 + self._nudge_s)
            if reading is None:
                self._last_bar = None
                self._bar_known = False
                self.last = None
                return None
            if reading.bar_line != self._source_line:
                # the source moved its bar line (a Start, a tap): it outranks an older resync
                self._source_line = reading.bar_line
                self._resync_line = None
                self._last_bar = None
            line = self._resync_line if self._resync_line is not None else reading.bar_line
            m, per_bar = self.multiplier, self.beats_per_bar
            position = reading.beat * m
            beat = math.floor(position)
            downbeat = False
            if line is None:
                bar_phase = (position % per_bar) / per_bar
                self._bar_known = False
                self._last_bar = None
            else:
                bars = (position - line * m) / per_bar
                bar = math.floor(bars)
                bar_phase = bars - bar
                if self._last_bar is not None and bar > self._last_bar:
                    downbeat = True
                if self._last_bar is None or bar > self._last_bar:
                    self._last_bar = bar
                self._bar_known = True
            self.last = BeatInfo(
                tempo=reading.tempo * m,
                phase=position - beat,
                beat=beat,
                bar_phase=bar_phase,
                beats_per_bar=per_bar,
                downbeat=downbeat,
            )
            return self.last


# ---- tap tempo ------------------------------------------------------------------------------


class TapTempo:
    """Tempo from taps: the mean interval of the last four, the last tap a
    bar line. A tap more than `RESET_S` after the previous one starts a new
    run: alone it re-phases at the tempo already set."""

    name = "tap"
    RESET_S = 2.0
    TAPS = 4

    def __init__(self, now: Callable[[], float] = time.perf_counter) -> None:
        self._now = now
        self._lock = threading.Lock()
        self._taps: deque[float] = deque(maxlen=self.TAPS)
        self._tempo: float | None = None
        self._anchor_t: float | None = None  # the last tap
        self._anchor_beat = 0.0  # its position on the count

    def tap(self, t: float | None = None) -> None:
        t = self._now() if t is None else t
        with self._lock:
            if self._taps and t - self._taps[-1] > self.RESET_S:
                self._taps.clear()
            if self._taps and t <= self._taps[-1]:
                return  # a bounce, or out of order
            self._taps.append(t)
            if len(self._taps) >= 2:
                taps = list(self._taps)
                self._tempo = 60.0 * (len(taps) - 1) / (taps[-1] - taps[0])
            if self._anchor_t is not None and self._tempo is not None:
                predicted = self._anchor_beat + (t - self._anchor_t) * self._tempo / 60.0
                self._anchor_beat = max(self._anchor_beat + 1, round(predicted))
            self._anchor_t = t

    def read(self, t: float) -> Reading | None:
        with self._lock:
            if self._tempo is None or self._anchor_t is None:
                return None
            beat = self._anchor_beat + (t - self._anchor_t) * self._tempo / 60.0
            return Reading(beat, self._tempo, self._anchor_beat)

    def status(self) -> dict:
        with self._lock:
            return {"tempo": self._tempo, "taps": len(self._taps)}

    def stop(self) -> None:
        pass


# ---- MIDI clock -----------------------------------------------------------------------------

CLOCK, START, CONTINUE, STOP, SONG_POSITION = 0xF8, 0xFA, 0xFB, 0xFC, 0xF2
PPQN = 24


class MidiClock:
    """MIDI beat clock. `handle()` takes each realtime / Song Position
    message as it arrives; `read()` fits a line through the recent ticks."""

    name = "midi"
    WINDOW = 4 * PPQN  # ticks in the fit: four beats
    MIN_TICKS = PPQN // 2  # half a beat before there is a tempo
    TIMEOUT_S = 0.5  # no ticks for this long: the clock has gone

    def __init__(self, now: Callable[[], float] = time.perf_counter) -> None:
        self._now = now
        self._lock = threading.Lock()
        self._ticks: deque[tuple[float, int]] = deque(maxlen=self.WINDOW)  # (time, tick count)
        self._count = 0  # the count the next tick will have
        self._bar_line: float | None = None
        self._playing = False
        self._resume: int | None = None  # where Continue picks up: the count at Stop, or a Song Position
        self.received = 0

    def handle(self, message: bytes | bytearray | list[int], t: float | None = None) -> None:
        t = self._now() if t is None else t
        if not message:
            return
        status = message[0]
        with self._lock:
            self.received += 1
            if status == CLOCK:
                # Counted while stopped too: the tempo fit needs every tick. The song
                # position is what Stop froze (`_resume`), restored by Continue.
                self._ticks.append((t, self._count))
                self._count += 1
            elif status == START:
                self._recount(0)
                self._bar_line = 0.0
                self._playing = True
                self._resume = None
            elif status == CONTINUE:
                if self._resume is not None:
                    self._recount(self._resume)
                    self._bar_line = 0.0
                    self._resume = None
                self._playing = True
            elif status == STOP:
                if self._playing:
                    self._resume = self._count if self._bar_line is not None else None
                self._playing = False
                self._bar_line = None  # ticks may keep coming, but the song is not moving
            elif status == SONG_POSITION and len(message) >= 3:
                sixteenths = (message[1] & 0x7F) | ((message[2] & 0x7F) << 7)
                position = sixteenths * PPQN // 4
                if self._playing:
                    self._recount(position)
                    self._bar_line = 0.0
                else:
                    self._resume = position

    def _recount(self, next_count: int) -> None:
        """The next tick has count `next_count`: shift the history to match,
        so the tempo fit carries on through a Start or a jump."""
        shift = next_count - self._count
        self._ticks = deque(((t, c + shift) for t, c in self._ticks), maxlen=self.WINDOW)
        self._count = next_count

    def read(self, t: float) -> Reading | None:
        with self._lock:
            if len(self._ticks) < self.MIN_TICKS or self._now() - self._ticks[-1][0] > self.TIMEOUT_S:
                return None
            times = [p[0] for p in self._ticks]
            counts = [p[1] for p in self._ticks]
            line = self._bar_line
        n = len(times)
        mean_t, mean_c = sum(times) / n, sum(counts) / n
        var = sum((x - mean_t) ** 2 for x in times)
        if var <= 0:
            return None
        slope = sum((x - mean_t) * (c - mean_c) for x, c in zip(times, counts)) / var  # ticks per second
        if slope <= 0:
            return None
        count = mean_c + slope * (t - mean_t)
        return Reading(count / PPQN, slope * 60.0 / PPQN, line)

    def status(self) -> dict:
        reading = self.read(self._now())
        return {"tempo": None if reading is None else reading.tempo, "playing": self._playing, "received": self.received}

    def stop(self) -> None:
        pass


# ---- Ableton Link ---------------------------------------------------------------------------


class LinkUnavailable(RuntimeError):
    """aalink is not installed, or Link could not start."""


class LinkSource:
    """Ableton Link through `aalink`. Link wants an asyncio loop, so it
    lives on a small thread of its own; reading it from the render thread
    is just a session-state capture. Its beat is read for now and carried
    forward to the frame's time at the session tempo."""

    name = "link"
    START_TIMEOUT_S = 5.0

    def __init__(self, quantum: int = 4, now: Callable[[], float] = time.perf_counter, link: object | None = None) -> None:
        self._now = now
        self._quantum = quantum
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._link = link  # injectable for tests: anything with beat, tempo, num_peers, quantum
        if link is None:
            self._start()

    def _start(self) -> None:
        try:
            import aalink  # noqa: F401  (imported in the thread too; fail here, loudly)
        except ImportError as exc:
            raise LinkUnavailable("aalink is not installed: pip install -e '.[beat]'") from exc
        ready = threading.Event()

        def run() -> None:
            async def main() -> None:
                import aalink

                self._loop = asyncio.get_running_loop()
                self._stopped = asyncio.Event()
                try:
                    link = aalink.Link(120)
                    link.quantum = self._quantum
                    link.enabled = True
                    self._link = link
                except Exception as exc:  # pragma: no cover - the C++ side failing to start
                    self._error = f"Link failed to start: {exc}"
                    ready.set()
                    return
                ready.set()
                await self._stopped.wait()
                link.enabled = False

            asyncio.run(main())

        self._thread = threading.Thread(target=run, name="ableton-link", daemon=True)
        self._thread.start()
        if not ready.wait(self.START_TIMEOUT_S) or self._error:
            raise LinkUnavailable(self._error or "Link did not start")

    def set_quantum(self, quantum: int) -> None:
        self._quantum = quantum
        if self._link is not None:
            self._link.quantum = quantum

    def read(self, t: float) -> Reading | None:
        link = self._link
        if link is None or link.num_peers == 0:
            return None
        beat, tempo, now = link.beat, link.tempo, self._now()
        # Link's beat count is quantum-aligned: beat 0 mod quantum is a bar line
        return Reading(beat + (t - now) * tempo / 60.0, tempo, 0.0)

    def status(self) -> dict:
        link = self._link
        if link is None:
            return {"tempo": None, "peers": 0}
        return {"tempo": link.tempo, "peers": link.num_peers}

    def stop(self) -> None:
        if self._loop is not None and self._stopped is not None:
            self._loop.call_soon_threadsafe(self._stopped.set)
        if self._thread is not None:
            self._thread.join(2.0)
