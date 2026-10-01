import math
import random

import pytest

from df2_pi.interfacing.beat import (
    CLOCK,
    CONTINUE,
    PPQN,
    SONG_POSITION,
    START,
    STOP,
    BeatClock,
    LinkSource,
    MidiClock,
    Reading,
    TapTempo,
)


class Now:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Steady:
    """A source at a fixed tempo, beat 0 (a bar line) at `t0`."""

    name = "steady"

    def __init__(self, tempo: float, t0: float = 0.0, bar_line: float | None = 0.0) -> None:
        self.tempo, self.t0, self.bar_line = tempo, t0, bar_line

    def read(self, t: float) -> Reading:
        return Reading((t - self.t0) * self.tempo / 60.0, self.tempo, self.bar_line)

    def status(self) -> dict:
        return {}

    def stop(self) -> None:
        pass


def frames(clock: BeatClock, start: float, seconds: float, fps: float = 30.0, jitter: float = 0.0, seed: int = 0):
    rng = random.Random(seed)
    n = int(seconds * fps)
    return [clock.info(start + i / fps + rng.uniform(-jitter, jitter)) for i in range(n)]


# ---- BeatClock --------------------------------------------------------------------------------


def test_no_source_means_no_beat():
    assert BeatClock().info(5.0) is None


def test_downbeat_fires_on_exactly_one_frame_per_bar():
    clock = BeatClock(beats_per_bar=4)
    clock.set_source(Steady(tempo=128.0))
    infos = frames(clock, start=0.01, seconds=60.0, jitter=0.004, seed=3)
    downbeats = [i for i, info in enumerate(infos) if info.downbeat]
    bars_crossed = math.floor(infos[-1].beat / 4) - math.floor(infos[0].beat / 4)
    assert len(downbeats) == bars_crossed == 31
    for i in downbeats:  # on the first frame past the line
        assert infos[i].bar_phase < 128 / 60 / 4 / 30 * 1.5
        assert infos[i - 1].bar_phase > 0.9


def test_beat_phase_and_bar_phase():
    clock = BeatClock(beats_per_bar=4)
    clock.set_source(Steady(tempo=120.0))
    info = clock.info(2.25)  # 4.5 beats in
    assert (info.beat, info.phase, info.tempo, info.beats_per_bar) == (4, pytest.approx(0.5), 120.0, 4)
    assert info.bar_phase == pytest.approx(0.125)


def test_the_multiplier_scales_tempo_and_count():
    clock = BeatClock(multiplier=2.0)
    clock.set_source(Steady(tempo=120.0))
    info = clock.info(1.25)  # 2.5 beats in
    assert (info.tempo, info.beat, info.phase) == (240.0, 5, pytest.approx(0.0))
    clock.configure(beats_per_bar=4, multiplier=0.5, offset_ms=0.0)
    info = clock.info(1.25)
    assert (info.tempo, info.beat, info.phase) == (60.0, 1, pytest.approx(0.25))


def test_offset_reads_the_music_later_than_the_frame():
    clock = BeatClock(offset_ms=100.0)
    clock.set_source(Steady(tempo=120.0))
    assert clock.info(1.0).phase == pytest.approx(0.2)  # 2.2 beats: the floor shows it 100 ms late


def test_nudge_accumulates_and_a_source_change_clears_it():
    clock = BeatClock()
    clock.set_source(Steady(tempo=120.0))
    clock.nudge(50)
    clock.nudge(50)
    assert clock.info(1.5).phase == pytest.approx(0.8)  # 3.0 beats, 100 ms later on the floor
    clock.set_source(Steady(tempo=120.0))
    assert clock.info(1.5).phase == pytest.approx(0.0)


def test_resync_makes_the_next_beat_a_downbeat():
    now = Now(1.1)  # 2.2 beats in: beat 3 is not a bar line until resync says so
    clock = BeatClock(beats_per_bar=4, now=now)
    clock.set_source(Steady(tempo=120.0))
    clock.info(1.1)
    clock.resync()
    infos = [clock.info(1.1 + i * 0.05) for i in range(1, 120)]  # to beat 14
    downbeat_beats = [info.beat for info in infos if info.downbeat]
    assert downbeat_beats == [3, 7, 11]


def test_without_a_bar_line_there_is_never_a_downbeat_until_resync():
    clock = BeatClock()
    clock.set_source(Steady(tempo=120.0, bar_line=None))
    assert not any(info.downbeat for info in frames(clock, 0.0, 10.0))
    assert not clock.bar_known
    clock.resync(10.0)
    assert any(info.downbeat for info in frames(clock, 10.0, 3.0))


def test_bad_settings_are_refused():
    clock = BeatClock()
    for kwargs in ({"beats_per_bar": 0}, {"multiplier": 3.0}, {"offset_ms": 900}):
        with pytest.raises(ValueError):
            clock.configure(**{"beats_per_bar": 4, "multiplier": 1.0, "offset_ms": 0.0, **kwargs})


# ---- tap tempo --------------------------------------------------------------------------------


def test_tap_tempo_is_the_mean_of_the_last_four_taps_and_the_last_is_beat_one():
    tap = TapTempo(now=Now())
    for t in (10.0, 10.6, 11.1, 11.6, 12.1):  # the first interval is dropped: last four taps only
        tap.tap(t)
    reading = tap.read(12.1)
    assert reading.tempo == pytest.approx(120.0)
    assert reading.bar_line == reading.beat == 4
    assert tap.read(12.35).beat == pytest.approx(4.5)


def test_one_tap_is_not_a_tempo_and_a_lone_tap_after_a_pause_rephases():
    tap = TapTempo(now=Now())
    tap.tap(1.0)
    assert tap.read(1.2) is None
    tap.tap(1.5)
    tap.tap(2.0)
    assert tap.read(2.0).tempo == pytest.approx(120.0)
    tap.tap(5.3)  # after a pause: same tempo, new downbeat here
    reading = tap.read(5.3)
    assert reading.tempo == pytest.approx(120.0) and reading.bar_line == reading.beat


def test_tapped_downbeats_land_on_the_last_tap():
    clock = BeatClock(beats_per_bar=4)
    tap = TapTempo(now=Now())
    clock.set_source(tap)
    for t in (0.0, 0.5, 1.0, 1.5):
        tap.tap(t)
    clock.info(1.49)
    infos = [clock.info(1.5 + i / 30) for i in range(1, 200)]
    downbeat_times = [1.5 + (i + 1) / 30 for i, info in enumerate(infos) if info.downbeat]
    assert downbeat_times == pytest.approx([1.5, 3.5, 5.5, 7.5], abs=1 / 30 + 1e-9)  # the last tap was beat 1


# ---- MIDI clock -------------------------------------------------------------------------------


def clock_ticks(midi: MidiClock, now: Now, tempo: float, beats: float, start: float, jitter: float = 0.0, seed: int = 0) -> float:
    rng = random.Random(seed)
    period = 60.0 / tempo / PPQN
    n = int(beats * PPQN)
    for i in range(n):
        now.t = start + i * period + rng.uniform(-jitter, jitter)
        midi.handle([CLOCK], now.t)
    return start + n * period  # when the next tick is due


def test_midi_clock_tempo_and_phase_hold_through_jitter():
    now = Now()
    midi = MidiClock(now=now)
    midi.handle([START], now.t)
    nominal = 60.0 / 125.0 / PPQN
    clock_ticks(midi, now, 125.0, beats=8, start=1000.0, jitter=0.002, seed=7)  # +-2 ms on 20 ms ticks
    t = 1000.0 + 8 * PPQN * nominal - nominal  # the last tick's nominal time
    reading = midi.read(t)
    assert reading.tempo == pytest.approx(125.0, abs=0.3)
    assert reading.beat == pytest.approx(8 - 1 / PPQN, abs=0.02)
    assert reading.bar_line == 0.0


def test_midi_clock_has_no_bar_until_start_or_song_position():
    now = Now()
    midi = MidiClock(now=now)
    clock_ticks(midi, now, 120.0, beats=4, start=1000.0)
    assert midi.read(now.t).bar_line is None
    clock = BeatClock()
    clock.set_source(midi)
    assert clock.info(now.t).downbeat is False and not clock.bar_known


def test_start_makes_the_next_tick_beat_zero_and_a_bar_line():
    now = Now()
    midi = MidiClock(now=now)
    nxt = clock_ticks(midi, now, 120.0, beats=2, start=1000.0)  # free-running before Start
    midi.handle([START], nxt - 0.001)
    nxt = clock_ticks(midi, now, 120.0, beats=2, start=nxt)
    reading = midi.read(nxt)
    assert reading.beat == pytest.approx(2.0, abs=1e-6) and reading.bar_line == 0.0


def test_song_position_then_continue_resumes_there():
    now = Now()
    midi = MidiClock(now=now)
    midi.handle([START], now.t)
    nxt = clock_ticks(midi, now, 120.0, beats=2, start=1000.0)
    midi.handle([STOP], nxt)
    assert midi.read(nxt).bar_line is None
    nxt = clock_ticks(midi, now, 120.0, beats=1, start=nxt)  # clock keeps running while stopped
    midi.handle([SONG_POSITION, 32, 0], nxt)  # 32 sixteenths = beat 8
    midi.handle([CONTINUE], nxt)
    nxt = clock_ticks(midi, now, 120.0, beats=1, start=nxt)
    reading = midi.read(nxt)
    assert reading.beat == pytest.approx(9.0, abs=1e-6) and reading.bar_line == 0.0


def test_continue_after_stop_picks_up_where_it_stopped():
    now = Now()
    midi = MidiClock(now=now)
    midi.handle([START], now.t)
    nxt = clock_ticks(midi, now, 120.0, beats=3, start=1000.0)
    midi.handle([STOP], nxt)
    nxt = clock_ticks(midi, now, 120.0, beats=2, start=nxt)  # stopped: the song does not move
    midi.handle([CONTINUE], nxt)
    nxt = clock_ticks(midi, now, 120.0, beats=1, start=nxt)
    assert midi.read(nxt).beat == pytest.approx(4.0, abs=1e-6)


def test_midi_clock_goes_quiet_after_the_timeout():
    now = Now()
    midi = MidiClock(now=now)
    clock_ticks(midi, now, 120.0, beats=2, start=1000.0)
    assert midi.read(now.t) is not None
    now.t += MidiClock.TIMEOUT_S + 0.01
    assert midi.read(now.t) is None


# ---- Link -------------------------------------------------------------------------------------


class FakeLink:
    """What LinkSource reads of aalink.Link, on a fake clock."""

    def __init__(self, now: Now, tempo: float = 124.0) -> None:
        self.now, self.tempo, self.num_peers, self.quantum = now, tempo, 1, 4

    @property
    def beat(self) -> float:
        return (self.now.t - 1000.0) * self.tempo / 60.0


def test_link_reads_ahead_to_the_frame_time_and_its_bars_are_quantum_aligned():
    now = Now(1010.0)
    link = LinkSource(now=now, link=FakeLink(now))
    reading = link.read(1010.5)
    assert reading.beat == pytest.approx(10.5 * 124 / 60) and reading.tempo == 124.0 and reading.bar_line == 0.0


def test_link_without_peers_is_no_beat():
    now = Now(1010.0)
    fake = FakeLink(now)
    fake.num_peers = 0
    assert LinkSource(now=now, link=fake).read(now.t) is None


def test_the_floor_follows_a_link_session():
    now = Now(1000.0)
    fake = FakeLink(now, tempo=124.0)
    clock = BeatClock(beats_per_bar=4)
    clock.set_source(LinkSource(now=now, link=fake))
    downbeats = []
    for i in range(30 * 20):
        now.t = 1000.0 + i / 30
        info = clock.info(now.t + 0.02)  # the frame is seen 20 ms from now
        if info.downbeat:
            downbeats.append(info.beat)
        if i == 300:
            fake.tempo = 128.0  # the session changes tempo; Link re-anchors, so do we
    assert all(b % 4 == 0 for b in downbeats) and len(downbeats) >= 9
