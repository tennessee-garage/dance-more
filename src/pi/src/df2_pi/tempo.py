"""Tempo-locked waveforms: pure functions of beat time (#126).

    from df2_pi.tempo import lfo, pulse

    level = lfo(ctx.t_beats, rate=0.5)              # once every two beats, 0..1
    flash = pulse(ctx.t_beats, decay=0.2)           # 1 on every beat, dying away
    wave = lfo(ctx.t_beats - phase_map, rate=0.25)  # arrays work: one value per tile or LED

`t_beats` is `ctx.t_beats`: the beat source's position when there is one
(`ctx.beat.beat + ctx.beat.phase`), otherwise the animation's own time at
the fallback tempo, so anything written against it runs sensibly either
way. Everything here returns 0..1 and takes scalars or numpy arrays
(returning a float, or an array when any input is one), so it composes with per-tile and
per-LED phase offsets.

`rate` is cycles per beat: 1 is every beat, 0.25 once a bar in 4/4, 2
twice a beat. `offset` shifts the cycle, in cycles.

Shapes, over one cycle starting on the beat:

    sine       0 on the beat, 1 halfway, back to 0 - a smooth swell
    tri        the same as straight lines
    saw        rising 0 -> 1, dropping back on the beat
    ramp_down  1 on the beat, falling to 0 - the same accent as `pulse`, linear
    square     1 for the first half of the cycle, 0 for the second
"""

from __future__ import annotations

import numpy as np

SHAPES = ("sine", "tri", "saw", "square", "ramp_down")


def _out(value: np.ndarray) -> float | np.ndarray:
    """A float when every input was a number; an array when any was an array."""
    return float(value) if np.ndim(value) == 0 else value


def lfo(t_beats, rate: float = 1.0, shape: str = "sine", offset: float = 0.0):
    """A low-frequency oscillator locked to the beat, 0..1."""
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    phase = np.mod(np.asarray(t_beats, dtype=np.float64) * rate + offset, 1.0)
    if shape == "sine":
        value = 0.5 - 0.5 * np.cos(2 * np.pi * phase)
    elif shape == "tri":
        value = 1.0 - np.abs(2.0 * phase - 1.0)
    elif shape == "saw":
        value = phase
    elif shape == "ramp_down":
        value = 1.0 - phase
    else:
        value = (phase < 0.5).astype(np.float64)
    return _out(value)


def pulse(t_beats, rate: float = 1.0, decay: float = 0.25):
    """1 on the beat (every 1/rate beats), decaying exponentially with a
    time constant of `decay` beats - an accent that dies away."""
    if decay <= 0:
        raise ValueError(f"decay must be positive, got {decay}")
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    since = np.mod(np.asarray(t_beats, dtype=np.float64) * rate, 1.0) / rate  # beats since the last pulse
    return _out(np.exp(-since / decay))
