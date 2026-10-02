"""A hue gradient that sweeps diagonally across the floor, on the beat.

Demonstrates position plus time, and beat time. Each tile's hue comes from
where it is (row + col, so the gradient runs diagonally) plus how far the
music has got: `ctx.t_beats` is the beat source's position when one is
running (Ableton Link, tap tempo) and the animation's own time at the
fallback tempo otherwise, so the sweep is locked to the music when there
is music and runs on its own when there is not. One sweep takes two bars
at Speed 1.

`df2_pi.tempo.pulse()` is 1 on every beat and dies away: the brightness
pumps with it, by Pump. `period` tells the playlist editor how long one
sweep takes at the default tempo, so a duration can land on whole cycles.
"""

import colorsys

from df2_pi.animation import Param, animation
from df2_pi.pixels import TileFrame
from df2_pi.tempo import pulse

BEATS_PER_SWEEP = 8  # two bars of 4/4 at Speed 1


@animation(
    name="Rainbow Sweep",
    description="A hue gradient that sweeps diagonally across the floor, pumping on the beat.",
    author="garth",
    format="tile",
    tags=["ambient", "colour"],
    params={
        "speed": Param(float, default=1.0, min=0.1, max=5.0, label="Speed"),
        "saturation": Param(float, default=1.0, min=0.0, max=1.0, label="Saturation", macro=1),
        "pump": Param(float, default=0.3, min=0.0, max=1.0, label="Pump", help="How far the brightness dips between beats", role="intensity"),
    },
    sync="beat",
    period=4.0,  # one sweep at Speed 1 and the default 120 BPM
    preview_hint="loop",
    energy="low",
)
def render(previous: TileFrame, ctx) -> TileFrame:
    frame = TileFrame.black(ctx.geometry)
    rows, cols = ctx.geometry.tile_rows, ctx.geometry.tile_cols
    phase = ctx.t_beats * ctx.params["speed"] / BEATS_PER_SWEEP
    level = 1.0 - ctx.params["pump"] * (1.0 - pulse(ctx.t_beats, decay=0.3))
    for r in range(rows):
        for c in range(cols):
            h = ((r + c) / (rows + cols - 2) + phase) % 1.0
            frame[r, c] = tuple(
                int(v * 255) for v in colorsys.hsv_to_rgb(h, ctx.params["saturation"], level)
            )
    return frame
