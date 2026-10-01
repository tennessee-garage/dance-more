"""A checkerboard that flips on a beat and drifts in colour.

Demonstrates `ctx.state`: a dict that is empty on frame 0 and persists for
the rest of the run, so an animation can carry anything it likes between
frames without writing a class. The colour here random-walks on every
flip - something you cannot compute from `ctx.t` alone, which is exactly
when state earns its keep. Because the walk uses `ctx.rng`, a run is
reproducible from its seed.

It also follows the music: with a beat source running (`ctx.beat` is not
None) it flips on every beat instead of every `interval` seconds. The
`sync="beat"` in the decorator tells the UI it does.
"""

import colorsys

from df2_pi.animation import Param, animation
from df2_pi.pixels import TileFrame


@animation(
    name="Checkerboard",
    description="Alternating tiles, flipping on a beat, colours drifting.",
    author="df2",
    format="tile",
    tags=["basic", "rhythm"],
    sync="beat",
    params={
        "interval": Param(float, default=0.5, min=0.1, max=4.0, label="Seconds per flip", curve="log", macro=1),
        "drift": Param(float, default=0.05, min=0.0, max=0.5, label="Hue drift per flip", role="variation"),
    },
)
def render(previous: TileFrame, ctx) -> TileFrame:
    state = ctx.state
    if not state:  # frame 0: set up
        state["phase"] = 0
        state["next_flip"] = ctx.params["interval"]
        state["hue"] = ctx.rng.random()

    if ctx.beat is not None:  # on the beat: flip whenever the beat count moves on
        flip = state.get("beat") is not None and ctx.beat.beat != state["beat"]
        state["beat"] = ctx.beat.beat
        state["next_flip"] = ctx.t + ctx.params["interval"]  # if the beat goes, carry on from here
    else:
        state["beat"] = None
        flip = ctx.t >= state["next_flip"]
        if flip:
            state["next_flip"] += ctx.params["interval"]
    if flip:
        state["phase"] ^= 1
        state["hue"] = (state["hue"] + ctx.rng.uniform(-1, 1) * ctx.params["drift"]) % 1.0

    a = tuple(int(v * 255) for v in colorsys.hsv_to_rgb(state["hue"], 1.0, 1.0))
    b = tuple(int(v * 255) for v in colorsys.hsv_to_rgb((state["hue"] + 0.5) % 1.0, 1.0, 0.6))

    frame = TileFrame.black(ctx.geometry)
    for row in range(ctx.geometry.tile_rows):
        for col in range(ctx.geometry.tile_cols):
            frame[row, col] = a if (row + col + state["phase"]) % 2 == 0 else b
    return frame
