"""Switchyard: Comet Train, until the corners start throwing switches.

It opens as Comet Train (comet_train.py): a comet on every edge running
one way, nose to tail, pulsing a few LEDs on each beat and now and then all
turning a corner together. After a while (Interval) the turns stop being
all at once. Instead one corner throws a switch: the stream through it
turns there, for good, and the floor re-routes round it so that no two
streams ever cross. Say everything runs right and the switch at the
middle corner turns up:

- Below the switch, nothing changes.
- Above and to the left, each line turns up one corner before the line
  below it, so as not to cross that line's upward stream: a diagonal of
  turns running up and left from the switch.
- Above and to the right, the lines have lost their supply. Each is fed
  from the top: a stream coming down a column and turning right one corner
  before the line below it - a diagonal running up and right. The edge just
  after the switch is left empty, between the stream going up and the
  first one coming down.

Every Interval another switch is thrown, up to Switches of them, each
re-routing the map it lands in, so it builds into a map of currents. Then
they come out again in reverse order, one per Interval, until the floor is
uniform and it is Comet Train once more.

The map, and the comets running on it, are `df2_pi.streams`; this file is
how they move: a few LEDs on each beat.
"""

import numpy as np

from df2_pi import streams
from df2_pi.animation import Param, animation
from df2_pi.palette import palette_param
from df2_pi.pixels import PixelFrame

EASE = 3  # a step eases out: 1 - (1 - x)^EASE


@animation(
    name="Switchyard",
    description="Comet Train until the corners throw switches: one by one a stream turns for good and the floor re-routes round it into a map of currents, then the switches come out again.",
    author="df2",
    format="pixel",
    tags=["edges", "rhythm", "palette", "experimental"],
    sync="beat",
    params={
        "interval": Param(float, default=20.0, min=10.0, max=120.0, label="Interval (s)", help="How long it runs as Comet Train, and then between switches going in or coming out", macro=1),
        "switches": Param(int, default=4, min=1, max=8, label="Switches", help="How many are thrown before they start coming out", role="density"),
        "tail": Param(int, default=14, min=1, max=14, label="Tail length (LEDs)", help="Behind the head: 14 makes each comet a whole tile side, 15 LEDs", role="scale"),
        "step": Param(int, default=3, choices=[1, 3, 5, 15], label="LEDs per pulse", help="Divides a tile side, so the heads keep landing on the corners"),
        "beats": Param(float, default=1.0, choices=[0.5, 1.0, 2.0, 4.0], label="Beats per pulse"),
        "move": Param(float, default=0.3, min=0.05, max=1.0, label="Step time", help="The part of each pulse spent stepping: short slams into place, 1 never stops"),
        "turns": Param(float, default=0.3, min=0.0, max=1.0, label="Turn chance", help="While it runs as Comet Train: the chance, each time the heads reach the corners, that they all turn", role="variation"),
        "palette": palette_param(),
        "drift": Param(float, default=0.05, min=0.0, max=0.5, label="Colour drift", help="How far round the palette each new line of comets moves on"),
        "variety": Param(float, default=0.15, min=0.0, max=1.0, label="Colour variety", help="How much new comets' colours differ from each other"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    p = ctx.params
    state = ctx.state
    if not state:
        streams.start(ctx)
        state["offset"] = 0  # LEDs the heads are past the corners
        state["pulse"] = None
    n = state["lattice"].n
    state["clock"] += ctx.dt

    position = ctx.t_beats / p["beats"]
    pulse = int(np.floor(position))
    if pulse != state["pulse"]:
        if state["pulse"] is not None:
            state["offset"] = int(round(state["to"])) - (n - 1)
            if state["offset"] >= n:
                streams.commit(state)
                state["offset"] = 0
        state["pulse"] = pulse
        if state["offset"] == 0:
            streams.at_corner(ctx)
        state["from"] = n - 1 + state["offset"]
        state["to"] = 2 * n - 1 if state["segment"]["turn"] else min(state["from"] + p["step"], 2 * n - 1)

    x = position - pulse
    if state["segment"]["turn"]:
        # An even run round the corner, landing on the next beat (or the next pulse, if that comes sooner).
        progress = min(x * max(p["beats"], 1.0), 1.0)
    else:
        progress = 1.0 - (1.0 - min(x / p["move"], 1.0)) ** EASE
    head = state["from"] + (state["to"] - state["from"]) * progress
    return streams.draw(ctx, state["segment"], head)
