"""Switchyard Flow: Switchyard with the comets gliding, and the beat in their brightness.

The same map of streams as Switchyard (switchyard.py): a comet on every
edge running one way, nose to tail; all of them now and then turning a
corner together; then, after a while (Interval), switches thrown one corner
at a time until the floor is a map of currents that never cross, and taken
out again in reverse order until it is uniform once more.

What differs is the motion. The comets run continuously at Speed, round
corners and through the switches going in and out without stopping, and
the beat is in the light instead: on every pulse they flash to full
brightness and fade back to the Between pulses level.

The map, and the comets running on it, are `df2_pi.streams`; this file is
how they move.
"""

from df2_pi import streams
from df2_pi.animation import Param, animation
from df2_pi.palette import palette_param
from df2_pi.pixels import PixelFrame
from df2_pi.tempo import pulse

DECAY = 0.3  # beats: how quickly a flash fades


@animation(
    name="Switchyard Flow",
    description="Switchyard's map of currents with the comets gliding continuously, flashing on the beat.",
    author="df2",
    format="pixel",
    tags=["edges", "rhythm", "palette", "experimental"],
    sync="beat",
    params={
        "interval": Param(float, default=20.0, min=10.0, max=120.0, label="Interval (s)", help="How long it runs with every line turning together, and then between switches going in or coming out", macro=1),
        "switches": Param(int, default=4, min=1, max=8, label="Switches", help="How many are thrown before they start coming out", role="density"),
        "speed": Param(float, default=30.0, min=5.0, max=150.0, label="Speed (LEDs/s)", help="15 LEDs is a tile side", curve="log"),
        "low": Param(float, default=0.35, min=0.0, max=1.0, label="Between pulses", help="Brightness between the beats, which flash to full; 1 for no pulse", role="intensity"),
        "beats": Param(float, default=1.0, choices=[0.5, 1.0, 2.0, 4.0], label="Beats per pulse"),
        "tail": Param(int, default=14, min=1, max=14, label="Tail length (LEDs)", help="Behind the head: 14 makes each comet a whole tile side, 15 LEDs", role="scale"),
        "turns": Param(float, default=0.15, min=0.0, max=1.0, label="Turn chance", help="Before the switches: the chance, each time the heads reach the corners, that they all turn", role="variation"),
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
        state["head"] = None  # how far along the segment's paths the heads are
    n = state["lattice"].n
    state["clock"] += ctx.dt

    if state["head"] is None:
        streams.at_corner(ctx)
        state["head"] = n - 1
    else:
        state["head"] += p["speed"] * ctx.dt
        while state["head"] >= 2 * n - 1:  # at the next corners: on to the next tile side
            streams.commit(state)
            state["head"] -= n
            streams.at_corner(ctx)

    gain = p["low"] + (1.0 - p["low"]) * pulse(ctx.t_beats, rate=1.0 / p["beats"], decay=DECAY)
    return streams.draw(ctx, state["segment"], state["head"], gain)
