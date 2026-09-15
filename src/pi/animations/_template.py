"""Copy me to start a new animation.

The loader skips files whose name starts with an underscore, so this one
never plays. Copy it to `my_animation.py` (the filename is the id
playlists use), fill in the blanks, and run:

    df2-pi play --no-hardware --terminal --animation my_animation

A syntax error shows as a warning and the other animations keep loading;
fix the file and run again. See docs/animations.md for the full tour.
"""

from df2_pi.animation import Param, animation
from df2_pi.pixels import PixelFrame, TileFrame


@animation(
    name="My Animation",  # shown in the UI
    description="What it looks like, in a sentence.",
    author="you",
    # "tile":  one colour per tile, an 8x8 TileFrame - for anything floor-scale.
    #          Cheap on the wire (32 bytes per row instead of 1,448).
    # "pixel": every LED, a PixelFrame - for anything that lives on the edges.
    format="tile",
    tags=["mine"],
    params={
        # Each Param becomes a control in the web UI and `--param name=value`.
        "speed": Param(float, default=1.0, min=0.1, max=5.0, label="Speed"),
    },
    # period=4.0,          # optional: the loop length in seconds, if it has one
    # effect=Effect(...),  # optional: a tile effect written on the first frame
)
def render(previous, ctx):
    """Return a NEW frame for time `ctx.t`. `previous` is read-only.

    Useful things in `ctx`:
        ctx.t, ctx.frame, ctx.dt, ctx.fps    time
        ctx.params["speed"]                  the resolved parameters
        ctx.geometry                         the floor: edges, seams, floor_ring, ...
        ctx.state                            a dict that persists across frames ({} on frame 0)
        ctx.rng, ctx.np_rng                  seeded random sources
    """
    frame = TileFrame.black(ctx.geometry)  # or PixelFrame.black(ctx.geometry) for format="pixel"

    # --- draw something ---------------------------------------------------
    phase = (ctx.t * ctx.params["speed"]) % 1.0
    frame.data[:] = (int(255 * phase), 0, int(255 * (1 - phase)))

    # To evolve the last frame instead (trails, decay):
    #     frame = previous.copy().gain(0.9)
    #     frame.splat(x, y, (255, 255, 255), radius=3)     # PixelFrame only

    return frame
