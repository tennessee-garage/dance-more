"""Video: a clip of real footage played on the floor - waves, fire, clouds.

Clips come from `df2-pi video import` (see `df2_pi/video.py`): a video
averaged down to the floor's cell grid and kept at its 3,840 LEDs, so
playback here is only blending stored frames. Big, soft movement reads
best - surf washing up a beach shot from above, flames, drifting cloud -
since the picture is seen through the lattice of tile edges.

- Speed plays it slower or faster; frames are blended in linear light, so
  slowed right down it still moves smoothly rather than stepping.
- Loop fade crossfades the end of the clip into its start, so the seam
  doesn't show.
- Contrast, Black level and Brightness shape it for LEDs. The floor shows
  black poorly, so dark footage usually wants a little black level.
- Rotation turns the picture to bring the waves in from the right side.

The clip list is read when the animation loads: after importing a new
one, reload animations (the Animations tab) for it to appear. With no
clips at all it shows a dim, slow blue wash.
"""

import math

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.gamma import DECODE_LUT, from_linear
from df2_pi.pixels import PixelFrame
from df2_pi.video import list_clips, load_clip

CLIPS = list_clips()
NO_CLIP = "(none)"


@animation(
    name="Video",
    description="A clip of real footage - surf, fire, cloud - imported with `df2-pi video import`.",
    author="df2",
    format="pixel",
    tags=["video", "ambient"],
    params={
        "clip": Param(str, default=(CLIPS or [NO_CLIP])[0], choices=CLIPS or [NO_CLIP], label="Clip", macro=1),
        "speed": Param(float, default=1.0, min=0.1, max=4.0, label="Speed", curve="log"),
        "loop_fade": Param(float, default=1.5, min=0.0, max=6.0, label="Loop fade (s)", help="Crossfade the end into the start"),
        "contrast": Param(float, default=1.0, min=0.5, max=2.0, label="Contrast"),
        "black": Param(float, default=0.05, min=0.0, max=0.4, label="Black level", help="Lifts the darkest parts: the floor shows black poorly"),
        "intensity": Param(float, default=1.0, min=0.1, max=1.0, label="Brightness"),
        "rotation": Param(str, default="0", choices=["0", "90", "180", "270"], label="Rotation"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    p = ctx.params
    clip = load_clip(p["clip"]) if p["clip"] != NO_CLIP else None
    if clip is None:
        return _placeholder(ctx)

    frames, n = clip.frames, len(clip.frames)
    fade = min(int(p["loop_fade"] * clip.fps), n // 3)
    loop = n - fade  # frames in one pass; the last `fade` are only ever crossfaded in
    position = (ctx.state.get("position", 0.0) + ctx.dt * p["speed"] * clip.fps) % loop
    ctx.state["position"] = position

    light = _sample(frames, position)
    if position < fade:  # the head of the clip, fading in over its own tail
        w = position / fade
        light = _sample(frames, position + loop) * (1.0 - w) + light * w

    shade = from_linear(light).astype(np.float32) / 255.0
    shade = np.clip((shade - 0.5) * p["contrast"] + 0.5, 0.0, 1.0)
    shade = (p["black"] + (1.0 - p["black"]) * shade) * p["intensity"]
    frame = PixelFrame(np.rint(shade * 255.0).astype(np.uint8), ctx.geometry)
    turns = int(p["rotation"]) // 90
    return frame.rotated(turns) if turns else frame


def _sample(frames: np.ndarray, position: float) -> np.ndarray:
    """Linear light at a fractional frame position: the two frames either side, blended."""
    i = int(position)
    j = min(i + 1, len(frames) - 1)
    f = position - i
    return DECODE_LUT[frames[i]] * (1.0 - f) + DECODE_LUT[frames[j]] * f


def _placeholder(ctx) -> PixelFrame:
    """No clip: a dim blue wash breathing slowly, so the floor isn't dark."""
    level = 0.12 + 0.06 * math.sin(ctx.t * 0.6)
    frame = PixelFrame.black(ctx.geometry)
    frame.data[...] = np.array([0.25, 0.45, 1.0]) * level * 255
    return frame
