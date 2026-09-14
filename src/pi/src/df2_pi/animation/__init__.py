"""Animation authoring and loading - see meta.py for the file format.

    from df2_pi.animation import animation, Param, Effect

    @animation(name="...", format="tile", params={...})
    def render(previous, ctx):
        ...
"""

from df2_pi.animation.context import BeatInfo, FrameContext
from df2_pi.animation.loader import (
    AnimationDef,
    AnimationError,
    AnimationRun,
    LoadError,
    Rendered,
    load_animation_file,
)
from df2_pi.animation.meta import FORMATS, AnimationMeta, Param, animation
from df2_pi.animation.registry import AnimationRegistry, default_animations_dir
from df2_pi.effects import FADE, HUE_SPLIT, NONE, Effect

__all__ = [
    "AnimationDef",
    "AnimationError",
    "AnimationMeta",
    "AnimationRegistry",
    "AnimationRun",
    "BeatInfo",
    "Effect",
    "FADE",
    "FORMATS",
    "FrameContext",
    "HUE_SPLIT",
    "LoadError",
    "NONE",
    "Param",
    "Rendered",
    "animation",
    "default_animations_dir",
    "load_animation_file",
]
