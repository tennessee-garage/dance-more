"""Loading one animation file, and running the animation it defines.

    definition = load_animation_file(path)      # AnimationDef, None, or raises LoadError
    run = definition.start(geometry)            # one playing instance
    rendered = run.render()                     # advances a frame
    rendered.frame, rendered.effects

`load_animation_file` imports the file in its own module namespace and
finds the one `@animation`-decorated function in it. Anything that goes
wrong - a SyntaxError, a failing import, a module body that raises, two
decorators, a render with the wrong signature - is raised
as a `LoadError` carrying the file, the stage it failed at and the
traceback, so the registry can record it against that one file and keep
loading the rest of the directory.

`AnimationRun` is the piece that upholds the contract from the other side:
it freezes the previous frame before handing it over, checks the animation
returned a NEW frame of the declared format (naming the file when it did
not), collects effect writes, and keeps the per-run state, clock and rng.
The playlist runner drives one of these per playing entry.

Animations are trusted local code. There is no sandbox.
"""

from __future__ import annotations

import importlib.util
import inspect
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from df2_pi.animation.context import BeatInfo, FrameContext
from df2_pi.animation.meta import META_ATTR, AnimationMeta
from df2_pi.effects import Effect
from df2_pi.geometry import FloorGeometry
from df2_pi.pixels import Frame, PixelFrame, TileFrame, check_ownership, default_geometry

MODULE_PREFIX = "df2_animations."

FRAME_TYPES: dict[str, type[Frame]] = {"tile": TileFrame, "pixel": PixelFrame}


class LoadError(Exception):
    """An animation file could not be loaded. `stage` is "syntax",
    "import" (the module body raised) or "validate" (it loaded but is not
    a usable animation); `traceback` is the formatted traceback when there
    is one."""

    def __init__(self, path: Path, stage: str, message: str, tb: str = "") -> None:
        super().__init__(f"{path.name}: {message}")
        self.path = path
        self.id = path.stem
        self.stage = stage
        self.message = message
        self.traceback = tb


class AnimationError(RuntimeError):
    """An animation misbehaved at render time - wrong frame type, returned
    the previous frame, or raised. Names the file."""


@dataclass(frozen=True)
class AnimationDef:
    """A loaded animation: its stable id (the filename stem, which is what
    playlists reference), the render function, and its metadata."""

    id: str
    path: Path
    meta: AnimationMeta
    render: Callable[[Frame, FrameContext], Frame]

    @property
    def frame_type(self) -> type[Frame]:
        return FRAME_TYPES[self.meta.format]

    def start(
        self,
        geometry: FloorGeometry | None = None,
        params: Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
        fps: float = 30.0,
    ) -> AnimationRun:
        """Begin a fresh run: frame 0 next, empty state, rng seeded with
        `seed` (or a random one - see `AnimationRun.seed`)."""
        return AnimationRun(self, geometry, params, seed=seed, fps=fps)


@dataclass(frozen=True)
class Rendered:
    """One frame out of a run, plus the effect writes made during it."""

    frame: Frame
    effects: dict[int, Effect]


class AnimationRun:
    """One playing instance of an animation.

    Owns what persists between frames - the previous frame, the frame
    counter, `state`, the rngs - and builds the `FrameContext` for each
    call. `render(t)` advances one frame; `t` is seconds since the run
    started as measured by whoever owns the clock, defaulting to
    `frame / fps` for callers without one (tests, recorders).

    `set_params()` swaps in a new resolved parameter dict between frames -
    how live UI edits reach a running animation.
    """

    def __init__(
        self,
        definition: AnimationDef,
        geometry: FloorGeometry | None = None,
        params: Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
        fps: float = 30.0,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.definition = definition
        self.geometry = geometry if geometry is not None else default_geometry()
        self.params = definition.meta.resolve_params(params)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        # Recorded so a recording can be reproduced: pass it back as `seed`.
        self.seed = seed if seed is not None else random.SystemRandom().getrandbits(32)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.state: dict[str, Any] = {}
        self.frame = 0
        self.previous: Frame = definition.frame_type.black(self.geometry).freeze()

    def set_params(self, params: Mapping[str, Any]) -> None:
        self.params = self.definition.meta.resolve_params(params)

    def render(self, t: float | None = None, beat: BeatInfo | None = None) -> Rendered:
        """Render the next frame. Raises `AnimationError` if the animation
        returns the wrong format, hands back `previous`, or raises."""
        definition = self.definition
        ctx = FrameContext(
            frame=self.frame,
            t=self.frame * self.dt if t is None else float(t),
            dt=self.dt,
            fps=self.fps,
            params=self.params,
            geometry=self.geometry,
            state=self.state,
            rng=self.rng,
            np_rng=self.np_rng,
            beat=beat,
        )
        if self.frame == 0 and definition.meta.effect is not None:
            effect = definition.meta.effect
            if isinstance(effect, Effect):
                ctx.send_effect_all(effect)
            else:
                for tile, value in effect.items():
                    ctx.send_effect(tile, value)

        previous = self.previous
        try:
            frame = definition.render(previous, ctx)
        except Exception as exc:
            raise AnimationError(
                f"{definition.path.name}: render() raised on frame {self.frame}: {exc!r}"
            ) from exc

        if not isinstance(frame, definition.frame_type):
            raise AnimationError(
                f"{definition.path.name}: render() returned "
                f"{type(frame).__name__}, but format={definition.meta.format!r} "
                f"declares {definition.frame_type.__name__}"
            )
        try:
            check_ownership(previous, frame)
        except ValueError as exc:
            raise AnimationError(f"{definition.path.name}: {exc}") from exc

        self.previous = frame.freeze()
        self.frame += 1
        return Rendered(frame, ctx.effects)


# ---- loading a file ---------------------------------------------------------------


def load_animation_file(path: Path) -> AnimationDef | None:
    """Import `path` and return the animation it defines, or None if the
    module has no `@animation` at all (a helper module, not an error).
    Raises `LoadError` for anything that stops it being a usable animation
    - the caller decides whether that is fatal (it is not, for the
    registry)."""
    path = Path(path)
    stem = path.stem
    if not stem.isidentifier():
        raise LoadError(
            path, "validate", f"filename {path.name!r} is not a valid animation id"
        )

    module_name = MODULE_PREFIX + stem
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build an import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        # Registered so dataclasses, pickling and friends can find the
        # module by name; a reload simply replaces the entry.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    except SyntaxError as exc:
        raise LoadError(path, "syntax", _syntax_message(exc), traceback.format_exc()) from exc
    except Exception as exc:
        raise LoadError(path, "import", f"{type(exc).__name__}: {exc}", traceback.format_exc()) from exc

    decorated = [
        (name, obj)
        for name, obj in vars(module).items()
        if callable(obj) and isinstance(getattr(obj, META_ATTR, None), AnimationMeta)
        and getattr(obj, "__module__", None) == module_name
    ]
    if not decorated:
        return None
    if len(decorated) > 1:
        names = ", ".join(name for name, _ in decorated)
        raise LoadError(path, "validate", f"more than one @animation in one file: {names}")

    name, fn = decorated[0]
    meta: AnimationMeta = getattr(fn, META_ATTR)
    _check_signature(path, name, fn)
    return AnimationDef(id=stem, path=path, meta=meta, render=fn)


def _check_signature(path: Path, name: str, fn: Callable) -> None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise LoadError(path, "validate", f"{name}() has no inspectable signature") from exc
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = [p for p in positional if p.default is p.empty]
    takes_varargs = any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values())
    if len(required) > 2 or (len(positional) < 2 and not takes_varargs):
        raise LoadError(
            path,
            "validate",
            f"{name}{sig} must accept (previous, ctx)",
        )


def _syntax_message(exc: SyntaxError) -> str:
    where = f"line {exc.lineno}" if exc.lineno else "unknown line"
    return f"SyntaxError at {where}: {exc.msg}"
