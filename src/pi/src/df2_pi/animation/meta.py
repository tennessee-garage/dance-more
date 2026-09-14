"""The `@animation` decorator and the metadata it attaches.

An animation file is a plain Python module with one decorated function:

    @animation(
        name="Rainbow Sweep",
        description="A hue gradient that sweeps diagonally across the floor.",
        author="garth",
        format="tile",                       # "tile" -> TileFrame, "pixel" -> PixelFrame
        tags=["ambient", "colour"],
        params={
            "speed": Param(float, default=1.0, min=0.1, max=5.0, label="Speed"),
        },
        period=12.0,                         # optional, advisory: natural loop length
        effect=Effect(FADE, (230, 0, 0, 0)), # optional: written to every tile on frame 0
        preview_hint="loop",                 # anything else is kept verbatim in meta.extra
    )
    def render(previous, ctx):
        ...

The decorator validates what it can at import time - a bad `format` or a
`Param` whose default is out of range fails when the file is loaded, with
the file named, rather than three frames into playback - and stows an
`AnimationMeta` on the function for the loader to find. It does not wrap
or otherwise change the function.

Metadata is deliberately open-ended: unknown keywords land in `extra` and
are served to the web UI as-is, so an animation can carry whatever an
interface wants to show or filter on without this module knowing about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from df2_pi.effects import Effect

FORMATS = ("tile", "pixel")

META_ATTR = "__df2_animation__"


@dataclass(frozen=True)
class Param:
    """One tunable parameter: what it is, its default, and the bounds the
    UI turns into a control. A float with min/max becomes a slider,
    `choices` becomes a select. `coerce()` is how an override from a
    playlist entry or a live edit is admitted."""

    type: type
    default: Any
    min: Any = None
    max: Any = None
    choices: tuple[Any, ...] | None = None
    label: str | None = None
    help: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, type):
            raise TypeError(f"Param type must be a type, got {self.type!r}")
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))
            if not self.choices:
                raise ValueError("Param choices must not be empty")
        # the default has to pass its own rules
        object.__setattr__(self, "default", self.coerce(self.default))

    def coerce(self, value: Any) -> Any:
        """Cast `value` to this param's type and check it against the
        bounds; raises ValueError (or TypeError) if it does not fit."""
        if self.type is bool and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                value = True
            elif lowered in ("false", "0", "no", "off"):
                value = False
            else:
                raise ValueError(f"{value!r} is not a boolean")
        elif self.type is bool and not isinstance(value, bool):
            raise TypeError(f"expected a bool, got {value!r}")
        elif isinstance(value, bool) and self.type in (int, float):
            raise TypeError(f"expected {self.type.__name__}, got {value!r}")
        try:
            value = self.type(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{value!r} is not a valid {self.type.__name__}") from exc
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{value!r} is not one of {list(self.choices)}")
        if self.min is not None and value < self.min:
            raise ValueError(f"{value!r} is below the minimum {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{value!r} is above the maximum {self.max!r}")
        return value


@dataclass(frozen=True)
class AnimationMeta:
    """What `@animation` recorded. `format` is "tile" or "pixel"; `params`
    are the declared `Param` specs; `effect` is the frame-0 effect write -
    one `Effect` for the whole floor or a `{tile: Effect}` mapping - or
    None; `period` is advisory and never read by the runner; `extra` holds
    every keyword the decorator did not recognise."""

    name: str
    description: str = ""
    author: str = ""
    format: str = "tile"
    tags: tuple[str, ...] = ()
    params: Mapping[str, Param] = field(default_factory=dict)
    period: float | None = None
    effect: Effect | Mapping[int, Effect] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("animation name must be a non-empty string")
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {self.format!r}")
        if isinstance(self.tags, str):
            raise TypeError("tags must be a list of strings, not a single string")
        object.__setattr__(self, "tags", tuple(str(t) for t in self.tags))
        params = dict(self.params)
        for key, spec in params.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise ValueError(f"param names must be identifiers, got {key!r}")
            if not isinstance(spec, Param):
                raise TypeError(f"param {key!r} must be a Param, got {spec!r}")
        object.__setattr__(self, "params", params)
        if self.period is not None and not self.period > 0:
            raise ValueError(f"period must be positive seconds, got {self.period!r}")
        if self.effect is not None and not isinstance(self.effect, Effect):
            try:
                effect = {int(tile): value for tile, value in dict(self.effect).items()}
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "effect must be an Effect or a {tile: Effect} mapping"
                ) from exc
            for tile, value in effect.items():
                if not isinstance(value, Effect):
                    raise TypeError(f"effect for tile {tile} must be an Effect, got {value!r}")
            object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "extra", dict(self.extra))

    def defaults(self) -> dict[str, Any]:
        """The declared parameter defaults, as a fresh dict."""
        return {key: spec.default for key, spec in self.params.items()}

    def resolve_params(self, *overrides: Mapping[str, Any] | None) -> dict[str, Any]:
        """Defaults, then each override layer in turn (playlist entry, then
        live UI edits) - every value coerced through its `Param`. An
        override naming a parameter this animation does not declare is an
        error, so a typo in a playlist is caught rather than ignored."""
        merged = self.defaults()
        for layer in overrides:
            if not layer:
                continue
            for key, value in layer.items():
                if key not in self.params:
                    raise ValueError(
                        f"unknown parameter {key!r} for {self.name!r}; "
                        f"declared: {sorted(self.params)}"
                    )
                merged[key] = self.params[key].coerce(value)
        return merged


def animation(
    *,
    name: str,
    description: str = "",
    author: str = "",
    format: str = "tile",
    tags=(),
    params: Mapping[str, Param] | None = None,
    period: float | None = None,
    effect: Effect | Mapping[int, Effect] | None = None,
    **extra: Any,
) -> Callable[[Callable], Callable]:
    """Mark a module's render function as its animation and attach its
    metadata. See the module docstring for the fields."""
    meta = AnimationMeta(
        name=name,
        description=description,
        author=author,
        format=format,
        tags=tags,
        params=params or {},
        period=period,
        effect=effect,
        extra=extra,
    )

    def decorate(fn: Callable) -> Callable:
        if not callable(fn):
            raise TypeError(f"@animation must decorate a function, got {fn!r}")
        setattr(fn, META_ATTR, meta)
        return fn

    return decorate
