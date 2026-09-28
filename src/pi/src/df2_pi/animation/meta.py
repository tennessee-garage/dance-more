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
        sync="beat",                         # optional: follows ctx.beat when present
        triggers=True,                       # optional: reacts to ctx.triggers
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

External control. A MIDI knob, a DMX channel or an OSC fader delivers a
value with no idea what is playing, so parameters carry a shared
vocabulary that lets one control do something sensible across the whole
library:

- `role`: one of `ROLES` (speed, intensity, density, scale, hue,
  variation). A binding targets the role; the running animation's param
  with that role receives it, and an animation without one ignores it. A
  param whose NAME is a role has that role unless it declares another, so
  `"speed": Param(...)` needs nothing extra. Roles carry meaning, not
  units - the value is mapped across the param's own min..max.
- `macro`: 1..`MACROS`. "Macro N" reaches whichever param the animation
  puts there - its most interesting control, role or not.
- `curve`: "linear" or "log", how 0..1 maps onto min..max.
  `Param.from_unit()` is the one place a normalised external value
  becomes a param value; `to_unit()` goes back, for feedback and for
  sliders that should agree with the knob.

Each role and each macro is claimed by at most one param per animation,
and only a param `from_unit()` can map (numbers with bounds, choices,
bools) may claim one - both checked at import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from df2_pi.effects import Effect

FORMATS = ("tile", "pixel")

META_ATTR = "__df2_animation__"

ROLES = ("speed", "intensity", "density", "scale", "hue", "variation")
MACROS = 4
CURVES = ("linear", "log")
SYNCS = ("beat",)


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
    role: str | None = None
    macro: int | None = None
    curve: str = "linear"

    def __post_init__(self) -> None:
        if not isinstance(self.type, type):
            raise TypeError(f"Param type must be a type, got {self.type!r}")
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))
            if not self.choices:
                raise ValueError("Param choices must not be empty")
        # the default has to pass its own rules
        object.__setattr__(self, "default", self.coerce(self.default))
        if self.role is not None and self.role not in ROLES:
            raise ValueError(f"Param role must be one of {ROLES}, got {self.role!r}")
        if self.macro is not None and (
            isinstance(self.macro, bool) or not isinstance(self.macro, int) or not 1 <= self.macro <= MACROS
        ):
            raise ValueError(f"Param macro must be 1..{MACROS}, got {self.macro!r}")
        if self.curve not in CURVES:
            raise ValueError(f"Param curve must be one of {CURVES}, got {self.curve!r}")
        if self.curve == "log" and not (self._ranged and self.min > 0):
            raise ValueError("a log curve needs a numeric Param with min > 0 and a max")
        if (self.role is not None or self.macro is not None) and not self.mappable:
            raise ValueError("a role or macro needs a Param from_unit() can map: numbers with min and max, choices, or bool")

    @property
    def _ranged(self) -> bool:
        return (
            self.choices is None
            and self.type in (int, float)
            and self.min is not None
            and self.max is not None
        )

    @property
    def mappable(self) -> bool:
        """Whether `from_unit()` can map onto this param: numbers with both
        bounds, choices, or bool."""
        return self.choices is not None or self.type is bool or self._ranged

    def from_unit(self, u: float) -> Any:
        """The value at `u` along this param's range, 0.0..1.0 (clamped):
        min..max along `curve` for bounded numbers (ints rounded), one equal
        slot per choice, `u >= 0.5` for a bool. What every external control
        (MIDI CC, DMX, OSC) normalises to and calls. Raises ValueError for a
        param with nothing to map onto."""
        u = min(1.0, max(0.0, float(u)))
        if self.choices is not None:
            return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]
        if self.type is bool:
            return u >= 0.5
        if not self._ranged:
            raise ValueError("this Param has no range for from_unit() to map onto")
        if self.curve == "log":
            value = self.min * (self.max / self.min) ** u
        else:
            value = self.min + u * (self.max - self.min)
        if self.type is int:
            value = round(value)
        return self.coerce(min(self.max, max(self.min, value)))

    def to_unit(self, value: Any) -> float:
        """Where `value` sits along this param's range, 0.0..1.0 - the
        inverse of `from_unit()` (a choice maps to its slot's centre)."""
        if self.choices is not None:
            return (self.choices.index(self.coerce(value)) + 0.5) / len(self.choices)
        if self.type is bool:
            return 1.0 if self.coerce(value) else 0.0
        if not self._ranged:
            raise ValueError("this Param has no range for to_unit() to map from")
        value = min(self.max, max(self.min, float(value)))
        if self.max == self.min:
            return 0.0
        if self.curve == "log":
            return math.log(value / self.min) / math.log(self.max / self.min)
        return (value - self.min) / (self.max - self.min)

    def clamp(self, value: Any) -> tuple[Any, str | None]:
        """The forgiving counterpart of `coerce()`, for values read back
        from storage after the spec may have changed: returns `(value,
        warning)`, where a value outside min/max is clamped to the bound,
        one not in `choices` or of the wrong type falls back to the
        default, and `warning` says what happened (None if nothing)."""
        try:
            return self.coerce(value), None
        except (TypeError, ValueError) as exc:
            reason = str(exc)
        if self.choices is None and self.type in (int, float) and not isinstance(value, bool):
            try:
                number = self.type(value)
            except (TypeError, ValueError):
                return self.default, f"{value!r}: {reason}; using default {self.default!r}"
            if self.min is not None and number < self.min:
                return self.min, f"{value!r} is below the minimum; clamped to {self.min!r}"
            if self.max is not None and number > self.max:
                return self.max, f"{value!r} is above the maximum; clamped to {self.max!r}"
        return self.default, f"{value!r}: {reason}; using default {self.default!r}"

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
    sync: str | None = None
    triggers: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)
    # Derived from params: role name -> param key, macro number -> param key.
    roles: Mapping[str, str] = field(init=False, default_factory=dict)
    macros: Mapping[int, str] = field(init=False, default_factory=dict)

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
        roles: dict[str, str] = {}
        macros: dict[int, str] = {}
        for key, spec in params.items():
            role = spec.role if spec.role is not None else (key if key in ROLES and spec.mappable else None)
            if role is not None:
                if role in roles:
                    raise ValueError(f"params {roles[role]!r} and {key!r} both have role {role!r}")
                roles[role] = key
            if spec.macro is not None:
                if spec.macro in macros:
                    raise ValueError(f"params {macros[spec.macro]!r} and {key!r} are both macro {spec.macro}")
                macros[spec.macro] = key
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "macros", macros)
        if self.sync is not None and self.sync not in SYNCS:
            raise ValueError(f"sync must be one of {SYNCS} or None, got {self.sync!r}")
        if not isinstance(self.triggers, bool):
            raise TypeError(f"triggers must be a bool, got {self.triggers!r}")
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

    def control(self, target: str) -> str | None:
        """The param key an external control reaches: `target` is a role
        name ("speed") or a macro ("macro1"). None when this animation has
        nothing there."""
        if target in ROLES:
            return self.roles.get(target)
        return self.macros.get(macro_number(target))

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


def macro_number(target: str) -> int:
    """"macro1".."macroN" as its number; ValueError for anything else."""
    if isinstance(target, str) and target.startswith("macro") and target[5:].isdigit():
        n = int(target[5:])
        if 1 <= n <= MACROS:
            return n
    raise ValueError(f"not a macro: {target!r}; expected macro1..macro{MACROS}")


def check_control_target(target: str) -> str:
    """`target` if it names a role or a macro; ValueError otherwise."""
    if target not in ROLES:
        try:
            macro_number(target)
        except ValueError:
            raise ValueError(f"control target must be a role {ROLES} or macro1..macro{MACROS}, got {target!r}") from None
    return target


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
    sync: str | None = None,
    triggers: bool = False,
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
        sync=sync,
        triggers=triggers,
        extra=extra,
    )

    def decorate(fn: Callable) -> Callable:
        if not callable(fn):
            raise TypeError(f"@animation must decorate a function, got {fn!r}")
        setattr(fn, META_ATTR, meta)
        return fn

    return decorate
