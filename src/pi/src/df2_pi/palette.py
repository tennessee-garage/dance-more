"""The shared floor palette (#128).

A lighting designer sets a colour scheme for the room; an animation that
sweeps the whole hue wheel fights it. A palette lets the desk choose the
colours and the animations choose what to do with them.

    pal = palette.choice(ctx, ctx.params["palette"])   # the floor's, or a named one
    colour = pal.at(0.25)                              # (3,) uint8
    colours = pal.at(offsets)                          # (..., 3) uint8, one per offset

`Palette` is 2..8 colour stops spaced evenly round a loop: `at(0)` is the
first stop, `at(k / n)` the k-th, and past the last stop it blends back to
the first, so `u` wraps and a value that keeps rising cycles through the
scheme without a jump. Blending is in linear light (`gamma.py`), so the
middle of red and green is a yellow as bright as either, not a muddy one.

The ACTIVE palette is runner state, chosen in the web UI or from a desk
(DMX control channel 18), and reaches animations as `ctx.palette`. An
animation opts in by declaring `palette_param()` - a `palette` param whose
choices are "floor" (follow the active palette, the default) and the
built-in library - and resolving it with `choice()`. Animations that do
not are unaffected. A change lands at the next frame boundary; an
animation that wants it smooth blends on its side.

`PaletteBook` is the library plus the user's own palettes, kept in the
settings table as JSON, and which one is active.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from df2_pi.gamma import from_linear, to_linear

if TYPE_CHECKING:
    from df2_pi.animation.meta import Param
    from df2_pi.engine.runner import Runner
    from df2_pi.playlists import PlaylistStore

log = logging.getLogger(__name__)

MIN_STOPS, MAX_STOPS = 2, 8
FLOOR = "floor"  # the param value meaning "follow the active palette"

# The built-in library, in the order the UI and DMX channel 18 count them.
LIBRARY: dict[str, tuple[str, ...]] = {
    "rainbow": ("ff0000", "ffff00", "00ff00", "00ffff", "0000ff", "ff00ff"),
    "fire": ("ff1000", "ff6000", "ffb000", "ffe890"),
    "ice": ("ffffff", "a0e0ff", "2080ff", "0018c0"),
    "ocean": ("003070", "0070b0", "00c0c0", "80ffe0"),
    "sunset": ("ff6000", "ff2060", "a020a0", "4020c0"),
    "forest": ("185018", "2a8a20", "90c820", "f0e060"),
    "neon": ("ff00ff", "00ffff", "ffff00", "ff0080"),
    "candy": ("ff80c0", "80c0ff", "c0ff80", "ffd080"),
    "night": ("140a48", "3a1a98", "6a24c8", "2060a8"),
}
DEFAULT_ACTIVE = "rainbow"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


class Palette:
    def __init__(self, stops: Iterable[str | Sequence[int]], name: str | None = None) -> None:
        colours = [_rgb(stop) for stop in stops]
        if not MIN_STOPS <= len(colours) <= MAX_STOPS:
            raise ValueError(f"a palette has {MIN_STOPS}..{MAX_STOPS} stops, got {len(colours)}")
        self.name = name
        self.stops = np.array(colours, dtype=np.uint8)  # (n, 3) perceptual bytes
        self._linear = to_linear(self.stops).astype(np.float64)  # (n, 3)

    def at(self, u) -> np.ndarray:
        """The colour at `u` round the loop (wrapping): (3,) uint8 for a
        number, (..., 3) for an array."""
        n = len(self.stops)
        position = np.mod(np.asarray(u, dtype=np.float64), 1.0) * n
        first = np.floor(position).astype(np.intp) % n
        frac = (position - np.floor(position))[..., None]
        light = self._linear[first] * (1.0 - frac) + self._linear[(first + 1) % n] * frac
        return from_linear(light)

    def hex(self) -> list[str]:
        return ["".join(f"{v:02x}" for v in stop) for stop in self.stops]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Palette) and np.array_equal(self.stops, other.stops)

    def __repr__(self) -> str:
        return f"Palette({self.name!r}, {self.hex()})"


def palette_param(**kwargs: Any) -> Param:
    """The param an animation declares to opt in: "floor" (the default)
    follows the active palette; the others are the built-in library."""
    from df2_pi.animation.meta import Param  # here, not at the top: the animation package imports this module

    kwargs.setdefault("label", "Palette")
    kwargs.setdefault("help", "floor: whatever the floor's palette is set to")
    return Param(str, default=FLOOR, choices=[FLOOR, *LIBRARY], **kwargs)


def choice(ctx, name: str) -> Palette:
    """Resolve a `palette_param` value: the active palette for "floor"."""
    return ctx.palette if name == FLOOR else BUILTIN[name]


def _rgb(stop: str | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(stop, str):
        match = _HEX.match(stop.strip())
        if not match:
            raise ValueError(f"a palette stop is a hex colour like 'ff8000', got {stop!r}")
        value = match.group(1)
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    rgb = tuple(int(v) for v in stop)
    if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
        raise ValueError(f"a palette stop is (r, g, b) in 0..255, got {stop!r}")
    return rgb  # type: ignore[return-value]


BUILTIN: dict[str, Palette] = {name: Palette(stops, name) for name, stops in LIBRARY.items()}


# ---- the book: built-ins, the user's own, and the active one ------------------------------------


class PaletteBook:
    """The palettes the floor knows and which is active. Thread-safe: the
    web UI and the DMX control block both call it."""

    def __init__(self, store: PlaylistStore | None = None, runner: Runner | None = None) -> None:
        self.store = store
        self.runner = runner
        self._lock = threading.Lock()
        self._user: dict[str, Palette] = self._load_user()
        active = store.get_str("active_palette") if store is not None else None
        self._active = active if active in self._all() else DEFAULT_ACTIVE
        if active is not None and active != self._active:
            log.warning("active palette %r no longer exists; using %r", active, self._active)
        self._push()

    @property
    def active(self) -> str:
        return self._active

    def names(self) -> list[str]:
        """Built-ins in library order, then the user's alphabetically: the
        order DMX channel 18 counts."""
        with self._lock:
            return list(self._all())

    def get(self, name: str) -> Palette:
        with self._lock:
            palettes = self._all()
            if name not in palettes:
                raise KeyError(name)
            return palettes[name]

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"name": name, "stops": p.hex(), "builtin": name in BUILTIN} for name, p in self._all().items()]

    def activate(self, name: str) -> None:
        with self._lock:
            if name not in self._all():
                raise KeyError(name)
            self._active = name
        if self.store is not None:
            self.store.set_setting("active_palette", name)
        self._push()

    def save(self, name: str, stops: Iterable[str | Sequence[int]]) -> Palette:
        """Create or replace a user palette. Raises ValueError for a bad
        name, a built-in's name, or bad stops."""
        if not _NAME.match(name) or name == FLOOR:
            raise ValueError("a palette name is 1-32 of a-z, 0-9, '-' and '_', starting with a letter or digit, and not 'floor'")
        if name in BUILTIN:
            raise ValueError(f"{name!r} is a built-in palette; save a copy under another name")
        palette = Palette(stops, name)
        with self._lock:
            self._user[name] = palette
            self._write_user()
        if name == self._active:
            self._push()
        return palette

    def delete(self, name: str) -> None:
        if name in BUILTIN:
            raise ValueError(f"{name!r} is a built-in palette and cannot be deleted")
        with self._lock:
            if name not in self._user:
                raise KeyError(name)
            del self._user[name]
            self._write_user()
            was_active = name == self._active
        if was_active:
            self.activate(DEFAULT_ACTIVE)

    def _all(self) -> dict[str, Palette]:
        return {**BUILTIN, **dict(sorted(self._user.items()))}

    def _push(self) -> None:
        if self.runner is not None:
            self.runner.set_palette(self.get(self._active))

    def _load_user(self) -> dict[str, Palette]:
        raw = self.store.get_setting("user_palettes") if self.store is not None else None
        if not raw:
            return {}
        try:
            stored = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("setting 'user_palettes' is not JSON; ignoring it")
            return {}
        palettes = {}
        for name, stops in (stored.items() if isinstance(stored, dict) else ()):
            try:
                if name in BUILTIN or not _NAME.match(str(name)):
                    raise ValueError("bad name")
                palettes[name] = Palette(stops, name)
            except (TypeError, ValueError) as exc:
                log.warning("user palette %r: %s; skipping it", name, exc)
        return palettes

    def _write_user(self) -> None:
        if self.store is not None:
            self.store.set_setting("user_palettes", json.dumps({n: p.hex() for n, p in sorted(self._user.items())}))
