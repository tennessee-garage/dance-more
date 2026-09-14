"""`AnimationRegistry`: every animation in a directory, and what failed.

    registry = AnimationRegistry.discover(path)
    registry.animations     # {id: AnimationDef}
    registry.errors         # {id: LoadError}
    registry.reload()       # re-import what changed on disk

Adding an animation is "copy a file, edit it, done": every `*.py` in the
directory (skipping `_`-prefixed names) is imported, and the filename stem
is the animation's stable id - what playlists reference. Renaming a file
breaks its playlist entries, by design; the playlist store handles that
visibly rather than silently.

One broken file never breaks the load. A SyntaxError, a failing import or
a module body that raises is captured into `errors` with its traceback and
the rest of the directory loads normally - a live floor must not go dark
because someone saved a typo. A module with no `@animation` is skipped,
not an error; anything decorated but unusable (two animations in a file,
a render with the wrong signature) is a validation error, reported the
same way. Both dicts are meant to be served to the admin UI so problems
are visible.

`reload()` re-imports files whose mtime changed, picks up new ones and
drops deleted ones. If a reload FAILS, the previously loaded version stays
live and the error is recorded alongside it - you can edit an animation on
the Pi while the floor is running and a syntax error costs nothing. It is
polled, not pushed: call it from a dev loop or a UI button. A file-watcher
can drive it later without this module changing.

Several directories may be given (a starter pack plus a user's own); an id
present in more than one is a duplicate, and the later one is recorded as
an error rather than silently shadowing the first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from df2_pi.animation.loader import AnimationDef, LoadError, load_animation_file


def default_animations_dir() -> Path:
    """`animations/` next to the package's `src/` - `src/pi/animations` in
    the repo, `~/dance-floor/animations` on the Pi."""
    return Path(__file__).resolve().parents[3] / "animations"


class AnimationRegistry:
    def __init__(self, paths: Path | str | Iterable[Path | str]) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        self.animations: dict[str, AnimationDef] = {}
        self.errors: dict[str, LoadError] = {}
        self._mtimes: dict[Path, float] = {}

    @classmethod
    def discover(cls, paths: Path | str | Iterable[Path | str]) -> AnimationRegistry:
        """Load everything under `paths`."""
        registry = cls(paths)
        registry.reload()
        return registry

    def files(self) -> list[Path]:
        """The animation files under `paths`, in load order: directory
        order as given, then by name."""
        out: list[Path] = []
        for directory in self.paths:
            if not directory.is_dir():
                continue
            out.extend(
                p for p in sorted(directory.glob("*.py"))
                if not p.name.startswith("_") and p.is_file()
            )
        return out

    def reload(self) -> list[str]:
        """Bring the registry up to date with the directories. Returns the
        ids whose entry changed: loaded, reloaded, newly failing, or
        removed."""
        changed: list[str] = []
        present = self.files()
        seen_ids: dict[str, Path] = {}

        for path in present:
            stem = path.stem
            if stem in seen_ids and seen_ids[stem] != path:
                self.errors[stem] = LoadError(
                    path,
                    "validate",
                    f"duplicate animation id {stem!r} - already loaded from {seen_ids[stem]}",
                )
                continue
            seen_ids[stem] = path
            mtime = _mtime(path)
            if mtime is None or self._mtimes.get(path) == mtime:
                continue
            self._mtimes[path] = mtime
            changed.append(stem)
            try:
                definition = load_animation_file(path)
            except LoadError as exc:
                # A previously loaded version, if any, stays live.
                self.errors[stem] = exc
                continue
            self.errors.pop(stem, None)
            if definition is None:
                # Not an animation (no decorator): a helper module. If it
                # used to be one, it no longer is.
                self.animations.pop(stem, None)
                continue
            self.animations[stem] = definition

        # Removed files: drop their entries (and forget their mtimes).
        present_paths = set(present)
        for path in list(self._mtimes):
            if path not in present_paths:
                del self._mtimes[path]
                stem = path.stem
                if stem not in seen_ids:
                    self.animations.pop(stem, None)
                    self.errors.pop(stem, None)
                    changed.append(stem)
        return changed

    def get(self, animation_id: str) -> AnimationDef | None:
        return self.animations.get(animation_id)

    def __getitem__(self, animation_id: str) -> AnimationDef:
        try:
            return self.animations[animation_id]
        except KeyError:
            error = self.errors.get(animation_id)
            detail = f" (failed to load: {error.message})" if error else ""
            raise KeyError(f"no animation {animation_id!r}{detail}") from None

    def __contains__(self, animation_id: str) -> bool:
        return animation_id in self.animations

    def __len__(self) -> int:
        return len(self.animations)

    def __iter__(self):
        return iter(self.animations.values())


def _mtime(path: Path) -> float | None:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None
