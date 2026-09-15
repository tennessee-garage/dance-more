"""`PlaylistStore`: playlists, entries, settings and the play log, in SQLite.

Animations are files; playlists are DATA. A playlist names an ordered set
of animations with per-entry durations and parameter overrides.

    store = PlaylistStore(path)                  # migrates on open
    pl = store.create_playlist("Friday")
    store.add_entry(pl, "rainbow_sweep", duration_s=90, params={"speed": 2})
    pl = store.playlist("Friday")                # .entries in position order
    resolved = store.resolve(pl, registry)       # entries bound to AnimationDefs

Animations are not rows. `Entry.animation_id` is the module stem, not a
foreign key: files stay the source of truth and the database only
records intent. An entry whose animation is missing or failed to load
resolves as `unresolved` and is skipped at playback, never deleted - so
renaming a file or breaking it with a typo degrades a playlist instead of
silently eating it, and fixing the file fixes the playlist.

Durations are always explicit. Animations declare none, so `add_entry(
duration_s=None)` fills in the `default_entry_duration` setting AT WRITE
TIME and the row is always concrete; changing the setting later affects
new entries only.

Parameters: `params` holds only the keys that DIFFER from the animation's
defaults, so changing a default in the file propagates to existing
playlists. With a registry attached, values are validated against the
`Param` specs on write; on `resolve()` they are clamped with a warning
rather than raised, because the specs can change under a stored playlist
and an out-of-range value must not take the render loop down.

Concurrency: WAL mode and one connection per thread, so the web thread
writes while the render thread reads. The render thread only ever holds
the frozen dataclasses this hands out; it never touches sqlite mid-frame.
`:memory:` databases exist per connection, so those get one shared
connection behind a lock instead.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from df2_pi.animation.loader import AnimationDef
from df2_pi.animation.registry import AnimationRegistry
from df2_pi.playlists.schema import migrate

log = logging.getLogger(__name__)

DEFAULT_SETTINGS: dict[str, Any] = {
    "startup_playlist": None,  # a playlist id
    "brightness": 255,
    "enabled": True,
    "fps": 30.0,
    "default_entry_duration": 60.0,
}

OUTCOMES = ("completed", "skipped", "error")


def default_db_path() -> Path:
    """`$DF2_DB`, else `~/.local/share/df2/df2.sqlite3` (XDG_DATA_HOME
    respected)."""
    env = os.environ.get("DF2_DB")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return root / "df2" / "df2.sqlite3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- the rows, as handed to the runner ----------------------------------------------


@dataclass(frozen=True)
class Entry:
    id: int
    playlist_id: int
    position: int
    animation_id: str
    duration_s: float
    params: Mapping[str, Any]
    enabled: bool


@dataclass(frozen=True)
class Playlist:
    id: int
    name: str
    description: str
    loop: bool
    shuffle: bool
    crossfade_s: float
    created_at: str
    updated_at: str
    entries: tuple[Entry, ...] = ()


@dataclass(frozen=True)
class ResolvedEntry:
    """An entry bound to its animation, or marked unresolved with the
    reason. `params` is the full resolved dict (defaults plus the stored
    overrides, clamped); `warnings` lists any clamping that happened."""

    entry: Entry
    definition: AnimationDef | None
    params: Mapping[str, Any]
    error: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def unresolved(self) -> bool:
        return self.definition is None

    @property
    def playable(self) -> bool:
        return self.entry.enabled and self.definition is not None


@dataclass(frozen=True)
class ResolvedPlaylist:
    playlist: Playlist
    entries: tuple[ResolvedEntry, ...]

    @property
    def playable(self) -> tuple[ResolvedEntry, ...]:
        return tuple(e for e in self.entries if e.playable)


@dataclass(frozen=True)
class PlayLogEntry:
    id: int
    started_at: str
    animation_id: str
    playlist_id: int | None
    duration_s: float | None
    outcome: str


# ---- the store ------------------------------------------------------------------------


class PlaylistStore:
    def __init__(self, path: Path | str | None = None, registry: AnimationRegistry | None = None) -> None:
        self.path = ":memory:" if path == ":memory:" else Path(path if path is not None else default_db_path())
        self.registry = registry
        self._local = threading.local()
        self._memory_conn: sqlite3.Connection | None = None
        self._memory_lock = threading.RLock()
        if self.path != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path == ":memory:":
            with self._memory_lock:
                self._memory_conn = self._open()
                migrate(self._memory_conn)
        else:
            migrate(self._thread_conn())

    # ---- connections ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            check_same_thread=self.path != ":memory:",
            isolation_level=None,  # autocommit; transactions are explicit
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _thread_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._open()
        return conn

    class _Txn:
        """`with store._conn() as conn:` - a transaction on this thread's
        connection (or the shared one, under its lock, for :memory:)."""

        def __init__(self, store: PlaylistStore) -> None:
            self.store = store

        def __enter__(self) -> sqlite3.Connection:
            store = self.store
            if store.path == ":memory:":
                store._memory_lock.acquire()
                if store._memory_conn is None:
                    store._memory_conn = store._open()
                conn = store._memory_conn
            else:
                conn = store._thread_conn()
            self.conn = conn
            conn.execute("BEGIN")
            return conn

        def __exit__(self, exc_type, exc, tb) -> None:
            try:
                if exc_type is None:
                    self.conn.execute("COMMIT")
                else:
                    self.conn.execute("ROLLBACK")
            finally:
                if self.store.path == ":memory:":
                    self.store._memory_lock.release()

    def _conn(self) -> PlaylistStore._Txn:
        return PlaylistStore._Txn(self)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
        with self._memory_lock:
            if self._memory_conn is not None:
                self._memory_conn.close()
                self._memory_conn = None

    # ---- playlists ---------------------------------------------------------------------

    def playlists(self) -> list[Playlist]:
        """Every playlist, by name, with entries."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM playlist ORDER BY name").fetchall()
            return [self._playlist_from_row(conn, row) for row in rows]

    def playlist(self, id_or_name: int | str) -> Playlist:
        """One playlist by id or name, entries in position order. Raises
        KeyError if there is no such playlist."""
        with self._conn() as conn:
            row = self._playlist_row(conn, id_or_name)
            return self._playlist_from_row(conn, row)

    def create_playlist(
        self,
        name: str,
        *,
        description: str = "",
        loop: bool = True,
        shuffle: bool = False,
        crossfade_s: float = 0.0,
    ) -> Playlist:
        if not name or not name.strip():
            raise ValueError("playlist name must not be empty")
        if crossfade_s < 0:
            raise ValueError(f"crossfade_s must be non-negative, got {crossfade_s}")
        now = _now()
        with self._conn() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO playlist (name, description, loop, shuffle, crossfade_s, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (name.strip(), description, int(loop), int(shuffle), float(crossfade_s), now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"a playlist named {name!r} already exists") from exc
            return self._playlist_from_row(conn, self._playlist_row(conn, cur.lastrowid))

    def update_playlist(self, playlist: Playlist | int, **fields: Any) -> Playlist:
        """Change name / description / loop / shuffle / crossfade_s."""
        allowed = {"name", "description", "loop", "shuffle", "crossfade_s"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"cannot update {sorted(unknown)}; allowed: {sorted(allowed)}")
        pid = playlist if isinstance(playlist, int) else playlist.id
        if not fields:
            return self.playlist(pid)
        values: dict[str, Any] = {}
        for key, value in fields.items():
            if key in ("loop", "shuffle"):
                values[key] = int(bool(value))
            elif key == "crossfade_s":
                if value < 0:
                    raise ValueError("crossfade_s must be non-negative")
                values[key] = float(value)
            elif key == "name":
                if not value or not str(value).strip():
                    raise ValueError("playlist name must not be empty")
                values[key] = str(value).strip()
            else:
                values[key] = str(value)
        values["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in values)
        with self._conn() as conn:
            self._playlist_row(conn, pid)  # KeyError if missing
            try:
                conn.execute(f"UPDATE playlist SET {assignments} WHERE id = ?", (*values.values(), pid))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"a playlist named {values.get('name')!r} already exists") from exc
            return self._playlist_from_row(conn, self._playlist_row(conn, pid))

    def delete_playlist(self, playlist: Playlist | int) -> None:
        pid = playlist if isinstance(playlist, int) else playlist.id
        with self._conn() as conn:
            self._playlist_row(conn, pid)
            conn.execute("DELETE FROM playlist WHERE id = ?", (pid,))

    # ---- entries -------------------------------------------------------------------------

    def add_entry(
        self,
        playlist: Playlist | int,
        animation_id: str,
        duration_s: float | None = None,
        params: Mapping[str, Any] | None = None,
        position: int | None = None,
        *,
        enabled: bool = True,
    ) -> Entry:
        """Append (or insert at `position`, shifting the rest down) an
        entry. `duration_s=None` takes the `default_entry_duration`
        setting now; `params` are reduced to the keys differing from the
        animation's defaults when the registry knows it."""
        pid = playlist if isinstance(playlist, int) else playlist.id
        if not animation_id or not animation_id.isidentifier():
            raise ValueError(f"animation_id must be a module stem, got {animation_id!r}")
        if duration_s is None:
            duration_s = self.get_float("default_entry_duration")
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        overrides = self._diff_params(animation_id, params or {})
        with self._conn() as conn:
            self._playlist_row(conn, pid)
            count = conn.execute(
                "SELECT COUNT(*) FROM playlist_entry WHERE playlist_id = ?", (pid,)
            ).fetchone()[0]
            if position is None or position >= count:
                position = count
            elif position < 0:
                raise ValueError(f"position must be non-negative, got {position}")
            else:
                self._shift_down(conn, pid, position)
            cur = conn.execute(
                "INSERT INTO playlist_entry (playlist_id, position, animation_id, duration_s, params_json, enabled)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (pid, position, animation_id, float(duration_s), json.dumps(overrides, sort_keys=True), int(enabled)),
            )
            self._touch(conn, pid)
            return self._entry_from_row(conn.execute("SELECT * FROM playlist_entry WHERE id = ?", (cur.lastrowid,)).fetchone())

    def entry(self, entry_id: int) -> Entry:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM playlist_entry WHERE id = ?", (entry_id,)).fetchone()
            if row is None:
                raise KeyError(f"no entry {entry_id}")
            return self._entry_from_row(row)

    def update_entry(self, entry: Entry | int, **fields: Any) -> Entry:
        """Change animation_id / duration_s / params / enabled. `params`
        replaces the overrides wholesale (and is diffed against defaults)."""
        allowed = {"animation_id", "duration_s", "params", "enabled"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"cannot update {sorted(unknown)}; allowed: {sorted(allowed)}")
        eid = entry if isinstance(entry, int) else entry.id
        current = self.entry(eid)
        values: dict[str, Any] = {}
        if "animation_id" in fields:
            animation_id = fields["animation_id"]
            if not animation_id or not str(animation_id).isidentifier():
                raise ValueError(f"animation_id must be a module stem, got {animation_id!r}")
            values["animation_id"] = animation_id
        if "duration_s" in fields:
            if fields["duration_s"] <= 0:
                raise ValueError("duration_s must be positive")
            values["duration_s"] = float(fields["duration_s"])
        if "params" in fields:
            animation_id = values.get("animation_id", current.animation_id)
            values["params_json"] = json.dumps(self._diff_params(animation_id, fields["params"] or {}), sort_keys=True)
        if "enabled" in fields:
            values["enabled"] = int(bool(fields["enabled"]))
        if not values:
            return current
        assignments = ", ".join(f"{k} = ?" for k in values)
        with self._conn() as conn:
            conn.execute(f"UPDATE playlist_entry SET {assignments} WHERE id = ?", (*values.values(), eid))
            self._touch(conn, current.playlist_id)
            return self._entry_from_row(conn.execute("SELECT * FROM playlist_entry WHERE id = ?", (eid,)).fetchone())

    def move_entry(self, entry: Entry | int, new_position: int) -> Entry:
        """Move an entry within its playlist; positions stay contiguous from 0."""
        eid = entry if isinstance(entry, int) else entry.id
        current = self.entry(eid)
        with self._conn() as conn:
            ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM playlist_entry WHERE playlist_id = ? ORDER BY position",
                    (current.playlist_id,),
                )
            ]
            new_position = max(0, min(new_position, len(ids) - 1))
            ids.remove(eid)
            ids.insert(new_position, eid)
            self._renumber(conn, ids)
            self._touch(conn, current.playlist_id)
            return self._entry_from_row(conn.execute("SELECT * FROM playlist_entry WHERE id = ?", (eid,)).fetchone())

    def remove_entry(self, entry: Entry | int) -> None:
        eid = entry if isinstance(entry, int) else entry.id
        current = self.entry(eid)
        with self._conn() as conn:
            conn.execute("DELETE FROM playlist_entry WHERE id = ?", (eid,))
            ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM playlist_entry WHERE playlist_id = ? ORDER BY position",
                    (current.playlist_id,),
                )
            ]
            self._renumber(conn, ids)
            self._touch(conn, current.playlist_id)

    # ---- resolving against the registry ------------------------------------------------

    def resolve(self, playlist: Playlist | int | str, registry: AnimationRegistry | None = None) -> ResolvedPlaylist:
        """Bind each entry to its `AnimationDef`. Entries whose animation is
        missing or failed to load are marked unresolved (with the load
        error when the registry has one) and never dropped; stored params
        are clamped to the current specs, with warnings."""
        registry = registry if registry is not None else self.registry
        if registry is None:
            raise ValueError("resolve() needs an AnimationRegistry")
        pl = playlist if isinstance(playlist, Playlist) else self.playlist(playlist)
        resolved = []
        for entry in pl.entries:
            definition = registry.get(entry.animation_id)
            if definition is None:
                load_error = registry.errors.get(entry.animation_id)
                error = (
                    f"failed to load: {load_error.message}" if load_error else "no such animation"
                )
                resolved.append(ResolvedEntry(entry, None, dict(entry.params), error=error))
                continue
            params = definition.meta.defaults()
            warnings = []
            for key, value in entry.params.items():
                spec = definition.meta.params.get(key)
                if spec is None:
                    warnings.append(f"{key!r} is not a parameter of {entry.animation_id!r}; ignored")
                    continue
                params[key], warning = spec.clamp(value)
                if warning:
                    warnings.append(f"{key}: {warning}")
            for warning in warnings:
                log.warning("playlist %r entry %d (%s): %s", pl.name, entry.position, entry.animation_id, warning)
            resolved.append(ResolvedEntry(entry, definition, params, warnings=tuple(warnings)))
        return ResolvedPlaylist(pl, tuple(resolved))

    # ---- settings ---------------------------------------------------------------------------

    def get_setting(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM setting WHERE key = ?", (key,)).fetchone()
            return None if row is None else row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        if value is None:
            with self._conn() as conn:
                conn.execute("DELETE FROM setting WHERE key = ?", (key,))
            return
        if isinstance(value, bool):
            value = "1" if value else "0"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO setting (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    def settings(self) -> dict[str, str]:
        with self._conn() as conn:
            return {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM setting ORDER BY key")}

    def get_str(self, key: str, default: str | None = None) -> str | None:
        value = self.get_setting(key)
        return self._default(key, default) if value is None else value

    def get_int(self, key: str, default: int | None = None) -> int | None:
        return self._typed(key, default, lambda v: int(float(v)))

    def get_float(self, key: str, default: float | None = None) -> float | None:
        return self._typed(key, default, float)

    def get_bool(self, key: str, default: bool | None = None) -> bool | None:
        def parse(v: str) -> bool:
            lowered = v.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            raise ValueError(v)

        return self._typed(key, default, parse)

    def _typed(self, key: str, default: Any, parse) -> Any:
        value = self.get_setting(key)
        if value is None:
            return self._default(key, default)
        try:
            return parse(value)
        except (TypeError, ValueError):
            fallback = self._default(key, default)
            log.warning("setting %r has malformed value %r; using %r", key, value, fallback)
            return fallback

    @staticmethod
    def _default(key: str, default: Any) -> Any:
        return DEFAULT_SETTINGS.get(key) if default is None else default

    # ---- play log ---------------------------------------------------------------------------

    def log_play(
        self,
        animation_id: str,
        outcome: str,
        *,
        playlist_id: int | None = None,
        duration_s: float | None = None,
        started_at: str | None = None,
    ) -> PlayLogEntry:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO play_log (started_at, animation_id, playlist_id, duration_s, outcome) VALUES (?, ?, ?, ?, ?)",
                (started_at or _now(), animation_id, playlist_id, duration_s, outcome),
            )
            row = conn.execute("SELECT * FROM play_log WHERE id = ?", (cur.lastrowid,)).fetchone()
            return PlayLogEntry(**dict(row))

    def play_log(self, limit: int = 100) -> list[PlayLogEntry]:
        """Most recent first."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM play_log ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
            return [PlayLogEntry(**dict(row)) for row in rows]

    # ---- seeding ---------------------------------------------------------------------------

    def seed_default(self, registry: AnimationRegistry | None = None, name: str = "Default") -> Playlist | None:
        """On an empty database, create a playlist of every animation the
        registry has (by id) and make it the startup playlist, so a fresh
        install lights the floor without anyone opening a UI. Returns the
        playlist created, or None if there already were playlists."""
        registry = registry if registry is not None else self.registry
        if self.playlists():
            return None
        ids = sorted(registry.animations) if registry is not None else []
        pl = self.create_playlist(name, description="Every animation, in order.")
        for animation_id in ids:
            self.add_entry(pl, animation_id)
        self.set_setting("startup_playlist", pl.id)
        return self.playlist(pl.id)

    def startup_playlist(self) -> Playlist | None:
        """The playlist to load at startup, or None if unset or gone."""
        pid = self.get_int("startup_playlist")
        if pid is None:
            return None
        try:
            return self.playlist(pid)
        except KeyError:
            log.warning("startup_playlist %r no longer exists", pid)
            return None

    # ---- internals ---------------------------------------------------------------------------

    def _diff_params(self, animation_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Only the keys whose value differs from the animation's default,
        validated against its specs when the registry knows it. Unknown
        animations keep whatever they were given."""
        definition = self.registry.get(animation_id) if self.registry is not None else None
        if definition is None:
            return dict(params)
        specs = definition.meta.params
        out: dict[str, Any] = {}
        for key, value in params.items():
            if key not in specs:
                raise ValueError(f"{key!r} is not a parameter of {animation_id!r}; declared: {sorted(specs)}")
            value = specs[key].coerce(value)
            if value != specs[key].default:
                out[key] = value
        return out

    @staticmethod
    def _playlist_row(conn: sqlite3.Connection, id_or_name: int | str) -> sqlite3.Row:
        if isinstance(id_or_name, int):
            row = conn.execute("SELECT * FROM playlist WHERE id = ?", (id_or_name,)).fetchone()
        else:
            row = conn.execute("SELECT * FROM playlist WHERE name = ?", (id_or_name,)).fetchone()
        if row is None:
            raise KeyError(f"no playlist {id_or_name!r}")
        return row

    def _playlist_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Playlist:
        entries = tuple(
            self._entry_from_row(r)
            for r in conn.execute(
                "SELECT * FROM playlist_entry WHERE playlist_id = ? ORDER BY position", (row["id"],)
            )
        )
        return Playlist(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            loop=bool(row["loop"]),
            shuffle=bool(row["shuffle"]),
            crossfade_s=row["crossfade_s"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            entries=entries,
        )

    @staticmethod
    def _entry_from_row(row: sqlite3.Row) -> Entry:
        try:
            params = json.loads(row["params_json"])
            if not isinstance(params, dict):
                raise ValueError("not an object")
        except ValueError:
            log.warning("entry %d has malformed params_json %r; ignoring", row["id"], row["params_json"])
            params = {}
        return Entry(
            id=row["id"],
            playlist_id=row["playlist_id"],
            position=row["position"],
            animation_id=row["animation_id"],
            duration_s=row["duration_s"],
            params=params,
            enabled=bool(row["enabled"]),
        )

    @staticmethod
    def _renumber(conn: sqlite3.Connection, ids: Iterable[int]) -> None:
        """Assign positions 0.. to `ids` in order. Two passes through
        negative numbers so the UNIQUE (playlist_id, position) constraint
        never sees a collision mid-way."""
        ids = list(ids)
        for i, eid in enumerate(ids):
            conn.execute("UPDATE playlist_entry SET position = ? WHERE id = ?", (-(i + 1), eid))
        for i, eid in enumerate(ids):
            conn.execute("UPDATE playlist_entry SET position = ? WHERE id = ?", (i, eid))

    def _shift_down(self, conn: sqlite3.Connection, pid: int, from_position: int) -> None:
        """Open a hole at `from_position` by moving that entry and every
        later one down by one."""
        ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM playlist_entry WHERE playlist_id = ? AND position >= ? ORDER BY position DESC",
                (pid, from_position),
            )
        ]
        for eid in ids:  # descending order, so each move lands in a free slot
            conn.execute("UPDATE playlist_entry SET position = position + 1 WHERE id = ?", (eid,))

    @staticmethod
    def _touch(conn: sqlite3.Connection, pid: int) -> None:
        conn.execute("UPDATE playlist SET updated_at = ? WHERE id = ?", (_now(), pid))
