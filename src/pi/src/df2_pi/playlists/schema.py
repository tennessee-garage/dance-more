"""The playlist database schema and its forward-only migration runner.

Four tables plus a version stamp, stdlib sqlite3, no ORM. `MIGRATIONS[i]`
takes the schema from version i to i+1 and is applied in one transaction;
`migrate()` runs whatever is missing and is a no-op on an up-to-date
database. Only ever append to the list.
"""

from __future__ import annotations

import sqlite3

MIGRATIONS: list[str] = [
    # 0 -> 1: the initial schema
    """
    CREATE TABLE playlist (
        id          INTEGER PRIMARY KEY,
        name        TEXT NOT NULL UNIQUE,
        description TEXT NOT NULL DEFAULT '',
        loop        INTEGER NOT NULL DEFAULT 1,
        shuffle     INTEGER NOT NULL DEFAULT 0,
        crossfade_s REAL    NOT NULL DEFAULT 0.0,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );

    CREATE TABLE playlist_entry (
        id           INTEGER PRIMARY KEY,
        playlist_id  INTEGER NOT NULL REFERENCES playlist(id) ON DELETE CASCADE,
        position     INTEGER NOT NULL,
        animation_id TEXT    NOT NULL,
        duration_s   REAL    NOT NULL,
        params_json  TEXT    NOT NULL DEFAULT '{}',
        enabled      INTEGER NOT NULL DEFAULT 1,
        UNIQUE (playlist_id, position)
    );

    CREATE TABLE setting (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE play_log (
        id           INTEGER PRIMARY KEY,
        started_at   TEXT NOT NULL,
        animation_id TEXT NOT NULL,
        playlist_id  INTEGER,
        duration_s   REAL,
        outcome      TEXT NOT NULL
    );

    CREATE TABLE schema_version (
        version INTEGER NOT NULL
    );
    INSERT INTO schema_version (version) VALUES (0);
    """,
]

SCHEMA_VERSION = len(MIGRATIONS)


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    return int(row[0]) if row is not None else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Bring `conn`'s database up to `SCHEMA_VERSION`. Returns the number
    of migrations applied (0 when already current)."""
    version = current_version(conn)
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema is version {version}, newer than this code's {SCHEMA_VERSION}"
        )
    applied = 0
    for target, sql in enumerate(MIGRATIONS[version:], start=version + 1):
        # One transaction per migration, version stamp included, so a
        # failure part-way leaves the database at the previous version.
        conn.executescript(
            f"BEGIN;{sql}UPDATE schema_version SET version = {int(target)};COMMIT;"
        )
        applied += 1
    return applied
