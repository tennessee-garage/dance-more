import json
import sqlite3
import textwrap
import threading
import time
from pathlib import Path

import pytest

from df2_pi.animation import AnimationRegistry
from df2_pi.playlists import DEFAULT_SETTINGS, PlaylistStore, ResolvedPlaylist, default_db_path
from df2_pi.playlists.schema import MIGRATIONS, SCHEMA_VERSION, current_version, migrate

SOLID = '''
from df2_pi.animation import animation, Param
from df2_pi.pixels import TileFrame

@animation(
    name="Solid",
    params={
        "level": Param(int, default=10, min=0, max=255),
        "mode": Param(str, default="up", choices=["up", "down"]),
    },
    period=2.0,
)
def render(previous, ctx):
    return TileFrame.black(ctx.geometry)
'''

BROKEN = '''
from df2_pi.animation import animation
@animation(name="Broken"
def render(previous, ctx):
    return previous
'''


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    d = tmp_path / "animations"
    d.mkdir()
    (d / "solid.py").write_text(textwrap.dedent(SOLID))
    (d / "other.py").write_text(textwrap.dedent(SOLID).replace('name="Solid"', 'name="Other"'))
    (d / "broken.py").write_text(textwrap.dedent(BROKEN))
    return AnimationRegistry.discover(d)


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path: Path, registry) -> PlaylistStore:
    path = ":memory:" if request.param == "memory" else tmp_path / "df2.sqlite3"
    s = PlaylistStore(path, registry=registry)
    yield s
    s.close()


# ---- schema ---------------------------------------------------------------------------


def test_migration_from_empty_creates_the_schema_and_rerunning_is_a_noop(tmp_path):
    conn = sqlite3.connect(tmp_path / "x.sqlite3")
    assert current_version(conn) == 0
    assert migrate(conn) == len(MIGRATIONS) == SCHEMA_VERSION
    assert current_version(conn) == SCHEMA_VERSION
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"playlist", "playlist_entry", "setting", "play_log", "schema_version"} <= tables
    assert migrate(conn) == 0
    assert current_version(conn) == SCHEMA_VERSION
    conn.execute("UPDATE schema_version SET version = 99")
    conn.commit()
    with pytest.raises(RuntimeError, match="newer"):
        migrate(conn)


def test_opening_a_store_twice_on_the_same_file_is_fine(tmp_path):
    path = tmp_path / "df2.sqlite3"
    a = PlaylistStore(path)
    a.create_playlist("A")
    b = PlaylistStore(path)
    assert [p.name for p in b.playlists()] == ["A"]
    a.close()
    b.close()


def test_default_db_path_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DF2_DB", str(tmp_path / "custom.sqlite3"))
    assert default_db_path() == tmp_path / "custom.sqlite3"
    monkeypatch.delenv("DF2_DB")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_db_path() == tmp_path / "xdg" / "df2" / "df2.sqlite3"


# ---- playlists ------------------------------------------------------------------------


def test_playlist_crud(store):
    pl = store.create_playlist("Friday", description="the usual", loop=False, shuffle=True, crossfade_s=1.5)
    assert (pl.name, pl.description, pl.loop, pl.shuffle, pl.crossfade_s) == ("Friday", "the usual", False, True, 1.5)
    assert pl.entries == () and pl.created_at == pl.updated_at
    assert store.playlist(pl.id) == pl and store.playlist("Friday") == pl
    with pytest.raises(ValueError, match="already exists"):
        store.create_playlist("Friday")
    with pytest.raises(ValueError):
        store.create_playlist("  ")
    with pytest.raises(ValueError):
        store.create_playlist("x", crossfade_s=-1)

    updated = store.update_playlist(pl, name="Saturday", loop=True, crossfade_s=0)
    assert (updated.name, updated.loop, updated.crossfade_s) == ("Saturday", True, 0.0)
    assert [p.name for p in store.playlists()] == ["Saturday"]
    with pytest.raises(ValueError, match="cannot update"):
        store.update_playlist(pl, id=9)
    with pytest.raises(KeyError):
        store.playlist("Friday")
    with pytest.raises(KeyError):
        store.playlist(999)

    store.delete_playlist(pl)
    assert store.playlists() == []
    with pytest.raises(KeyError):
        store.delete_playlist(pl.id)


# ---- entries -----------------------------------------------------------------------------


def test_entry_crud_and_default_duration_resolved_at_write_time(store):
    pl = store.create_playlist("P")
    e = store.add_entry(pl, "solid")
    assert e.duration_s == DEFAULT_SETTINGS["default_entry_duration"] == 60.0
    assert e.position == 0 and e.params == {} and e.enabled
    store.set_setting("default_entry_duration", 15)
    e2 = store.add_entry(pl, "solid", params={"level": 200})
    assert e2.duration_s == 15.0 and e2.position == 1 and e2.params == {"level": 200}
    assert store.entry(e.id).duration_s == 60.0  # existing rows untouched

    e3 = store.add_entry(pl, "solid", duration_s=5, position=0)  # insert at the front
    assert [x.animation_id for x in store.playlist(pl.id).entries] == ["solid"] * 3
    assert [x.id for x in store.playlist(pl.id).entries] == [e3.id, e.id, e2.id]
    assert [x.position for x in store.playlist(pl.id).entries] == [0, 1, 2]

    changed = store.update_entry(e, duration_s=7, enabled=False, params={"mode": "down"})
    assert (changed.duration_s, changed.enabled, changed.params) == (7.0, False, {"mode": "down"})
    assert store.playlist(pl.id).updated_at >= pl.updated_at
    with pytest.raises(ValueError):
        store.update_entry(e, duration_s=0)
    with pytest.raises(ValueError):
        store.add_entry(pl, "not a stem!")
    with pytest.raises(ValueError):
        store.add_entry(pl, "solid", duration_s=-1)
    with pytest.raises(ValueError):
        store.add_entry(pl, "solid", position=-1)
    with pytest.raises(KeyError):
        store.add_entry(999, "solid")


def test_move_entry_keeps_positions_contiguous(store):
    pl = store.create_playlist("P")
    ids = [store.add_entry(pl, "solid", duration_s=i + 1).id for i in range(5)]

    def order():
        entries = store.playlist(pl.id).entries
        assert [x.position for x in entries] == list(range(len(entries)))
        return [x.id for x in entries]

    store.move_entry(ids[4], 0)
    assert order() == [ids[4], ids[0], ids[1], ids[2], ids[3]]
    store.move_entry(ids[4], 4)
    assert order() == ids
    store.move_entry(ids[1], 3)
    assert order() == [ids[0], ids[2], ids[3], ids[1], ids[4]]
    store.move_entry(ids[1], 99)  # clamps to the end
    assert order() == [ids[0], ids[2], ids[3], ids[4], ids[1]]
    store.move_entry(ids[1], -5)  # clamps to the front
    assert order() == [ids[1], ids[0], ids[2], ids[3], ids[4]]
    store.remove_entry(ids[2])
    assert order() == [ids[1], ids[0], ids[3], ids[4]]
    store.remove_entry(ids[1])
    assert order() == [ids[0], ids[3], ids[4]]
    with pytest.raises(KeyError):
        store.remove_entry(ids[1])


def test_deleting_a_playlist_cascades_to_its_entries(store):
    pl = store.create_playlist("P")
    keep = store.create_playlist("K")
    e = store.add_entry(pl, "solid")
    store.add_entry(keep, "solid")
    store.delete_playlist(pl)
    with pytest.raises(KeyError):
        store.entry(e.id)
    assert len(store.playlist(keep.id).entries) == 1


# ---- params ----------------------------------------------------------------------------------


def test_params_json_stores_only_diffs_from_defaults(store):
    pl = store.create_playlist("P")
    e = store.add_entry(pl, "solid", params={"level": 10, "mode": "down"})
    assert e.params == {"mode": "down"}  # level == default, dropped
    with store._conn() as conn:  # noqa: SLF001
        raw = conn.execute("SELECT params_json FROM playlist_entry WHERE id = ?", (e.id,)).fetchone()[0]
    assert json.loads(raw) == {"mode": "down"}
    e = store.update_entry(e, params={"level": "42"})  # coerced through the spec
    assert e.params == {"level": 42}
    with pytest.raises(ValueError, match="not a parameter"):
        store.add_entry(pl, "solid", params={"brightness": 1})
    with pytest.raises(ValueError, match="above the maximum"):
        store.add_entry(pl, "solid", params={"level": 999})
    # an animation the registry does not know keeps its params verbatim
    ghost = store.add_entry(pl, "ghost", params={"anything": [1, 2]})
    assert ghost.params == {"anything": [1, 2]}


def test_out_of_range_params_clamp_and_warn_on_resolve(store, caplog):
    pl = store.create_playlist("P")
    e = store.add_entry(pl, "solid")
    # the spec changed under the stored value: write raw JSON past the store's validation
    with store._conn() as conn:  # noqa: SLF001
        conn.execute(
            "UPDATE playlist_entry SET params_json = ? WHERE id = ?",
            (json.dumps({"level": 999, "mode": "sideways", "gone": 1}), e.id),
        )
    with caplog.at_level("WARNING"):
        resolved = store.resolve(pl.id)
    [r] = resolved.entries
    assert not r.unresolved and r.playable
    assert r.params == {"level": 255, "mode": "up"}
    assert len(r.warnings) == 3
    assert any("clamped to 255" in w for w in r.warnings)
    assert any("using default 'up'" in w for w in r.warnings)
    assert any("'gone' is not a parameter" in w for w in r.warnings)
    assert "clamped" in caplog.text


def test_malformed_params_json_is_ignored_not_fatal(store):
    pl = store.create_playlist("P")
    e = store.add_entry(pl, "solid")
    with store._conn() as conn:  # noqa: SLF001
        conn.execute("UPDATE playlist_entry SET params_json = 'nope' WHERE id = ?", (e.id,))
    assert store.entry(e.id).params == {}


# ---- resolving ----------------------------------------------------------------------------------


def test_unknown_and_broken_animations_resolve_as_unresolved_and_survive(store, registry):
    pl = store.create_playlist("P")
    store.add_entry(pl, "solid", params={"level": 5})
    store.add_entry(pl, "vanished")
    store.add_entry(pl, "broken")
    store.add_entry(pl, "other", enabled=False)
    resolved = store.resolve("P")
    assert isinstance(resolved, ResolvedPlaylist)
    assert [r.unresolved for r in resolved.entries] == [False, True, True, False]
    assert resolved.entries[1].error == "no such animation"
    assert resolved.entries[2].error.startswith("failed to load: SyntaxError")
    assert resolved.entries[0].params == {"level": 5, "mode": "up"}
    assert resolved.entries[0].definition is registry["solid"]
    assert [r.entry.animation_id for r in resolved.playable] == ["solid"]  # disabled and unresolved skipped
    # nothing was deleted: fixing the file fixes the playlist
    assert [e.animation_id for e in store.playlist("P").entries] == ["solid", "vanished", "broken", "other"]
    (registry.paths[0] / "vanished.py").write_text(textwrap.dedent(SOLID))
    registry.reload()
    assert not store.resolve("P").entries[1].unresolved


def test_resolve_needs_a_registry():
    s = PlaylistStore(":memory:")
    pl = s.create_playlist("P")
    with pytest.raises(ValueError, match="Registry"):
        s.resolve(pl)
    s.close()


# ---- settings -----------------------------------------------------------------------------------


def test_settings_accessors_return_defaults_for_missing_and_malformed(store, caplog):
    assert store.get_int("brightness") == 255
    assert store.get_float("fps") == 30.0
    assert store.get_bool("enabled") is True
    assert store.get_str("startup_playlist") is None
    assert store.get_int("nothing", 7) == 7
    store.set_setting("brightness", 128)
    store.set_setting("fps", "25")
    store.set_setting("enabled", False)
    assert store.get_int("brightness") == 128 and store.get_float("fps") == 25.0
    assert store.get_bool("enabled") is False
    assert store.settings() == {"brightness": "128", "enabled": "0", "fps": "25"}
    store.set_setting("brightness", "bright")
    store.set_setting("enabled", "maybe")
    with caplog.at_level("WARNING"):
        assert store.get_int("brightness") == 255
        assert store.get_bool("enabled") is True
        assert store.get_int("brightness", 10) == 10
    assert "malformed" in caplog.text
    store.set_setting("brightness", None)
    assert store.get_setting("brightness") is None


# ---- play log ------------------------------------------------------------------------------------


def test_play_log(store):
    pl = store.create_playlist("P")
    a = store.log_play("solid", "completed", playlist_id=pl.id, duration_s=60)
    b = store.log_play("ghost", "skipped", started_at="2026-09-14T00:00:00+00:00")
    assert a.animation_id == "solid" and a.playlist_id == pl.id and a.duration_s == 60
    assert b.started_at == "2026-09-14T00:00:00+00:00" and b.playlist_id is None
    assert [x.id for x in store.play_log()] == [b.id, a.id]
    assert len(store.play_log(limit=1)) == 1
    with pytest.raises(ValueError):
        store.log_play("x", "exploded")


# ---- seeding ----------------------------------------------------------------------------------------


def test_seed_default_on_an_empty_database_only(store, registry):
    pl = store.seed_default()
    assert pl is not None and pl.name == "Default"
    assert [e.animation_id for e in pl.entries] == ["other", "solid"]  # what loaded, by id
    assert all(e.duration_s == 60.0 for e in pl.entries)
    assert store.startup_playlist() == pl
    assert store.seed_default() is None
    store.delete_playlist(pl)
    assert store.startup_playlist() is None  # gone, not an error


# ---- concurrency ------------------------------------------------------------------------------------


def test_web_thread_writes_while_render_thread_reads_under_wal(tmp_path, registry):
    path = tmp_path / "df2.sqlite3"
    store = PlaylistStore(path, registry=registry)
    pl = store.create_playlist("P")
    store.add_entry(pl, "solid")
    stop = threading.Event()
    read_errors: list[Exception] = []
    reads = [0]
    max_read_ms = [0.0]

    def render_thread():
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                got = store.playlist(pl.id)
                assert [e.position for e in got.entries] == list(range(len(got.entries)))
            except Exception as exc:  # noqa: BLE001
                read_errors.append(exc)
                return
            max_read_ms[0] = max(max_read_ms[0], (time.perf_counter() - t0) * 1000)
            reads[0] += 1

    reader = threading.Thread(target=render_thread)
    reader.start()
    writes = 0
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        e = store.add_entry(pl, "solid", duration_s=writes + 1)
        store.move_entry(e, 0)
        if writes % 3 == 0:
            store.remove_entry(e)
        writes += 1
    stop.set()
    reader.join()
    assert not read_errors
    assert writes > 20 and reads[0] > 20
    assert max_read_ms[0] < 100  # a read never waited on the writer's lock
    with store._conn() as conn:  # noqa: SLF001
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    store.close()
