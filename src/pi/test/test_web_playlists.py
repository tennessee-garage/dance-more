import dataclasses
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from df2_pi.animation import AnimationRegistry
from df2_pi.engine import Runner
from df2_pi.output import FanOut, NullSink, PreviewSink
from df2_pi.playlists import PlaylistStore
from df2_pi.web import AppContext, create_app

SOLID = '''
from df2_pi.animation import animation, Param
from df2_pi.pixels import TileFrame

@animation(name="Solid", period=4.0, params={
    "level": Param(int, default=10, min=0, max=255),
    "mode": Param(str, default="up", choices=["up", "down"]),
})
def render(previous, ctx):
    return TileFrame.black(ctx.geometry)
'''

PIXEL = '''
from df2_pi.animation import animation
from df2_pi.pixels import PixelFrame

@animation(name="Pixel", format="pixel")
def render(previous, ctx):
    return PixelFrame.black(ctx.geometry)
'''


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    d = tmp_path / "animations"
    d.mkdir()
    (d / "solid.py").write_text(textwrap.dedent(SOLID))
    (d / "pixel.py").write_text(textwrap.dedent(PIXEL))
    (d / "broken.py").write_text("def render(:\n")
    return AnimationRegistry.discover(d)


@pytest.fixture
def store(registry) -> PlaylistStore:
    s = PlaylistStore(":memory:", registry=registry)
    yield s
    s.close()


@pytest.fixture
def runner(registry):
    runner = MagicMock(spec=Runner)
    runner.state = Runner(registry, FanOut([NullSink()])).state  # nothing loaded
    return runner


@pytest.fixture
def client(registry, store, runner) -> TestClient:
    preview = PreviewSink()
    return TestClient(create_app(AppContext(registry, store, FanOut([NullSink(), preview]), runner, preview)))


def make(client, name="Party", **fields) -> dict:
    response = client.post("/api/playlists", json={"name": name, **fields})
    assert response.status_code == 201, response.text
    return response.json()


def add(client, pid, animation_id="solid", **fields) -> dict:
    response = client.post(f"/api/playlists/{pid}/entries", json={"animation_id": animation_id, **fields})
    assert response.status_code == 201, response.text
    return response.json()


def positions(playlist: dict) -> list[int]:
    return [e["position"] for e in playlist["entries"]]


# ---- playlists ------------------------------------------------------------------------------


def test_create_read_update_delete(client):
    created = make(client, "Party", description="Friday", loop=False, shuffle=True, crossfade_s=1.5)
    assert {k: created[k] for k in ("name", "description", "loop", "shuffle", "crossfade_s", "entry_count", "startup", "loaded")} == {
        "name": "Party", "description": "Friday", "loop": False, "shuffle": True, "crossfade_s": 1.5,
        "entry_count": 0, "startup": False, "loaded": False,
    }
    pid = created["id"]
    assert client.get(f"/api/playlists/{pid}").json()["name"] == "Party"

    updated = client.patch(f"/api/playlists/{pid}", json={"name": "Saturday", "loop": True}).json()
    assert (updated["name"], updated["loop"], updated["shuffle"]) == ("Saturday", True, True)  # unsent fields kept

    assert client.delete(f"/api/playlists/{pid}").status_code == 204
    assert client.get(f"/api/playlists/{pid}").status_code == 404


def test_list_summarises_each_playlist_by_name(client):
    b = make(client, "B")
    make(client, "A")
    add(client, b["id"], duration_s=30)
    add(client, b["id"], "pixel", duration_s=12.5)
    listed = client.get("/api/playlists").json()
    assert [p["name"] for p in listed] == ["A", "B"]
    assert (listed[1]["entry_count"], listed[1]["total_duration_s"]) == (2, 42.5)


def test_startup_marker_and_deleting_the_startup_playlist_clears_it(client, store):
    a = make(client, "A")
    b = make(client, "B")
    assert client.post(f"/api/playlists/{b['id']}/startup").json()["startup"] is True
    assert [p["startup"] for p in client.get("/api/playlists").json()] == [False, True]
    assert client.get("/api/settings").json()["startup_playlist"] == b["id"]
    client.delete(f"/api/playlists/{b['id']}")
    assert client.get("/api/settings").json()["startup_playlist"] is None
    assert client.post(f"/api/playlists/{a['id'] + 100}/startup").status_code == 404


# ---- entries --------------------------------------------------------------------------------


def test_entries_add_update_move_remove_keep_positions_contiguous(client):
    pid = make(client)["id"]
    add(client, pid, duration_s=10)
    add(client, pid, "pixel", duration_s=20)
    pl = add(client, pid, duration_s=30, position=1)  # inserted, the rest shift down
    assert [e["duration_s"] for e in pl["entries"]] == [10, 30, 20] and positions(pl) == [0, 1, 2]

    first, middle, last = (e["id"] for e in pl["entries"])
    pl = client.post(f"/api/playlists/{pid}/entries/{first}/move", json={"position": 2}).json()
    assert [e["id"] for e in pl["entries"]] == [middle, last, first] and positions(pl) == [0, 1, 2]
    pl = client.post(f"/api/playlists/{pid}/entries/{first}/move", json={"position": 99}).json()  # clamped
    assert [e["id"] for e in pl["entries"]][-1] == first

    pl = client.patch(f"/api/playlists/{pid}/entries/{middle}", json={"duration_s": 45, "enabled": False}).json()
    entry = next(e for e in pl["entries"] if e["id"] == middle)
    assert (entry["duration_s"], entry["enabled"], entry["playable"]) == (45, False, False)

    pl = client.delete(f"/api/playlists/{pid}/entries/{middle}").json()
    assert [e["id"] for e in pl["entries"]] == [last, first] and positions(pl) == [0, 1]


def test_duration_omitted_takes_the_default_setting(client):
    client.patch("/api/settings", json={"default_entry_duration": 75})
    pl = add(client, make(client)["id"])
    assert pl["entries"][0]["duration_s"] == 75


def test_params_store_only_diffs_and_resolve_in_full(client):
    pid = make(client)["id"]
    pl = add(client, pid, params={"level": 10, "mode": "down"})  # level is the default
    entry = pl["entries"][0]
    assert entry["params"] == {"mode": "down"}
    assert entry["resolved_params"] == {"level": 10, "mode": "down"}
    assert entry["animation"] == {"name": "Solid", "format": "tile", "period": 4.0}

    pl = client.patch(f"/api/playlists/{pid}/entries/{entry['id']}", json={"params": {"level": 99}}).json()
    assert pl["entries"][0]["params"] == {"level": 99}  # replaced wholesale, not merged


def test_changing_the_animation_clears_the_old_overrides(client):
    pid = make(client)["id"]
    entry = add(client, pid, params={"level": 99})["entries"][0]
    pl = client.patch(f"/api/playlists/{pid}/entries/{entry['id']}", json={"animation_id": "pixel"}).json()
    assert (pl["entries"][0]["animation_id"], pl["entries"][0]["params"]) == ("pixel", {})


def test_unresolved_entries_are_served_with_the_error_not_dropped(client):
    pid = make(client)["id"]
    add(client, pid, "broken")
    pl = add(client, pid, "gone", params={"anything": 1})
    broken, gone = pl["entries"]
    assert broken["unresolved"] and broken["animation"] is None and not broken["playable"]
    assert "failed to load" in broken["error"] and "SyntaxError" in broken["error"]
    assert gone["unresolved"] and gone["error"] == "no such animation"
    assert gone["params"] == {"anything": 1}  # an unknown animation keeps what it was given
    assert client.get("/api/playlists").json()[0]["entry_count"] == 2


def test_an_entry_of_another_playlist_is_404(client):
    a = make(client, "A")["id"]
    b = make(client, "B")["id"]
    entry = add(client, a)["entries"][0]["id"]
    for method, path, body in [
        ("patch", f"/api/playlists/{b}/entries/{entry}", {"duration_s": 5}),
        ("delete", f"/api/playlists/{b}/entries/{entry}", None),
        ("post", f"/api/playlists/{b}/entries/{entry}/move", {"position": 0}),
    ]:
        response = client.request(method.upper(), path, json=body)
        assert response.status_code == 404, (method, path, response.text)
    assert client.get(f"/api/playlists/{a}").json()["entry_count"] == 1


# ---- error mapping --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, path, body, status, fragment",
    [
        ("get", "/api/playlists/999", None, 404, "999"),
        ("patch", "/api/playlists/999", {"loop": False}, 404, "999"),
        ("delete", "/api/playlists/999", None, 404, "999"),
        ("post", "/api/playlists/999/entries", {"animation_id": "solid"}, 404, "999"),
        ("patch", "/api/playlists/{pid}/entries/999", {"duration_s": 5}, 404, "999"),
        ("post", "/api/playlists", {"name": "Party"}, 409, "already exists"),
        ("patch", "/api/playlists/{other}", {"name": "Party"}, 409, "already exists"),
        ("post", "/api/playlists", {"name": "   "}, 422, "must not be empty"),
        ("post", "/api/playlists", {"name": "X", "crossfade_s": -1}, 422, "greater than or equal to 0"),
        ("post", "/api/playlists/{pid}/entries", {"animation_id": "solid", "params": {"level": 300}}, 422, "above the maximum"),
        ("post", "/api/playlists/{pid}/entries", {"animation_id": "solid", "params": {"speed": 1}}, 422, "not a parameter"),
        ("post", "/api/playlists/{pid}/entries", {"animation_id": "not-an-id"}, 422, "module stem"),
        ("post", "/api/playlists/{pid}/entries", {"animation_id": "solid", "duration_s": 0}, 422, "greater than 0"),
        ("post", "/api/playlists/{pid}/entries/{entry}/move", {"position": -1}, 422, "greater than or equal to 0"),
    ],
)
def test_store_errors_map_to_status_codes(client, method, path, body, status, fragment):
    pid = make(client, "Party")["id"]
    other = make(client, "Other")["id"]
    entry = add(client, pid)["entries"][0]["id"]
    before = client.get(f"/api/playlists/{pid}").json()
    response = client.request(method.upper(), path.format(pid=pid, other=other, entry=entry), json=body)
    assert response.status_code == status, response.text
    assert fragment in response.text
    assert client.get(f"/api/playlists/{pid}").json()["entries"] == before["entries"]  # nothing half-written


# ---- the loaded playlist ----------------------------------------------------------------------


def test_editing_the_loaded_playlist_says_so(client, runner):
    loaded = make(client, "Loaded")["id"]
    other = make(client, "Other")["id"]
    runner.state = dataclasses.replace(runner.state, playlist=(loaded, "Loaded"))
    assert add(client, loaded)["loaded"] is True
    assert add(client, other)["loaded"] is False
    assert client.patch(f"/api/playlists/{loaded}", json={"loop": False}).json()["loaded"] is True
    assert [p["loaded"] for p in client.get("/api/playlists").json()] == [True, False]
    runner.load_playlist.assert_not_called()  # the database only: the runner keeps its copy


# ---- settings -------------------------------------------------------------------------------


def test_settings_round_trip(client):
    assert client.get("/api/settings").json() == {
        "brightness": 255, "rotation": 0, "strobe_max_hz": 10.0, "default_entry_duration": 60.0, "startup_playlist": None,
    }
    pid = make(client)["id"]
    changed = client.patch(
        "/api/settings",
        json={"brightness": 128, "rotation": 270, "strobe_max_hz": 6, "default_entry_duration": 90, "startup_playlist": pid},
    ).json()
    assert changed == {
        "brightness": 128, "rotation": 270, "strobe_max_hz": 6.0, "default_entry_duration": 90.0, "startup_playlist": pid,
    }
    assert client.get("/api/settings").json() == changed
    assert client.patch("/api/settings", json={"startup_playlist": None}).json()["startup_playlist"] is None


def test_a_new_brightness_setting_is_applied_to_the_floor_now(client, runner):
    client.patch("/api/settings", json={"brightness": 90})
    runner.set_brightness.assert_called_once_with(90)
    client.patch("/api/settings", json={"default_entry_duration": 30})
    runner.set_brightness.assert_called_once()  # untouched when brightness is not sent


def test_a_new_rotation_setting_is_applied_to_the_floor_now(client, runner):
    client.patch("/api/settings", json={"rotation": 90})
    runner.set_rotation.assert_called_once_with(90)
    client.patch("/api/settings", json={"brightness": 30})
    runner.set_rotation.assert_called_once()  # untouched when rotation is not sent


def test_a_new_strobe_cap_is_applied_to_the_floor_now(client, runner):
    client.patch("/api/settings", json={"strobe_max_hz": 4})
    runner.set_strobe_max.assert_called_once_with(4.0)


def test_an_invalid_stored_strobe_cap_reads_as_the_default(client, store):
    store.set_setting("strobe_max_hz", 99)
    assert client.get("/api/settings").json()["strobe_max_hz"] == 10.0


def test_an_invalid_stored_rotation_reads_as_zero(client, store):
    store.set_setting("rotation", 45)
    assert client.get("/api/settings").json()["rotation"] == 0


@pytest.mark.parametrize(
    "body",
    [
        {"brightness": 256},
        {"brightness": "loud"},
        {"default_entry_duration": -5},
        {"startup_playlist": 999},
        {"brightness": 100, "default_entry_duration": -1},  # one bad value: nothing written
        {"brightness": None},
        {"rotation": 45},
        {"rotation": -90},
        {"rotation": "90"},
        {"rotation": None},
        {"strobe_max_hz": 16},
        {"strobe_max_hz": -1},
        {"default_entry_duration": None},
        {"fps": 25},  # not a setting here: refused, not ignored
    ],
)
def test_malformed_settings_never_reach_the_store(client, store, runner, body):
    before = store.settings()
    response = client.patch("/api/settings", json=body)
    assert response.status_code == 422, response.text
    assert store.settings() == before
    runner.set_brightness.assert_not_called()
    runner.set_rotation.assert_not_called()
