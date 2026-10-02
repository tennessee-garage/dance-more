import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from df2_pi.animation import AnimationRegistry
from df2_pi.engine import Runner
from df2_pi.gamma import from_linear, to_linear
from df2_pi.output import FanOut, NullSink, PreviewSink
from df2_pi.palette import BUILTIN, DEFAULT_ACTIVE, FLOOR, LIBRARY, Palette, PaletteBook, choice, palette_param
from df2_pi.playlists import PlaylistStore
from df2_pi.web import AppContext, create_app


# ---- Palette --------------------------------------------------------------------------------------


def test_stops_land_on_their_evenly_spaced_places():
    pal = Palette(["ff0000", "00ff00", "0000ff", "ffffff"])
    assert [pal.at(k / 4).tolist() for k in range(4)] == [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]


def test_blending_is_in_linear_light():
    pal = Palette(["ff0000", "00ff00"])
    middle = pal.at(0.25)  # halfway from red to green
    half = int(from_linear(np.array([0.5]))[0])  # half the light of a channel at full
    assert middle.tolist() == [half, half, 0] and half > 160  # a bright yellow, not 128's muddy one
    assert to_linear(middle.astype(np.uint8)).sum() == pytest.approx(1.0, abs=0.01)  # as much light as either stop


def test_u_wraps_and_the_last_stop_blends_back_to_the_first():
    pal = Palette(["ff0000", "0000ff"])
    assert pal.at(1.0).tolist() == pal.at(0.0).tolist() == [255, 0, 0]
    assert pal.at(-0.5).tolist() == pal.at(0.5).tolist() == [0, 0, 255]
    assert pal.at(0.75).tolist() == pal.at(0.25).tolist()  # red->blue and blue->red meet halfway


def test_arrays_in_colours_out():
    pal = BUILTIN["fire"]
    out = pal.at(np.zeros((8, 8)))
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8
    assert (out == pal.stops[0]).all()


def test_stops_are_hex_or_rgb_and_two_to_eight_of_them():
    assert Palette(["#FF8000", (0, 128, 255)]).hex() == ["ff8000", "0080ff"]
    for bad in (["ff0000"], ["ff0000"] * 9, ["red", "blue"], [(256, 0, 0), (0, 0, 0)], [(1, 2), (3, 4)]):
        with pytest.raises(ValueError):
            Palette(bad)


def test_every_builtin_is_valid_and_rainbow_is_the_default():
    assert set(BUILTIN) == set(LIBRARY) and DEFAULT_ACTIVE == "rainbow"
    assert all(2 <= len(p.stops) <= 8 for p in BUILTIN.values())


def test_the_param_and_choice():
    spec = palette_param()
    assert spec.default == FLOOR and list(spec.choices) == [FLOOR, *LIBRARY]
    ctx = MagicMock()
    ctx.palette = Palette(["000000", "ffffff"], "mine")
    assert choice(ctx, FLOOR) is ctx.palette
    assert choice(ctx, "ice") is BUILTIN["ice"]


# ---- PaletteBook ----------------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    (tmp_path / "animations").mkdir()
    return AnimationRegistry.discover(tmp_path / "animations")


@pytest.fixture
def store(registry):
    s = PlaylistStore(":memory:", registry=registry)
    yield s
    s.close()


def test_a_fresh_book_is_the_library_with_rainbow_active_and_tells_the_runner(store):
    runner = MagicMock()
    book = PaletteBook(store, runner)
    assert book.active == "rainbow" and book.names() == list(LIBRARY)
    runner.set_palette.assert_called_once_with(BUILTIN["rainbow"])


def test_user_palettes_and_the_active_one_survive_a_restart(store):
    book = PaletteBook(store)
    book.save("club", ["ff0040", "4000ff", "00ffc0"])
    book.activate("club")
    runner = MagicMock()
    again = PaletteBook(store, runner)
    assert again.active == "club" and again.names()[-1] == "club"
    assert runner.set_palette.call_args.args[0].hex() == ["ff0040", "4000ff", "00ffc0"]


def test_user_palettes_follow_the_builtins_alphabetically(store):
    book = PaletteBook(store)
    book.save("zebra", ["000000", "ffffff"])
    book.save("amber", ["ffb000", "ff6000"])
    assert book.names() == [*LIBRARY, "amber", "zebra"]


def test_saving_the_active_palette_updates_the_floor(store):
    runner = MagicMock()
    book = PaletteBook(store, runner)
    book.save("mine", ["ff0000", "00ff00"])
    book.activate("mine")
    book.save("mine", ["0000ff", "ffffff"])
    assert runner.set_palette.call_args.args[0].hex() == ["0000ff", "ffffff"]


def test_builtins_are_read_only_and_names_are_checked(store):
    book = PaletteBook(store)
    for name in ("fire", "floor", "Has Spaces", "", "x" * 40):
        with pytest.raises(ValueError):
            book.save(name, ["000000", "ffffff"])
    with pytest.raises(ValueError):
        book.delete("rainbow")
    with pytest.raises(KeyError):
        book.activate("nope")


def test_deleting_the_active_palette_goes_back_to_the_default(store):
    runner = MagicMock()
    book = PaletteBook(store, runner)
    book.save("mine", ["ff0000", "00ff00"])
    book.activate("mine")
    book.delete("mine")
    assert book.active == "rainbow" and "mine" not in book.names()
    assert runner.set_palette.call_args.args[0] == BUILTIN["rainbow"]


def test_a_bad_stored_palette_is_skipped_not_fatal(store):
    store.set_setting("user_palettes", json.dumps({"good": ["ff0000", "00ff00"], "bad": ["nope"], "fire": ["000000", "111111"]}))
    store.set_setting("active_palette", "gone")
    book = PaletteBook(store)
    assert book.names()[-1] == "good" and "bad" not in book.names()
    assert book.get("fire") == BUILTIN["fire"] and book.active == "rainbow"


# ---- the runner and the API -----------------------------------------------------------------------


def build(registry, store, with_book=True):
    runner = MagicMock(spec=Runner)
    runner.state = Runner(registry, FanOut([NullSink()])).state
    book = PaletteBook(store, runner) if with_book else None
    preview = PreviewSink()
    ctx = AppContext(registry, store, FanOut([NullSink(), preview]), runner, preview, None, None, book)
    return TestClient(create_app(ctx)), book, runner


def test_the_api_lists_activates_saves_and_deletes(registry, store):
    client, book, runner = build(registry, store)
    body = client.get("/api/palettes").json()
    assert body["active"] == "rainbow" and [p["name"] for p in body["palettes"]] == list(LIBRARY)
    assert all(p["builtin"] for p in body["palettes"])

    body = client.put("/api/palettes/club", json={"stops": ["ff0040", "4000ff"]}).json()
    assert body["palettes"][-1] == {"name": "club", "stops": ["ff0040", "4000ff"], "builtin": False}
    assert client.post("/api/palettes/active", json={"name": "club"}).json()["active"] == "club"
    assert runner.set_palette.call_args.args[0].name == "club"
    assert client.delete("/api/palettes/club").json()["active"] == "rainbow"


def test_the_api_refuses_what_the_book_refuses(registry, store):
    client, _, _ = build(registry, store)
    assert client.put("/api/palettes/fire", json={"stops": ["000000", "ffffff"]}).status_code == 422
    assert client.put("/api/palettes/mine", json={"stops": ["000000"]}).status_code == 422
    assert client.put("/api/palettes/mine", json={"stops": ["zzzzzz", "000000"]}).status_code == 422
    assert client.delete("/api/palettes/rainbow").status_code == 422
    assert client.delete("/api/palettes/nope").status_code == 404
    assert client.post("/api/palettes/active", json={"name": "nope"}).status_code == 404


def test_without_a_book_the_api_answers_503(registry, store):
    client, _, _ = build(registry, store, with_book=False)
    assert client.get("/api/palettes").status_code == 503
