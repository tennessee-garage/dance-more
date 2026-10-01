from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from df2_pi import cli
from df2_pi.animation import AnimationRegistry
from df2_pi.engine import Runner
from df2_pi.interfacing.dmx_control import WIDTH
from df2_pi.interfacing.external import EXTERNAL_ID
from df2_pi.interfacing.service import ExternalInput
from df2_pi.output import FanOut, NullSink, PreviewSink
from df2_pi.playlists import PlaylistStore
from df2_pi.web import AppContext, create_app


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    (tmp_path / "animations").mkdir()
    return AnimationRegistry.discover(tmp_path / "animations")


@pytest.fixture
def store(registry):
    s = PlaylistStore(":memory:", registry=registry)
    yield s
    s.close()


def build(registry, store, with_external=True):
    runner = MagicMock(spec=Runner)
    runner.state = Runner(registry, FanOut([NullSink()])).state
    external = ExternalInput(runner, registry, store, artnet_port=0, sacn_port=0, bind="127.0.0.1") if with_external else None
    preview = PreviewSink()
    ctx = AppContext(registry, store, FanOut([NullSink(), preview]), runner, preview, external)
    return TestClient(create_app(ctx)), external, runner


def test_settings_and_status_are_served(registry, store):
    client, _, _ = build(registry, store)
    body = client.get("/api/external").json()
    assert body["settings"] == {
        "artnet_enabled": True, "sacn_enabled": True, "mode": "tile", "grid_width": 34, "grid_height": 34,
        "artnet_universe": 0, "sacn_universe": 1, "source": "external", "mix": 0.5, "timeout_s": 2.0,
        "dmx_enabled": False, "dmx_artnet_universe": 0, "dmx_sacn_universe": 1, "dmx_address": 201,
    }
    dmx = body["status"]["dmx"]
    assert (dmx["live"], dmx["packets"], dmx["values"]) == (False, 0, None)
    assert dmx["channels"][0] == "Master dimmer" and len(dmx["channels"]) == WIDTH
    status = body["status"]
    assert (status["live"], status["applied"], status["universes"], status["frames"]) == (False, "internal", 1, 0)


def test_a_change_is_applied_now_and_stored(registry, store):
    client, external, runner = build(registry, store)
    body = client.patch("/api/external", json={"mode": "grid", "grid_width": 20, "grid_height": 17, "artnet_universe": 5, "source": "mix"}).json()
    assert body["settings"]["mode"] == "grid" and body["status"]["universes"] == 2  # 340 pixels
    assert external.receiver.layout.mode == "grid" and external.source.source == "mix"
    assert (store.get_str("external_mode"), store.get_int("artnet_universe"), store.get_str("external_source")) == ("grid", 5, "mix")
    # a fresh input starts the way it was left
    again = ExternalInput(runner, registry, store, artnet_port=0, sacn_port=0)
    assert (again.settings()["grid_width"], again.receiver.layout.universes) == (20, 2)


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "video"},
        {"mix": 2},
        {"timeout_s": 0},
        {"sacn_universe": 0},
        {"mode": "raw", "artnet_universe": 32760},  # 23 universes do not fit
        {"loud": True},
        {"mode": None},
        {"dmx_address": 498},  # the 16-channel block would run past 512
        {"dmx_sacn_universe": 0},
    ],
)
def test_a_bad_change_changes_nothing(registry, store, body):
    client, external, _ = build(registry, store)
    before = (external.settings(), store.settings())
    assert client.patch("/api/external", json=body).status_code == 422
    assert (external.settings(), store.settings()) == before


def test_a_bad_stored_value_falls_back_to_the_default(registry, store):
    store.set_setting("external_mode", "video")
    store.set_setting("external_mix", 0.25)
    external = ExternalInput(MagicMock(), registry, store, artnet_port=0, sacn_port=0)
    assert external.settings()["mode"] == "tile" and external.settings()["mix"] == 0.25


def test_the_external_animation_is_registered_and_not_seeded(registry, store):
    build(registry, store)
    assert registry.is_builtin(EXTERNAL_ID)
    store.seed_default(registry)
    assert all(e.animation_id != EXTERNAL_ID for e in store.playlists()[0].entries)


def test_without_the_receiver_the_api_says_so(registry, store):
    client, _, _ = build(registry, store, with_external=False)
    assert client.get("/api/external").status_code == 503


def test_led_positions_as_json_and_csv(registry, store):
    client, _, _ = build(registry, store)
    rows = client.get("/api/floor/leds").json()
    assert len(rows) == 3840
    assert rows[0] == {"index": 0, "tile": 0, "led": 0, "x": 0.5, "y": 1.5, "u": round(0.5 / 136, 6), "v": round(1.5 / 136, 6), "universe": 0, "channel": 1}
    assert (rows[170]["universe"], rows[170]["channel"], rows[171]["channel"]) == (1, 1, 4)
    csv = client.get("/api/floor/leds?format=csv")
    assert csv.headers["content-type"].startswith("text/csv")
    lines = csv.text.splitlines()
    assert lines[0] == "index,tile,led,x,y,u,v,universe,channel" and len(lines) == 3841


def test_the_cli_prints_the_led_table_and_serve_can_skip_the_receiver(capsys, tmp_path):
    assert cli.main(["leds"]) == 0
    assert capsys.readouterr().out.startswith("index,tile,led,x,y,u,v,universe,channel\n0,0,0,0.5,1.5,")
    args = cli.build_parser().parse_args(["--db", str(tmp_path / "x.sqlite3"), "--animations", str(tmp_path), "serve", "--no-hardware", "--no-external"])
    assert cli.build_app(args).state.ctx.external is None
