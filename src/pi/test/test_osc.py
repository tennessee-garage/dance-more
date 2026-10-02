import socket
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("pythonosc", reason="OSC needs the [web] extra")

from pythonosc.udp_client import SimpleUDPClient  # noqa: E402

from df2_pi.animation import AnimationRegistry, BeatInfo, Param, animation  # noqa: E402
from df2_pi.engine import Runner  # noqa: E402
from df2_pi.interfacing.osc import SUBSCRIPTION_S, OscControl, source_from_unit, speed_from_unit  # noqa: E402
from df2_pi.output import FanOut, NullSink, PreviewSink  # noqa: E402
from df2_pi.playlists import PlaylistStore  # noqa: E402
from df2_pi.web import AppContext, create_app  # noqa: E402

SOLID = """
from df2_pi.animation import animation, Param
from df2_pi.pixels import TileFrame

@animation(name="{name}", params={{"speed": Param(float, default=1.0, min=0.5, max=4.5), "mode": Param(str, default="a", choices=["a", "b", "c"])}})
def render(previous, ctx):
    return TileFrame.black(ctx.geometry)
"""


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    d = tmp_path / "animations"
    d.mkdir()
    for name in ("waves", "stardust"):
        (d / f"{name}.py").write_text(SOLID.format(name=name.title()))
    return AnimationRegistry.discover(d)


@pytest.fixture
def store(registry):
    s = PlaylistStore(":memory:", registry=registry)
    yield s
    s.close()


def state(**changes):
    s = MagicMock()
    s.show.strobe_max_hz = 10.0
    s.animation = ("waves", "Waves")
    s.playlist = (1, "Party")
    s.entry_index = 2
    s.palette = "ocean"
    s.brightness = 255
    s.blacked_out = False
    s.timer_held = False
    s.beat = None
    for k, v in changes.items():
        setattr(s, k, v)
    return s


@pytest.fixture
def rig(registry, store):
    runner = MagicMock()
    runner.state = state()
    beat, palettes, external = MagicMock(), MagicMock(), MagicMock()
    palettes.names.return_value = ["rainbow", "fire", "ice", "ocean"]
    clock = Clock()
    sent = []
    osc = OscControl(runner, registry, store=store, beat=beat, palettes=palettes, external=external, now=clock, send=lambda t, a, v: sent.append((t, a, v)))
    return osc, runner, beat, palettes, external, clock, sent


# ---- every address to the call it makes -----------------------------------------------------------


@pytest.mark.parametrize(
    "address, args, call, expected",
    [
        ("/floor/brightness", [0.5], "set_brightness", (128,)),
        ("/floor/brightness", [2.0], "set_brightness", (255,)),  # clamped
        ("/floor/blackout", [1], "blackout", ()),
        ("/floor/blackout", [0], "unblackout", ()),
        ("/floor/speed", [0.5], "set_speed", (1.0,)),
        ("/floor/speed", [1.0], "set_speed", (4.0,)),
        ("/floor/strobe", [0.5], "set_strobe", (5.0,)),
        ("/floor/bump", [], "bump", (1.0, 0.25)),
        ("/floor/bump", [0.6], "bump", (0.6, 0.25)),
        ("/floor/tint", [1.0, 0.5, 0.0, 0.8], "set_tint", (255, 128, 0, 0.8)),
        ("/floor/hue", [0.25], "set_hue_shift", (0.25,)),
        ("/floor/saturation", [0.5], "set_saturation", (1.0,)),
        ("/floor/freeze", [1], "freeze", (True,)),
        ("/floor/hold", [0.9], "hold", (True,)),
        ("/floor/hold", [0], "hold", (False,)),
        ("/floor/reset", [], "reset_show", ()),
        ("/floor/next", [], "next", ()),
        ("/floor/previous", [1], "previous", ()),
        ("/floor/restart", [1.0], "restart", ()),
        ("/floor/goto", [3], "goto", (3,)),
        ("/floor/goto/5", [1], "goto", (5,)),
        ("/floor/play", ["stardust"], "play_animation", ("stardust",)),
        ("/floor/play/waves", [1], "play_animation", ("waves",)),
        ("/floor/macro/2", [0.75], "set_control", ("macro2", 0.75)),
        ("/floor/trigger", [3, 0.5], "trigger", (3, 0.5)),
        ("/floor/trigger", [4], "trigger", (4, 1.0)),
        ("/floor/trigger/7", [0.8], "trigger", (7, 0.8)),
        ("/floor/trigger/7", [], "trigger", (7, 1.0)),
    ],
)
def test_runner_addresses(rig, address, args, call, expected):
    osc, runner, *_ = rig
    osc.handle(address, *args)
    getattr(runner, call).assert_called_once_with(*expected)
    assert osc.errors == 0


def test_param_maps_through_the_param_spec(rig):
    osc, runner, *_ = rig
    osc.handle("/floor/param/speed", 0.5)  # 0.5..4.5, linear: halfway is 2.5
    runner.set_params.assert_called_once_with(speed=2.5)
    osc.handle("/floor/param/mode", 1.0)
    runner.set_params.assert_called_with(mode="c")
    osc.handle("/floor/param/nope", 0.5)
    assert osc.errors == 1 and "no parameter" in osc.status()["recent"][0]["error"]


def test_source_mix_tempo_and_palette(rig):
    osc, runner, beat, palettes, external, *_ = rig
    osc.handle("/floor/mix", 0.3)
    external.source.set_mix.assert_called_once_with(0.3)
    osc.handle("/floor/source", 0.9)
    osc.handle("/floor/source", "internal")
    assert [c.args for c in external.source.set_source.call_args_list] == [("mix",), ("internal",)]
    osc.handle("/floor/tempo/tap")
    osc.handle("/floor/tempo/resync", 1)
    osc.handle("/floor/tempo/nudge", -10)
    beat.tap.assert_called_once_with()
    beat.resync.assert_called_once_with()
    beat.nudge.assert_called_once_with(-10.0)
    osc.handle("/floor/palette", 1)  # by its place in the list
    osc.handle("/floor/palette", "ice")
    osc.handle("/floor/palette/ocean", 1)
    osc.handle("/floor/palette", 9)  # no such palette
    assert [c.args for c in palettes.activate.call_args_list] == [("fire",), ("ice",), ("ocean",)]
    assert osc.errors == 1


def test_playlists_load_by_id_or_name(rig, store):
    osc, runner, *_ = rig
    party = store.create_playlist("Party")
    osc.handle("/floor/playlist", "Party")
    osc.handle("/floor/playlist", party.id)
    osc.handle(f"/floor/playlist/{party.id}", 1)
    assert runner.load_playlist.call_count == 3
    assert all(c.args[0].playlist.name == "Party" for c in runner.load_playlist.call_args_list)
    osc.handle("/floor/playlist", "Nope")
    assert osc.errors == 1


def test_mappings_agree_with_the_dmx_faders():
    assert [speed_from_unit(u) for u in (0.0, 0.25, 0.5, 1.0)] == [0.0, 0.5, 1.0, 4.0]
    assert [source_from_unit(u) for u in (0.0, 0.4, 0.9)] == ["internal", "external", "mix"]


# ---- presses ---------------------------------------------------------------------------------------


def test_a_press_fires_once_per_rise_and_ignores_the_release(rig):
    osc, runner, *_, clock, _ = rig
    for value in (1, 0, 1, 0):
        osc.handle("/floor/next", value)
        clock.t += 0.05
    assert runner.next.call_count == 2


def test_a_held_button_repeating_1_fires_once_but_separate_presses_of_1_each_fire(rig):
    osc, runner, *_, clock, _ = rig
    for _ in range(30):  # 1 every frame for a second: one press
        osc.handle("/floor/play/waves", 1.0)
        clock.t += 1 / 30
    assert runner.play_animation.call_count == 1
    clock.t += 1.0
    osc.handle("/floor/play/waves", 1.0)  # a clip launched again, a second later, no release between
    assert runner.play_animation.call_count == 2


def test_triggers_ignore_a_zero_velocity(rig):
    osc, runner, *_ = rig
    osc.handle("/floor/trigger/2", 0.0)
    osc.handle("/floor/trigger", 2, 0)
    runner.trigger.assert_not_called()


def test_bad_messages_are_counted_not_raised(rig):
    osc, runner, *_ = rig
    for address, args in (("/other/thing", []), ("/floor/nope", []), ("/floor/brightness", []), ("/floor/tint", [1, 2]), ("/floor/play", ["missing"])):
        osc.handle(address, *args)
    assert osc.errors == 5 and osc.received == 5
    runner.play_animation.assert_not_called()


# ---- feedback ---------------------------------------------------------------------------------------


def test_a_subscriber_gets_everything_then_only_changes_and_lapses(rig):
    osc, runner, *_, clock, sent = rig
    osc.handle("/floor/subscribe", 9000, sender=("10.0.0.5", 55555))
    first = {a: v for _, a, v in sent}
    assert first["/floor/state/animation"] == "Waves" and first["/floor/state/entry"] == 2 and first["/floor/state/palette"] == "ocean"
    assert all(t == ("10.0.0.5", 9000) for t, _, _ in sent)
    sent.clear()
    osc.push()
    assert sent == []  # nothing changed
    runner.state = state(palette="fire", beat=BeatInfo(tempo=128.0, phase=0.2, beat=17, bar_phase=0.3, beats_per_bar=4, downbeat=False))
    osc.push()
    assert {a: v for _, a, v in sent} == {"/floor/state/palette": "fire", "/floor/state/tempo": 128.0, "/floor/state/beat": 17}
    sent.clear()
    clock.t += SUBSCRIPTION_S + 1
    runner.state = state(palette="ice")
    osc.push()
    assert sent == [] and osc.status()["subscribers"] == []


def test_renewing_keeps_a_subscription(rig):
    osc, runner, *_, clock, sent = rig
    osc.handle("/floor/subscribe", 9000, sender=("10.0.0.5", 1))
    clock.t += SUBSCRIPTION_S - 5
    osc.handle("/floor/subscribe", 9000, sender=("10.0.0.5", 1))
    clock.t += SUBSCRIPTION_S - 5
    assert osc.status()["subscribers"] == ["10.0.0.5:9000"]


# ---- over the wire ---------------------------------------------------------------------------------


def test_real_udp_in_and_feedback_out(registry, store):
    runner = MagicMock()
    runner.state = state()
    osc = OscControl(runner, registry, store=store)
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(2.0)
    osc.start(0, bind="127.0.0.1")
    try:
        client = SimpleUDPClient("127.0.0.1", osc.port)
        client.send_message("/floor/brightness", 0.5)
        client.send_message("/floor/subscribe", listener.getsockname()[1])
        end = time.monotonic() + 2.0
        while runner.set_brightness.call_count == 0 and time.monotonic() < end:
            time.sleep(0.01)
        runner.set_brightness.assert_called_once_with(128)
        packet, _ = listener.recvfrom(1024)
        assert packet.startswith(b"/floor/state/")
    finally:
        osc.stop()
        listener.close()


# ---- settings ---------------------------------------------------------------------------------------


def test_settings_api_restarts_and_stores(registry, store):
    runner = MagicMock(spec=Runner)
    runner.state = Runner(registry, FanOut([NullSink()])).state
    osc = OscControl(runner, registry, store=store)
    preview = PreviewSink()
    client = TestClient(create_app(AppContext(registry, store, FanOut([NullSink(), preview]), runner, preview, None, None, None, osc)))
    body = client.get("/api/osc").json()
    assert body["settings"] == {"enabled": True, "port": 7000}
    body = client.patch("/api/osc", json={"enabled": False}).json()
    assert body["settings"]["enabled"] is False and body["status"]["listening"] is False
    assert store.get_bool("osc_enabled") is False
    assert client.patch("/api/osc", json={"port": 70000}).status_code == 422
    assert OscControl(runner, registry, store=store).settings()["enabled"] is False  # survives a restart
