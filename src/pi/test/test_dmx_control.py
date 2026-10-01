import socket
import time
from pathlib import Path
from unittest.mock import MagicMock
from xml.etree import ElementTree

import pytest

from df2_pi.interfacing.dmx_control import CHANNELS, WIDTH, DmxControl, qlc_fixture, source_of, speed_of
from df2_pi.interfacing.packets import artdmx
from df2_pi.interfacing.receiver import Layout, Receiver

FIXTURE = Path(__file__).resolve().parents[3] / "docs" / "fixtures" / "dance-floor-v2.qxf"


class Clock:
    def __init__(self) -> None:
        self.t = 50.0

    def __call__(self) -> float:
        return self.t


def block(**values) -> bytes:
    """A control block: every channel at 0 except those named (by offset)."""
    data = bytearray(WIDTH)
    names = {"dimmer": 0, "strobe": 1, "source": 2, "mix": 3, "bank": 4, "program": 5, "speed": 6,
             "macro1": 7, "macro2": 8, "red": 11, "green": 12, "blue": 13, "tint": 14, "bump": 15, "hold": 16}
    for name, value in values.items():
        data[names[name]] = value
    return bytes(data)


@pytest.fixture
def rig():
    runner = MagicMock()
    runner.state.show.strobe_max_hz = 10.0
    runner.state.playlist = None
    source = MagicMock()
    store = MagicMock()
    store.get_int.return_value = 180  # the stored brightness
    store.get_str.return_value = "external"
    store.get_float.return_value = 0.5
    clock = Clock()
    return DmxControl(runner, source, store, timeout_s=2.0, now=clock), runner, source, store, clock


def test_the_mappings():
    assert [speed_of(v) for v in (0, 64, 128, 255)] == [0.0, 0.5, 1.0, 4.0]
    assert [source_of(v) for v in (0, 84, 85, 169, 170, 255)] == ["internal", "internal", "external", "external", "mix", "mix"]


def test_the_first_packet_sets_continuous_controls_but_fires_no_triggers(rig):
    control, runner, source, _, _ = rig
    control.handle(block(dimmer=200, strobe=51, source=200, mix=128, speed=128, red=255, tint=255, bank=1, program=2, macro1=90, bump=255))
    runner.set_brightness.assert_called_once_with(200)
    runner.set_strobe.assert_called_once_with(pytest.approx(2.0))  # 51/255 of the 10 Hz cap
    source.set_source.assert_called_once_with("mix")
    source.set_mix.assert_called_once_with(pytest.approx(128 / 255))
    runner.set_speed.assert_called_once_with(1.0)
    runner.set_tint.assert_called_once_with(255, 0, 0, 1.0)
    runner.load_playlist.assert_not_called()
    runner.goto.assert_not_called()
    runner.set_control.assert_not_called()
    runner.bump.assert_not_called()


def test_only_changed_channels_act_after_that(rig):
    control, runner, source, _, _ = rig
    control.handle(block(dimmer=200))
    runner.reset_mock()
    source.reset_mock()
    control.handle(block(dimmer=200))
    assert runner.method_calls == [] and source.method_calls == []
    control.handle(block(dimmer=10, macro2=255))
    runner.set_brightness.assert_called_once_with(10)
    runner.set_control.assert_called_once_with("macro2", 1.0)


def test_bump_fires_when_it_rises(rig):
    control, runner, _, _, _ = rig
    control.handle(block())
    control.handle(block(bump=255))
    control.handle(block(bump=255))  # held: no second flash
    control.handle(block(bump=0))
    control.handle(block(bump=128))
    assert [c.args for c in runner.bump.call_args_list] == [(1.0, 0.25), (128 / 255, 0.25)]


def test_hold_follows_the_channel_across_128(rig):
    control, runner, _, _, _ = rig
    control.handle(block(hold=0))  # connecting with it down leaves a web UI hold alone
    control.handle(block(hold=127))
    control.handle(block(hold=128))
    control.handle(block(hold=255))
    control.handle(block(hold=40))
    assert [c.args for c in runner.hold.call_args_list] == [(True,), (False,)]


def test_a_first_packet_with_hold_up_holds(rig):
    control, runner, _, _, _ = rig
    control.handle(block(hold=200))
    runner.hold.assert_called_once_with(True)


def test_silence_releases_a_hold_the_desk_applied_but_not_one_it_did_not(rig):
    control, runner, _, _, clock = rig
    control.handle(block(hold=255))
    clock.t += 3.0
    control.poll()
    assert [c.args for c in runner.hold.call_args_list] == [(True,), (False,)]
    runner.reset_mock()
    control.handle(block(hold=0))
    clock.t += 3.0
    control.poll()
    runner.hold.assert_not_called()


def test_bank_and_program_load_a_playlist_by_name_order_and_go_to_an_entry(rig):
    control, runner, _, store, _ = rig
    playlists = [MagicMock(id=7, entries=[1, 2, 3]), MagicMock(id=3, entries=[1])]
    playlists[0].name, playlists[1].name = "Chill", "Party"
    store.playlists.return_value = playlists
    control.handle(block())
    control.handle(block(bank=0, program=2))  # the program changed: bank 0 is not loaded yet
    store.resolve.assert_called_once_with(7)
    runner.goto.assert_called_once_with(2)
    runner.state.playlist = (7, "Chill")
    runner.reset_mock()
    store.resolve.reset_mock()
    control.handle(block(bank=0, program=1))  # same bank, loaded: just go
    store.resolve.assert_not_called()
    runner.goto.assert_called_once_with(1)
    control.handle(block(bank=5, program=1))  # no such playlist: ignored
    control.handle(block(bank=1, program=4))  # no such entry: ignored
    assert runner.goto.call_count == 1


def test_silence_releases_the_continuous_controls(rig):
    control, runner, source, _, clock = rig
    control.handle(block(dimmer=20, speed=255, strobe=100))
    runner.reset_mock()
    clock.t += 1.9
    control.poll()
    assert runner.method_calls == []
    clock.t += 0.2
    control.poll()
    runner.set_brightness.assert_called_once_with(180)
    runner.set_strobe.assert_called_once_with(0.0)
    runner.set_speed.assert_called_once_with(1.0)
    runner.set_tint.assert_called_once_with(255, 255, 255, 0.0)
    source.set_source.assert_called_with("external")
    assert not control.live()
    runner.reset_mock()
    control.handle(block(dimmer=20))  # back again: a first packet once more
    runner.set_brightness.assert_called_once_with(20)


def test_the_receiver_hands_over_the_block_and_ignores_short_packets():
    received = []
    receiver = Receiver(Layout("tile"), artnet_port=0, sacn_port=0, bind="127.0.0.1")
    receiver.configure_control(artnet_universe=0, sacn_universe=1, address=201, width=WIDTH)
    receiver.on_control = received.append
    receiver.start()
    try:
        port = receiver.ports()["artnet"]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(artdmx(0, bytes(192)), ("127.0.0.1", port))  # stops short of channel 201
            data = bytearray(512)
            data[200 : 200 + WIDTH] = bytes(range(1, WIDTH + 1))
            s.sendto(artdmx(0, bytes(data)), ("127.0.0.1", port))
        end = time.monotonic() + 2.0
        while not received and time.monotonic() < end:
            time.sleep(0.01)
    finally:
        receiver.stop()
    assert received == [bytes(range(1, WIDTH + 1))]
    assert receiver.latest() is not None  # the same universe still carried the tile pixels


def test_the_qlc_fixture_matches_the_table():
    xml = qlc_fixture()
    ns = {"q": "http://www.qlcplus.org/FixtureDefinition"}
    root = ElementTree.fromstring(xml)
    mode = root.find("q:Mode", ns)
    assert [c.text for c in mode.findall("q:Channel", ns)] == [c.name for c in CHANNELS]
    assert len(root.findall("q:Channel", ns)) == WIDTH


@pytest.mark.skipif(not FIXTURE.exists(), reason="no repo docs/ beside the package (e.g. the Pi's synced tree)")
def test_the_qlc_fixture_matches_the_shipped_file():
    assert FIXTURE.read_text() == qlc_fixture(), "regenerate docs/fixtures/dance-floor-v2.qxf with `df2-pi fixture`"
