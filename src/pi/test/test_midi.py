import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

mido = pytest.importorskip("mido", reason="MIDI needs the [midi] extra")
pytest.importorskip("yaml", reason="MIDI needs the [midi] extra")

from df2_pi.animation import AnimationRegistry  # noqa: E402
from df2_pi.engine import Runner  # noqa: E402
from df2_pi.interfacing.controls import FloorControls  # noqa: E402
from df2_pi.interfacing.midi import DEFAULT_MAP, Binding, MidiControl, MidiMap  # noqa: E402
from df2_pi.output import FanOut, NullSink, PreviewSink  # noqa: E402
from df2_pi.playlists import PlaylistStore  # noqa: E402
from df2_pi.web import AppContext, create_app  # noqa: E402

SOLID = """
from df2_pi.animation import animation
from df2_pi.pixels import TileFrame

@animation(name="{name}")
def render(previous, ctx):
    return TileFrame.black(ctx.geometry)
"""


def note(n, velocity=127, channel=0):
    return mido.Message("note_on", note=n, velocity=velocity, channel=channel)


def off(n, channel=0):
    return mido.Message("note_off", note=n, velocity=0, channel=channel)


def cc(control, value, channel=0):
    return mido.Message("control_change", control=control, value=value, channel=channel)


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    d = tmp_path / "animations"
    d.mkdir()
    for name in ("waves", "lightning", "ripple"):
        (d / f"{name}.py").write_text(SOLID.format(name=name.title()))
    return AnimationRegistry.discover(d)


@pytest.fixture
def store(registry):
    s = PlaylistStore(":memory:", registry=registry)
    yield s
    s.close()


@pytest.fixture
def rig(registry, store, tmp_path):
    runner = MagicMock()
    runner.state.show.strobe_max_hz = 10.0
    runner.state.playlist = None
    beat, palettes = MagicMock(), MagicMock()
    controls = FloorControls(runner, registry, store=store, beat=beat, palettes=palettes)
    path = tmp_path / "midi.yaml"
    midi = MidiControl(controls, store=store, beat=beat, map_path=path, lister=lambda: [], opener=None)
    return midi, runner, beat, palettes, path


def use(midi, path, text):
    path.write_text(text)
    midi.load_map()


# ---- the mapping file ------------------------------------------------------------------------------


def test_with_no_file_the_apc_mini_default_is_copied_and_loaded(rig):
    midi, _, _, _, path = rig
    assert path.read_text() == DEFAULT_MAP.read_text()
    notes = {b.number for b in midi.map.bindings if b.kind == "note"}
    assert set(range(40)) <= notes  # pad rows 1-5
    assert {64, 71, 82, 89, 100, 107, 112, 119} <= notes  # mk1 and mk2 buttons both
    assert {b.number for b in midi.map.bindings if b.kind == "cc"} == set(range(48, 57))  # 8 faders and the master
    assert midi.map.program_change and midi.map.clock


def test_bindings_parse_validate_and_round_trip():
    text = """
program_change: false
clock: true
bindings:
  - {note: 0, to: /floor/goto/0}
  - {cc: [7, 39], to: /floor/speed, channel: 2, port: apc}
  - {note: 64, to: /floor/blackout, toggle: true}
  - {note: 86, to: /floor/speed, value: 0.25}
"""
    m = MidiMap.parse(text)
    assert m.bindings[1] == Binding("cc", 7, "/floor/speed", lsb=39, channel=2, port="apc")
    assert MidiMap.parse(m.dump()) == m
    for bad in ("bindings: [{note: 200, to: /floor/next}]", "bindings: [{note: 1, to: /elsewhere}]",
                "bindings: [{note: 1, cc: 2, to: /floor/next}]", "bindings: [{note: [1, 2], to: /floor/next}]",
                "bindings: [{note: 1, to: /floor/next, channel: 17}]", "bindings: [{note: 1, to: /floor/next, colour: red}]"):
        with pytest.raises(ValueError):
            MidiMap.parse(bad)


def test_a_bad_edit_keeps_the_last_good_map(rig):
    midi, runner, *_, path = rig
    use(midi, path, "bindings:\n  - {note: 1, to: /floor/next}\n")
    use(midi, path, "bindings: [this is not right")
    assert midi.map_error and len(midi.map.bindings) == 1
    midi.handle("any", note(1))
    runner.next.assert_called_once_with()


def test_the_file_is_reread_when_it_changes(rig):
    midi, runner, *_, path = rig
    use(midi, path, "bindings:\n  - {note: 1, to: /floor/next}\n")
    time.sleep(0.01)
    path.write_text("bindings:\n  - {note: 1, to: /floor/previous}\n")
    midi.poll()
    midi.handle("any", note(1))
    runner.previous.assert_called_once_with() and runner.next.assert_not_called()


# ---- from MIDI to the runner -----------------------------------------------------------------------


def test_the_default_map_drives_the_floor(rig):
    midi, runner, beat, palettes, _ = rig
    midi.handle("APC MINI", note(3))  # pad row 1: entry 3
    midi.handle("APC MINI", note(17))  # pad row 3: fire
    midi.handle("APC MINI", note(25, velocity=64))  # pad row 4: trigger 1, half as hard
    midi.handle("APC MINI", note(32))  # pad row 5: lightning
    midi.handle("APC MINI", cc(56, 127))  # master fader
    midi.handle("APC MINI", cc(48, 0))  # fader 1: macro 1
    runner.goto.assert_called_once_with(3)
    palettes.activate.assert_called_once_with("fire")
    runner.trigger.assert_called_once_with(1, pytest.approx(64 / 127))
    runner.play_animation.assert_called_once_with("lightning")
    runner.set_brightness.assert_called_once_with(255)
    runner.set_control.assert_called_once_with("macro1", 0.0)
    assert midi.errors == 0


def test_toggles_flip_on_each_press_and_ignore_releases(rig):
    midi, runner, *_ = rig
    for message in (note(68), off(68), note(68), off(68)):  # mk1 track button 5: blackout
        midi.handle("APC MINI", message)
    runner.blackout.assert_called_once_with()
    runner.unblackout.assert_called_once_with()


def test_a_fixed_value_is_sent_on_press_only(rig):
    midi, runner, *_ = rig
    midi.handle("APC MINI", note(116))  # mk2 scene button 5: half speed
    midi.handle("APC MINI", off(116))
    runner.set_speed.assert_called_once_with(0.5)


def test_a_pad_press_fires_once_and_its_release_does_nothing(rig):
    midi, runner, *_, path = rig
    use(midi, path, "bindings:\n  - {note: 5, to: /floor/next}\n")
    midi.handle("any", note(5))
    midi.handle("any", note(5, velocity=0))  # note-on at velocity 0 is a release
    midi.handle("any", note(5))
    assert runner.next.call_count == 2


def test_14_bit_pairs_combine_msb_and_lsb(rig):
    midi, runner, *_, path = rig
    use(midi, path, "bindings:\n  - {cc: [1, 33], to: /floor/brightness}\n")
    midi.handle("any", cc(1, 64))  # coarse half: 64*128 / 16383
    midi.handle("any", cc(33, 127))  # fine: (64*128 + 127) / 16383
    midi.handle("any", cc(1, 127))
    midi.handle("any", cc(33, 127))  # full: 16383 / 16383
    values = [c.args[0] for c in runner.set_brightness.call_args_list]
    assert values == [round(8192 / 16383 * 255), round(8319 / 16383 * 255), round(16256 / 16383 * 255), 255]


def test_bank_select_and_program_change_pick_a_playlist_and_entry(rig, store):
    midi, runner, *_ = rig
    for name, count in (("Chill", 2), ("Party", 4)):  # bank 0 and 1, in name order
        pl = store.create_playlist(name)
        for _ in range(count):
            store.add_entry(pl, "waves")
    midi.handle("desk", cc(0, 0))
    midi.handle("desk", cc(32, 1))
    midi.handle("desk", mido.Message("program_change", program=3))
    loaded = runner.load_playlist.call_args.args[0]
    assert loaded.playlist.name == "Party"
    runner.goto.assert_called_once_with(3)
    midi.handle("desk", mido.Message("program_change", program=9))  # Party has 4 entries
    assert midi.errors == 1 and runner.goto.call_count == 1


def test_channel_and_port_scoping(rig):
    midi, runner, *_, path = rig
    use(midi, path, "bindings:\n  - {note: 1, to: /floor/next, channel: 10, port: launchpad}\n")
    midi.handle("Launchpad X MIDI 1", note(1, channel=0))  # channel 1: no
    midi.handle("APC MINI", note(1, channel=9))  # wrong port: no
    midi.handle("LAUNCHPAD X MIDI 1", note(1, channel=9))  # yes, case-insensitively
    runner.next.assert_called_once_with()


def test_clock_feeds_beat_sync_unless_turned_off(rig):
    midi, runner, beat, _, path = rig
    midi.handle("DAW", mido.Message("clock"))
    midi.handle("DAW", mido.Message("start"))
    assert [c.args[0] for c in beat.midi_message.call_args_list] == [[0xF8], [0xFA]]
    midi.set_options(clock=False)
    midi.handle("DAW", mido.Message("clock"))
    assert beat.midi_message.call_count == 2 and midi.clock_received == 3
    assert midi.status()["recent"] == []  # clock isn't shown: there'd be nothing else


def test_raw_bytes_and_errors_are_recorded_not_raised(rig):
    midi, runner, *_, path = rig
    use(midi, path, "bindings:\n  - {note: 1, to: /floor/play/missing}\n")
    midi.handle("any", [0x90, 1, 100])
    entry = midi.status()["recent"][0]
    assert entry["message"] == "note on 1 vel 100 ch 1" and "missing" in entry["error"]
    midi.handle("any", [0xFF, 0xFF])  # not even MIDI
    assert midi.errors == 2


# ---- learn ----------------------------------------------------------------------------------------


def test_learn_binds_the_next_control_moved_and_survives_a_restart(rig, store, registry):
    midi, runner, beat, palettes, path = rig
    midi.learn("/floor/palette/ice")
    midi.handle("nanoKONTROL2:nanoKONTROL2 MIDI 1 20:0", cc(20, 90))
    assert midi.learning is None
    midi.handle("nanoKONTROL2:nanoKONTROL2 MIDI 1 20:0", cc(20, 127))
    palettes.activate.assert_called_once_with("ice")
    again = MidiControl(FloorControls(runner, registry, store=store, palettes=palettes), store=store, map_path=path, lister=lambda: [])
    assert again.map.bindings[-1] == Binding("cc", 20, "/floor/palette/ice")


def test_learn_replaces_what_the_control_did_and_keeps_its_options(rig):
    midi, runner, *_, path = rig
    midi.learn("/floor/freeze", toggle=True, this_port_only=True)
    midi.handle("APC MINI:APC MINI MIDI 1 24:0", note(0))  # pad 1 was entry 0
    bound = [b for b in midi.map.bindings if b.kind == "note" and b.number == 0]
    assert bound == [Binding("note", 0, "/floor/freeze", port="APC MINI", toggle=True)]
    runner.goto.assert_not_called()  # the learning press itself does nothing else


def test_learn_refuses_a_non_floor_address(rig):
    midi, *_ = rig
    with pytest.raises(ValueError):
        midi.learn("/composition/layers/1")


# ---- ports ----------------------------------------------------------------------------------------


def test_ports_are_opened_skipping_midi_through_and_let_go_when_they_vanish(registry, store, tmp_path):
    names = ["Midi Through:Midi Through Port-0 14:0", "APC MINI:APC MINI MIDI 1 24:0"]
    opened, closed = [], []

    class Port:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    def opener(name, callback):
        opened.append(name)
        return Port(name)

    midi = MidiControl(FloorControls(MagicMock(), registry), map_path=tmp_path / "m.yaml", lister=lambda: list(names), opener=opener)
    midi.poll()
    assert opened == ["APC MINI:APC MINI MIDI 1 24:0"] and midi.status()["ports"] == opened
    names.pop()
    midi.poll()
    assert closed == ["APC MINI:APC MINI MIDI 1 24:0"] and midi.status()["ports"] == []


def test_a_real_virtual_port(registry, store, tmp_path):
    try:
        out = mido.open_output("df2 test controller", virtual=True)
    except Exception as exc:  # no rtmidi backend, or no virtual ports on this system
        pytest.skip(f"no virtual MIDI ports: {exc}")
    runner = MagicMock()
    try:
        midi = MidiControl(FloorControls(runner, registry), map_path=tmp_path / "m.yaml")
        midi.poll()
        if not any("df2 test controller" in p for p in midi.status()["ports"]):
            pytest.skip("the virtual port isn't listed as an input here")
        out.send(note(65))  # mk1 track button 2: next
        end = time.monotonic() + 2.0
        while runner.next.call_count == 0 and time.monotonic() < end:
            time.sleep(0.01)
        runner.next.assert_called_once_with()
    finally:
        midi.stop()
        out.close()


# ---- the API --------------------------------------------------------------------------------------


def test_the_api_learns_removes_and_resets(registry, store, tmp_path):
    runner = MagicMock(spec=Runner)
    runner.state = Runner(registry, FanOut([NullSink()])).state
    midi = MidiControl(FloorControls(runner, registry, store=store), store=store, map_path=tmp_path / "midi.yaml", lister=lambda: [])
    preview = PreviewSink()
    client = TestClient(create_app(AppContext(registry, store, FanOut([NullSink(), preview]), runner, preview, None, None, None, None, midi)))
    body = client.get("/api/midi").json()
    count = len(body["map"]["bindings"])
    assert body["map"]["bindings"][0] == {"note": 0, "to": "/floor/goto/0", "control": "note 0"}
    assert client.post("/api/midi/learn", json={"to": "/floor/bump"}).json()["status"]["learning"]["to"] == "/floor/bump"
    assert client.delete("/api/midi/learn").json()["status"]["learning"] is None
    assert client.post("/api/midi/learn", json={"to": "nope"}).status_code == 422
    assert len(client.delete("/api/midi/bindings/0").json()["map"]["bindings"]) == count - 1
    assert client.delete("/api/midi/bindings/999").status_code == 404
    assert client.patch("/api/midi/map", json={"clock": False}).json()["map"]["clock"] is False
    assert len(client.post("/api/midi/default").json()["map"]["bindings"]) == count
    assert client.patch("/api/midi", json={"enabled": False}).json()["status"]["enabled"] is False
    assert store.get_bool("midi_enabled") is False
