import io
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from df2_pi import cli
from df2_pi.animation import AnimationRegistry
from df2_pi.engine.clock import FrameInfo
from df2_pi.output.dev import (
    FrameCollector,
    TerminalSink,
    display_grid,
    rasterize,
    render_bloom,
    render_terminal,
    truecolor_supported,
)
from df2_pi.output.preview import read_recording, wait_for
from df2_pi.pixels import PixelFrame, TileFrame, default_geometry

SOLID = '''
from df2_pi.animation import animation, Param
from df2_pi.pixels import TileFrame

@animation(name="Solid", params={"level": Param(int, default=10, min=0, max=255), "mode": Param(str, default="up", choices=["up", "down"])}, period=2.0)
def render(previous, ctx):
    frame = TileFrame.black(ctx.geometry)
    frame.data[:] = (ctx.params["level"], min(255, ctx.frame), 0)  # changes every frame: a GIF writer merges identical ones
    return frame
'''


@pytest.fixture
def anim_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animations"
    d.mkdir()
    (d / "solid.py").write_text(textwrap.dedent(SOLID))
    (d / "broken.py").write_text("def render(:\n")
    return d


@pytest.fixture
def env(tmp_path: Path, anim_dir: Path) -> list[str]:
    """Global flags pointing every command at a scratch db and registry."""
    return ["--db", str(tmp_path / "df2.sqlite3"), "--animations", str(anim_dir)]


@pytest.fixture(autouse=True)
def no_serial(monkeypatch):
    """Any attempt to open a serial port or a GPIO line is a test failure."""
    import serial

    def boom(*a, **k):
        raise AssertionError("serial.Serial was touched in a headless run")

    monkeypatch.setattr(serial, "Serial", boom)
    try:
        import gpiozero

        monkeypatch.setattr(gpiozero, "OutputDevice", boom)
    except ImportError:
        pass


# ---- parsing ------------------------------------------------------------------------------------


def test_every_command_parses():
    p = cli.build_parser()
    for argv in (
        ["play"],
        ["play", "--playlist", "Party", "--fps", "25", "--brightness", "128"],
        ["play", "--animation", "ripple", "--param", "speed=2.0", "--param", "hue=0.6"],
        ["play", "--no-hardware", "--terminal"],
        ["play", "--no-hardware", "--terminal=tiles", "--frames", "10"],
        ["play", "--animation", "lightning", "--window", "--fps", "30", "--realtime"],
        ["play", "--animation", "chase", "--record", "out.gif", "--frames", "90"],
        ["animations"],
        ["animations", "-v"],
        ["playlists"],
        ["playlists", "list"],
        ["playlists", "show", "Party"],
        ["playlists", "create", "Party", "--shuffle", "--crossfade", "2"],
        ["playlists", "add", "Party", "ripple", "--duration", "30", "--param", "speed=2", "--position", "0"],
        ["playlists", "move", "Party", "2", "0"],
        ["playlists", "remove", "Party", "1"],
        ["playlists", "set-startup", "1"],
        ["playlists", "delete", "Party"],
        ["ledwalk", "--row", "0", "--slot", "0"],
        ["ledwalk", "--tile", "9", "--delay", "0.1", "--color", "255,0,0", "--no-hardware", "--terminal"],
        ["tilewalk", "--no-hardware", "--frames", "3"],
        ["scan"],
        ["status", "3"],
        ["version"],
        ["blackout"],
        ["--chain", "/dev/ttyAMA0:23", "--baudrate", "1000000", "scan"],
    ):
        args = p.parse_args(argv)
        assert callable(args.func), argv
    args = p.parse_args(["play", "--animation", "x", "--param", "speed=2.0"])
    assert args.param == [("speed", "2.0")]
    assert p.parse_args(["play", "--no-hardware", "--terminal"]).terminal == "grid"
    assert p.parse_args(["play"]).terminal is None
    assert p.parse_args(["playlists"]).action == "list"
    assert p.parse_args(["ledwalk", "--color", "1,2,3"]).color == (1, 2, 3)


def test_bad_arguments_are_rejected(capsys):
    p = cli.build_parser()
    for argv in (
        ["play", "--param", "nokey"],
        ["play", "--terminal=huge"],
        ["ledwalk", "--color", "1,2"],
        ["ledwalk", "--color", "1,2,300"],
        ["nonsense"],
    ):
        with pytest.raises(SystemExit):
            p.parse_args(argv)
    with pytest.raises(SystemExit):
        cli.main(["play", "--no-hardware", "--brightness", "300"])


def test_param_coercion_against_the_specs(anim_dir):
    registry = AnimationRegistry.discover(anim_dir)
    solid = registry["solid"]
    assert cli.coerce_params(solid, [("level", "42"), ("mode", "down")]) == {"level": 42, "mode": "down"}
    with pytest.raises(SystemExit, match="above the maximum"):
        cli.coerce_params(solid, [("level", "999")])
    with pytest.raises(SystemExit, match="not one of"):
        cli.coerce_params(solid, [("mode", "sideways")])
    with pytest.raises(SystemExit, match="no parameter 'speed'"):
        cli.coerce_params(solid, [("speed", "1")])


# ---- headless runs ------------------------------------------------------------------------------


def test_no_hardware_frames_10_runs_end_to_end(env, capsys):
    assert cli.main([*env, "play", "--no-hardware", "--frames", "10", "--fps", "120"]) == 0
    err = capsys.readouterr().err
    assert "seeded a Default playlist" in err
    assert "warning: broken:" in err
    assert "10 frames, 0 dropped" in err


def test_one_animation_with_params_in_the_terminal(env, capsys):
    rc = cli.main([*env, "play", "--no-hardware", "--terminal=tiles", "--animation", "solid", "--param", "level=200", "--frames", "4", "--fps", "120"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "\x1b[38;" in out and "██" in out  # ANSI colour, tile blocks
    assert "frame 0" in out


def test_unknown_animation_and_bad_param_exit_nonzero(env, capsys):
    assert cli.main([*env, "play", "--no-hardware", "--animation", "nope", "--frames", "1"]) == 1
    assert "no animation 'nope'" in capsys.readouterr().err
    assert cli.main([*env, "play", "--no-hardware", "--animation", "broken", "--frames", "1"]) == 1
    assert "SyntaxError" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="above the maximum"):
        cli.main([*env, "play", "--no-hardware", "--animation", "solid", "--param", "level=999", "--frames", "1"])
    assert cli.main([*env, "play", "--no-hardware", "--param", "x=1", "--frames", "1"]) == 2


def test_record_writes_a_readable_gif_with_the_expected_frame_count(env, tmp_path, capsys):
    pytest.importorskip("imageio")
    from df2_pi.output.dev import frame_count_of

    out = tmp_path / "out.gif"
    rc = cli.main([*env, "play", "--no-hardware", "--animation", "solid", "--record", str(out), "--frames", "9", "--fps", "120"])
    assert rc == 0
    assert out.exists() and frame_count_of(out) == 9
    assert "wrote 9 frames" in capsys.readouterr().err


def test_record_to_df2rec_and_a_bad_extension(env, tmp_path):
    out = tmp_path / "take.df2rec"
    assert cli.main([*env, "play", "--no-hardware", "--animation", "solid", "--record", str(out), "--frames", "5", "--fps", "120"]) == 0
    played = list(read_recording(out))
    assert [r.frame_no for r in played] == [0, 1, 2, 3, 4]
    assert all(r.frame.data[0, 0, 0] == 10 for r in played)
    with pytest.raises(ValueError, match="use .gif"):
        FrameCollector().export(tmp_path / "x.png")


def test_headless_import_path_with_gpiozero_lgpio_and_serial_unimportable(env):
    script = textwrap.dedent(
        """
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("gpiozero", "lgpio", "serial"):
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        from df2_pi import cli
        rc = cli.main(sys.argv[1:] + ["play", "--no-hardware", "--frames", "3", "--fps", "120"])
        assert rc == 0, rc
        assert not any(m.split(".")[0] in ("gpiozero", "lgpio", "serial") for m in sys.modules), "hardware module imported"
        assert "df2_pi.transport.row_bus" not in sys.modules, "transport imported on a headless run"
        print("clean")
        """
    )
    result = subprocess.run([sys.executable, "-c", script, *env], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# ---- animations and playlists ---------------------------------------------------------------------


def test_animations_lists_and_reports_errors(env, capsys):
    assert cli.main([*env, "animations", "-v"]) == 0
    captured = capsys.readouterr()
    assert "solid" in captured.out and "Solid" in captured.out and "--param level=10 0..255" in captured.out
    assert "broken" in captured.err and "SyntaxError" in captured.err


def test_playlists_round_trip(env, capsys):
    run = lambda *a: (cli.main([*env, "playlists", *a]), capsys.readouterr())  # noqa: E731
    rc, out = run("create", "Party", "--crossfade", "2")
    assert rc == 0 and "created playlist" in out.out
    rc, out = run("add", "Party", "solid", "--duration", "20", "--param", "level=50")
    assert rc == 0 and "position 0 for 20s" in out.out
    rc, out = run("add", "Party", "ghost")
    assert rc == 0 and "not a loaded animation" in out.err
    rc, out = run("show", "Party")
    assert "0  solid" in out.out and "{'level': 50}" in out.out and "UNRESOLVED" in out.out
    rc, out = run("move", "Party", "1", "0")
    assert "moved ghost to position 0" in out.out
    rc, out = run("set-startup", "Party")
    rc, out = run("list")
    assert "* " in out.out and "Party" in out.out
    rc, out = run("remove", "Party", "0")
    rc, out = run("show", "Party")
    assert "ghost" not in out.out
    rc, out = run("delete", "Party")
    rc, out = run("show", "Party")
    assert rc == 1 and "no playlist" in out.err
    with pytest.raises(SystemExit, match="above the maximum"):
        run("create", "P2")
        run("add", "P2", "solid", "--param", "level=999")


def test_play_a_named_playlist(env, capsys):
    cli.main([*env, "playlists", "create", "Party"])
    cli.main([*env, "playlists", "add", "Party", "solid", "--duration", "1"])
    capsys.readouterr()
    assert cli.main([*env, "play", "--no-hardware", "--playlist", "Party", "--frames", "3", "--fps", "120"]) == 0
    assert cli.main([*env, "play", "--no-hardware", "--playlist", "Nope", "--frames", "1"]) == 1


# ---- ledwalk and tilewalk ---------------------------------------------------------------------------


def test_tilewalk_emits_tiles_in_reading_order_and_targets_divmod_on_the_bus():
    geo = default_geometry()
    steps = list(cli.tilewalk_steps(geo))
    assert len(steps) == 64
    for n, (label, frame, (row, col)) in enumerate(steps):
        assert (row, col) == divmod(n, 8)
        assert f"tile {n:2d}" in label and f"bus (row_addr {row}, slot {col})" in label
        frame[row, col] = (255, 255, 255)
        lit = np.argwhere(frame.data.any(axis=-1))
        assert lit.tolist() == [[row, col]]
        assert geo.address_of(n) == (row, col)


def test_tilewalk_and_ledwalk_run_headless(env, capsys):
    assert cli.main([*env, "tilewalk", "--no-hardware", "--delay", "0", "--frames", "3"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "tile  0  (row 0, col 0)  bus (row_addr 0, slot 0)",
        "tile  1  (row 0, col 1)  bus (row_addr 0, slot 1)",
        "tile  2  (row 0, col 2)  bus (row_addr 0, slot 2)",
    ]
    assert cli.main([*env, "ledwalk", "--no-hardware", "--row", "1", "--slot", "1", "--delay", "0", "--frames", "16"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("LED  0  side: left, climbing") and "(ly= 1, lx= 0)" in lines[0]
    assert lines[14].startswith("LED 14  side: left, climbing") and "(ly=15, lx= 0)" in lines[14]
    assert lines[15].startswith("LED 15  side: top, left to right") and "(ly=16, lx= 1)" in lines[15]
    assert cli.main([*env, "ledwalk", "--no-hardware", "--tile", "64"]) == 2


def test_ledwalk_steps_light_exactly_one_led_along_the_chain():
    geo = default_geometry()
    steps = list(cli.ledwalk_steps(geo, 9))
    assert len(steps) == 60
    for led, (label, frame, (tile, index)) in enumerate(steps):
        assert (tile, index) == (9, led)
        frame.tile(tile)[index] = (255, 255, 255)
        assert frame.data.any(axis=-1).sum() == 1
        y, x = geo.led_to_cell[9, led]
        assert frame.grid[y, x].tolist() == [255, 255, 255]


# ---- the dev sinks ----------------------------------------------------------------------------------


def test_terminal_render_is_byte_stable_for_a_known_frame():
    frame = TileFrame.black()
    frame[0, 0] = (255, 0, 0)  # row 0, col 0: nearest the Pi, bottom-left as displayed
    frame[7, 7] = (0, 0, 255)
    text = render_terminal(frame, "tiles", truecolor=True)
    lines = text.split("\n")
    assert len(lines) == 9 and lines[-1] == ""
    # display orientation: the blue tile (row 7) is on the FIRST line, red (row 0) on the last
    assert lines[0].startswith("\x1b[38;2;0;0;0m██" * 7 + "\x1b[38;2;0;0;255m██")
    assert lines[7].startswith("\x1b[38;2;255;0;0m██" + "\x1b[38;2;0;0;0m██" * 7)
    assert all(line.endswith("\x1b[0m") for line in lines[:-1])
    assert text.encode() == render_terminal(frame, "tiles", truecolor=True).encode()

    grid = render_terminal(frame, "grid", truecolor=True)
    grid_lines = grid.split("\n")
    # 17-cell tiles are padded to an even 18 plus one blank row on top, so
    # every horizontal seam lands on ONE half-block line: (1 + 8*18) -> 73
    assert len(grid_lines) == 73 + 1
    assert grid_lines[0].count("▀") == 136
    cells = grid_lines[0].split("▀")
    assert cells[7 * 17 + 1] == "\x1b[38;2;0;0;0m\x1b[48;2;0;0;255m"  # blank over the blue tile's north edge
    assert cells[7 * 17] == "\x1b[38;2;0;0;0m\x1b[48;2;0;0;0m"  # its dark corner
    # the last line: the south edge of canonical row 0 (red tile at col 0), repeated
    # into the background so the bottom of the floor is colour over colour
    assert grid_lines[72].startswith("\x1b[38;2;0;0;0m\x1b[48;2;0;0;0m▀\x1b[38;2;255;0;0m\x1b[48;2;255;0;0m▀")


def test_terminal_grid_puts_every_horizontal_seam_on_one_line():
    frame = TileFrame.black()
    for r in range(8):
        frame.data[r, :] = (r * 30 + 10, 0, 0)  # each tile row its own red
    lines = render_terminal(frame, "grid", truecolor=True).split("\n")[:-1]
    reds = [r * 30 + 10 for r in range(8)]
    for k in range(7):  # seam k: display block k (canonical row 7-k) over block k+1
        line = lines[9 * (k + 1)]
        upper, lower = reds[7 - k], reds[6 - k]
        assert f"\x1b[38;2;{upper};0;0m\x1b[48;2;{lower};0;0m▀" in line
    # and the vertical edge lines are unbroken: column 0 is lit on every line of every tile
    for k in range(8):
        for line in lines[9 * k + 1 : 9 * k + 9]:
            assert line.startswith(f"\x1b[38;2;{reds[7 - k]};0;0m\x1b[48;2;{reds[7 - k]};0;0m▀")

    fallback = render_terminal(frame, "tiles", truecolor=False)
    assert "\x1b[38;5;196m██" in fallback and "38;2;" not in fallback  # 256-colour cube red
    with pytest.raises(ValueError):
        render_terminal(frame, "huge")


def test_terminal_sink_self_limits_and_restores_the_cursor():
    out = io.StringIO()
    sink = TerminalSink("tiles", max_fps=10, truecolor=True, out=out)
    frame = TileFrame.black()
    for n in range(30):
        sink.submit(frame, FrameInfo(n, n / 30, 0.0))
        assert wait_for(lambda: sink.frames_handled == n + 1)
    sink.close()
    assert sink.drawn == 10  # 30 frames at 30 FPS, drawn at 10
    text = out.getvalue()
    assert text.startswith("\x1b[2J\x1b[?25l") and text.endswith("\x1b[?25h\n")
    assert text.count("\x1b[H") == 10


def test_truecolor_detection():
    assert truecolor_supported({"COLORTERM": "truecolor"})
    assert truecolor_supported({"COLORTERM": "24bit"})
    assert not truecolor_supported({"COLORTERM": ""})
    assert not truecolor_supported({})


def test_rasterisers_use_display_orientation():
    p = PixelFrame.black()
    p.tile(0)[0] = (255, 255, 255)  # LED 0 of tile 0: canonical cell (1, 0)
    img = display_grid(p)
    assert img.shape == (136, 136, 3)
    assert img[136 - 2, 0].tolist() == [255, 255, 255]  # flipped: near the bottom
    assert not img[1, 0].any()
    big = rasterize(p, scale=3)
    assert big.shape == (408, 408, 3) and big[3 * 134, 0].tolist() == [255, 255, 255]
    bloom = render_bloom(p, scale=4)
    assert bloom.shape == (544, 544, 3) and bloom.dtype == np.uint8
    y, x = (136 - 2) * 4 + 2, 2
    assert bloom[y, x].max() > bloom[y, x + 6].max() > bloom[y, x + 12].max() > 0  # bright centre, bloom falling off
    assert not bloom[50, 300].any()  # dark far away
