"""Command-line entry point: `df2-pi`.

    df2-pi play                               # startup playlist, on the real floor
    df2-pi play --playlist "Party"
    df2-pi play --animation ripple            # one animation, looping: the authoring loop
    df2-pi play --animation ripple --param speed=2.0
    df2-pi play --no-hardware --terminal      # no floor at all; renders in the terminal
    df2-pi play --animation bolt --window     # a real window (pip install -e ".[preview]")
    df2-pi play --animation chase --no-hardware --record out.gif --frames 90

    df2-pi animations                         # what the registry found, and what failed
    df2-pi playlists [list|show|create|add|move|remove|set-startup|delete]
    df2-pi ledwalk --row 0 --slot 0           # one LED at a time: verify LED 0 and the winding
    df2-pi tilewalk                           # one tile at a time: verify the install wiring

    df2-pi scan | status ROW | version | blackout      # Row Bus admin

The transport (serial, GPIO) is imported only by the commands that touch
the floor, so `--no-hardware` runs on a laptop with none of it installed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .protocol.constants import DEFAULT_BAUDRATE, Cmd

STATUS_STATE_NAMES = {0x00: "idle", 0x01: "discovering", 0x02: "running", 0x03: "error"}

log = logging.getLogger("df2_pi")


# ---- hardware (imported lazily) ------------------------------------------------------------


def _open_floor(args: argparse.Namespace):
    from .transport.chain_map import RowChainMap
    from .transport.floor import ChainConfig, Floor, default_chain_configs

    def parse_chain(spec: str) -> ChainConfig:
        """PORT[:XDIR_GPIO] - e.g. /dev/ttyAMA0:17, or /dev/ttyAMA0:none"""
        port, _, xdir = spec.partition(":")
        if not xdir:
            raise argparse.ArgumentTypeError(f"--chain needs PORT:XDIR_GPIO, got {spec!r}")
        pin = None if xdir.lower() in ("none", "off") else int(xdir)
        return ChainConfig(port, pin, args.baudrate)

    if args.chain:
        chains = [parse_chain(spec) for spec in args.chain]
    else:
        chains = default_chain_configs(args.baudrate)
    chain_map = RowChainMap.single_chain() if len(chains) == 1 else RowChainMap.alternating(len(chains))
    return Floor(chains=chains, chain_map=chain_map)


def _cmd_status(args: argparse.Namespace) -> int:
    from .transport.floor import RowNotResponding

    with _open_floor(args) as floor:
        try:
            frame = floor.request(args.row, Cmd.STATUS)
        except RowNotResponding as exc:
            print(exc, file=sys.stderr)
            return 1
        state, tiles = frame.payload[0], frame.payload[1]
        # Uptime is appended after the 8 tile-status bytes; a shorter payload
        # is firmware that predates the field, not a malformed reply.
        up = ""
        if len(frame.payload) >= 14:
            p = frame.payload
            secs = (p[10] << 24) | (p[11] << 16) | (p[12] << 8) | p[13]
            h, rem = divmod(secs, 3600)
            m, sec = divmod(rem, 60)
            up = f" up={h}h{m:02d}m{sec:02d}s"
        print(
            f"row 0x{frame.addr:02X} (chain {floor.chain_map.chain_for(args.row)}): "
            f"state={STATUS_STATE_NAMES.get(state, hex(state))} tiles_found={tiles}{up}"
        )
        return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    with _open_floor(args) as floor:
        found = floor.scan()
        if not found:
            print("no row controllers responded", file=sys.stderr)
            return 1
        for row, chain in sorted(found.items()):
            print(f"row 0x{row:02X}  chain {chain}")
        return 0


def _cmd_version(args: argparse.Namespace) -> int:
    from .transport.floor import RowNotResponding
    from .version_report import RowVersionReport, format_version_report

    with _open_floor(args) as floor:
        found = floor.scan()
        if not found:
            print("no row controllers responded", file=sys.stderr)
            return 1

        row_reports: dict[int, RowVersionReport | None] = {}
        for row in sorted(found):
            try:
                frame = floor.request(row, Cmd.VERSION)
                row_reports[row] = RowVersionReport.decode(frame.payload)
            except (RowNotResponding, ValueError) as exc:
                print(f"row 0x{row:02X}: {exc}", file=sys.stderr)
                row_reports[row] = None

        text, ok = format_version_report(row_reports)
        print(text)
        return 0 if ok else 1


def _cmd_blackout(args: argparse.Namespace) -> int:
    with _open_floor(args) as floor:
        floor.blackout()
        print("blackout sent to all chains")
        return 0


# ---- the driver --------------------------------------------------------------------------------


def _registry(args: argparse.Namespace):
    from .animation import AnimationRegistry, default_animations_dir

    path = Path(args.animations) if args.animations else default_animations_dir()
    return AnimationRegistry.discover(path)


def _store(args: argparse.Namespace, registry):
    from .playlists import PlaylistStore

    return PlaylistStore(args.db, registry=registry)


def parse_param(text: str) -> tuple[str, str]:
    """`--param key=value` -> (key, value); coercion happens against the
    animation's Param spec once it is known."""
    key, sep, value = text.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(f"--param needs KEY=VALUE, got {text!r}")
    return key.strip(), value


def coerce_params(definition, pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Coerce `--param` strings through the animation's specs; unknown
    keys and out-of-range values are errors at the command line."""
    params: dict[str, Any] = {}
    for key, value in pairs:
        spec = definition.meta.params.get(key)
        if spec is None:
            raise SystemExit(
                f"{definition.id} has no parameter {key!r}; it declares: {sorted(definition.meta.params) or 'none'}"
            )
        try:
            params[key] = spec.coerce(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"--param {key}: {exc}") from exc
    return params


def build_sinks(args: argparse.Namespace, clock=None):
    """The fan-out for a `play`/`ledwalk`/`tilewalk` run: hardware unless
    `--no-hardware`, plus whatever development sinks were asked for.
    Returns (fanout, window, collector); the caller pumps the window and
    exports the collector."""
    from .encode import FrameEncoder
    from .output import FanOut

    sinks: list = []
    window = None
    collector = None
    if not args.no_hardware:
        from .output.hardware import HardwareSink

        encoder = FrameEncoder(brightness=args.brightness)
        sinks.append(HardwareSink(_open_floor(args), encoder, clock=clock))
    if args.terminal:
        from .output.dev import TerminalSink

        sinks.append(TerminalSink(args.terminal))
    if args.window:
        from .output.dev import WindowSink

        window = WindowSink()
        sinks.append(window)
    if args.record:
        from .output.dev import FrameCollector

        collector = FrameCollector(limit=args.frames)
        sinks.append(collector)
    if not sinks:
        from .output import NullSink

        sinks.append(NullSink())
    return FanOut(sinks), window, collector


def _run_until_done(runner, window, frames: int | None) -> None:
    """Drive the runner to completion: on the main thread with a window to
    pump, otherwise directly. `frames` stops after that many."""
    if frames is not None:
        from .output import NullSink

        class Counter(NullSink):
            def submit(self, frame, info, effects=None):
                super().submit(frame, info, effects)
                if self.frames >= frames:
                    runner.stop()

        runner.fanout.attach(Counter("frame-counter"))
    if window is None:
        runner.run()
        return
    runner.start()
    try:
        while runner.alive:
            if not window.pump():
                runner.stop()
                break
    finally:
        runner.join(5.0)
        window.shutdown()


def _cmd_play(args: argparse.Namespace) -> int:
    from .engine import FrameClock, Runner

    registry = _registry(args)
    store = _store(args, registry)
    for animation_id, error in registry.errors.items():
        print(f"warning: {animation_id}: {error.message}", file=sys.stderr)

    clock = FrameClock(fps=args.fps, realtime=args.realtime)
    fanout, window, collector = build_sinks(args, clock)
    runner = Runner(registry, fanout, store=store, clock=clock, brightness=args.brightness, seed=args.seed)

    if args.animation:
        definition = registry.get(args.animation)
        if definition is None:
            error = registry.errors.get(args.animation)
            detail = f": {error.message}" if error else ""
            print(f"no animation {args.animation!r}{detail}", file=sys.stderr)
            return 1
        params = coerce_params(definition, args.param)
        runner.play_animation(args.animation, params)  # loops until stopped
    else:
        if args.param:
            print("--param needs --animation", file=sys.stderr)
            return 2
        if args.playlist:
            try:
                playlist = store.resolve(_playlist_key(args.playlist), registry)
            except KeyError as exc:
                print(exc, file=sys.stderr)
                return 1
        else:
            if store.seed_default(registry) is not None:
                print("seeded a Default playlist", file=sys.stderr)
            startup = store.startup_playlist()
            playlist = store.resolve(startup, registry) if startup is not None else None
        if playlist is None:
            print("no playlist; playing the idle animation", file=sys.stderr)
        else:
            for entry in playlist.entries:
                if entry.unresolved:
                    print(f"warning: entry {entry.entry.position} ({entry.entry.animation_id}) unresolved: {entry.error}", file=sys.stderr)
        runner.load_playlist(playlist)

    runner.install_signal_handlers()
    _run_until_done(runner, window, args.frames)
    if collector is not None:
        out = collector.export(args.record, fps=args.fps, bloom=args.bloom)
        print(f"wrote {len(collector.frames)} frames to {out}", file=sys.stderr)
    telemetry = runner.clock.telemetry()
    print(
        f"{telemetry.frames} frames, {telemetry.dropped} dropped, jitter p95 {telemetry.jitter_ms.p95:.2f} ms",
        file=sys.stderr,
    )
    return 0


def _playlist_key(text: str) -> int | str:
    return int(text) if text.isdigit() else text


# ---- animations and playlists -----------------------------------------------------------------


def _cmd_animations(args: argparse.Namespace) -> int:
    registry = _registry(args)
    for definition in sorted(registry, key=lambda d: d.id):
        meta = definition.meta
        line = f"{definition.id:<24} {meta.name:<28} {meta.format:<6}"
        if meta.period:
            line += f" period={meta.period:g}s"
        if meta.tags:
            line += f"  [{', '.join(meta.tags)}]"
        print(line)
        if args.verbose:
            for key, spec in meta.params.items():
                bounds = ""
                if spec.choices is not None:
                    bounds = f" one of {list(spec.choices)}"
                elif spec.min is not None or spec.max is not None:
                    bounds = f" {spec.min}..{spec.max}"
                print(f"    --param {key}={spec.default!r}{bounds}")
            if meta.description:
                print(f"    {meta.description}")
    for animation_id, error in sorted(registry.errors.items()):
        print(f"{animation_id:<24} ERROR ({error.stage}): {error.message}", file=sys.stderr)
    return 1 if registry.errors and not registry.animations else 0


def _cmd_playlists(args: argparse.Namespace) -> int:
    registry = _registry(args)
    store = _store(args, registry)
    action = args.action

    if action == "list":
        startup = store.get_int("startup_playlist")
        for pl in store.playlists():
            mark = "*" if pl.id == startup else " "
            flags = ", ".join(f for f, on in (("loop", pl.loop), ("shuffle", pl.shuffle)) if on)
            total = sum(e.duration_s for e in pl.entries)
            print(f"{mark} {pl.id:>3}  {pl.name:<24} {len(pl.entries):>3} entries  {total:6.0f}s  {flags}")
        return 0

    if action == "create":
        pl = store.create_playlist(args.name, loop=not args.no_loop, shuffle=args.shuffle, crossfade_s=args.crossfade)
        print(f"created playlist {pl.id} {pl.name!r}")
        return 0

    try:
        pl = store.playlist(_playlist_key(args.playlist))
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    if action == "show":
        resolved = store.resolve(pl, registry)
        print(f"{pl.id} {pl.name!r}  loop={pl.loop} shuffle={pl.shuffle} crossfade={pl.crossfade_s:g}s")
        for r in resolved.entries:
            e = r.entry
            status = "" if r.playable else ("  (disabled)" if not e.enabled else f"  UNRESOLVED: {r.error}")
            params = f"  {dict(e.params)}" if e.params else ""
            print(f"  {e.position:>2}  {e.animation_id:<24} {e.duration_s:6.1f}s{params}{status}")
        return 0
    if action == "add":
        definition = registry.get(args.animation)
        params = coerce_params(definition, args.param) if definition is not None else dict(args.param)
        if definition is None:
            print(f"warning: {args.animation!r} is not a loaded animation; adding anyway", file=sys.stderr)
        entry = store.add_entry(pl, args.animation, duration_s=args.duration, params=params, position=args.position)
        print(f"added {entry.animation_id} at position {entry.position} for {entry.duration_s:g}s")
        return 0
    if action == "move":
        entry = pl.entries[args.position]
        moved = store.move_entry(entry, args.new_position)
        print(f"moved {moved.animation_id} to position {moved.position}")
        return 0
    if action == "remove":
        entry = pl.entries[args.position]
        store.remove_entry(entry)
        print(f"removed {entry.animation_id} from position {entry.position}")
        return 0
    if action == "set-startup":
        store.set_setting("startup_playlist", pl.id)
        print(f"startup playlist is now {pl.name!r}")
        return 0
    if action == "delete":
        store.delete_playlist(pl)
        print(f"deleted {pl.name!r}")
        return 0
    raise AssertionError(action)


# ---- ledwalk and tilewalk -----------------------------------------------------------------------


def parse_color(text: str) -> tuple[int, int, int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--color needs R,G,B, got {text!r}")
    rgb = tuple(int(p) for p in parts)
    if any(not 0 <= v <= 255 for v in rgb):
        raise argparse.ArgumentTypeError(f"--color channels must be 0..255, got {text!r}")
    return rgb  # type: ignore[return-value]


def ledwalk_steps(geometry, tile: int):
    """(label, frame) per step: one LED lit at a time along the chain."""
    from .pixels import PixelFrame

    n = geometry.leds_per_side
    sides = ["left, climbing", "top, left to right", "right, descending", "bottom, right to left"]
    for led in range(geometry.leds_per_tile):
        frame = PixelFrame.black(geometry)
        y, x = (int(v) for v in geometry.led_to_cell[tile, led])
        y0, x0 = geometry.tile_origin(tile)
        label = f"LED {led:2d}  side: {sides[led // n]:<22} local cell (ly={y - y0:2d}, lx={x - x0:2d})"
        yield label, frame, (tile, led)


def tilewalk_steps(geometry):
    """(label, frame) per step: one tile lit at a time in floor order."""
    from .pixels import TileFrame

    for tile in range(geometry.tiles):
        frame = TileFrame.black(geometry)
        row, col = divmod(tile, geometry.tile_cols)
        yield f"tile {tile:2d}  (row {row}, col {col})  bus (row_addr {row}, slot {col})", frame, (row, col)


def _walk(args: argparse.Namespace, steps, paint) -> int:
    from .engine.clock import FrameInfo
    from .pixels import default_geometry

    geometry = default_geometry()
    fanout, window, _ = build_sinks(args)
    try:
        for n, (label, frame, where) in enumerate(steps(geometry)):
            paint(frame, where, args.color)
            print(label, flush=True)
            fanout.submit(frame, FrameInfo(n, n * args.delay, 0.0))
            fanout.latch()
            if window is not None:
                deadline = time.monotonic() + args.delay
                while time.monotonic() < deadline:
                    if not window.pump(0.02):
                        return 0
            else:
                time.sleep(args.delay)
            if args.frames is not None and n + 1 >= args.frames:
                break
    finally:
        fanout.blackout()
        fanout.latch()
        fanout.close()
        if window is not None:
            window.shutdown()
    return 0


def _cmd_ledwalk(args: argparse.Namespace) -> int:
    from .pixels import default_geometry

    geometry = default_geometry()
    if args.tile is not None:
        tile = args.tile
    else:
        tile = args.row * geometry.tile_cols + args.slot
    if not 0 <= tile < geometry.tiles:
        print(f"tile must be 0..{geometry.tiles - 1}", file=sys.stderr)
        return 2
    row, col = divmod(tile, geometry.tile_cols)
    print(
        f"walking tile {tile} (row {row}, slot {col}), one LED at a time.\n"
        "LED 0 must light at the LOWER-LEFT of the tile and 0-14 climb the left side;\n"
        "then across the top left-to-right, down the right, back along the bottom.\n"
        "Anything else: the strip is laid in from the wrong end.",
        file=sys.stderr,
    )

    def paint(frame, where, color):
        _tile, led = where
        frame.tile(_tile)[led] = color

    return _walk(args, lambda geo: ledwalk_steps(geo, tile), paint)


def _cmd_tilewalk(args: argparse.Namespace) -> int:
    print(
        "walking tiles 0-63 in floor order, one at a time.\n"
        "Tile 0 must light in the corner nearest the Pi; 0-7 sweep along row 0 away\n"
        "from the row controllers; each row starts at the controller end.\n"
        "A row lit backwards has its Tile Bus cabled from the wrong end; a row out of\n"
        "sequence has the wrong address baked in.",
        file=sys.stderr,
    )

    def paint(frame, where, color):
        row, col = where
        frame[row, col] = color

    return _walk(args, tilewalk_steps, paint)


# ---- the parser ---------------------------------------------------------------------------------


def _add_output_flags(p: argparse.ArgumentParser, *, frames_help: str) -> None:
    p.add_argument("--no-hardware", action="store_true", help="never touch the floor (no serial, no GPIO)")
    p.add_argument(
        "--terminal",
        nargs="?",
        const="grid",
        choices=("grid", "tiles"),
        help="render in the terminal: the full grid (default) or the 8x8 tile view",
    )
    p.add_argument("--window", action="store_true", help="render in a window (needs pygame)")
    p.add_argument("--record", metavar="FILE", help="capture every frame to a .gif, .mp4 or .df2rec (headless)")
    p.add_argument("--bloom", action="store_true", help="render --record frames with the preview bloom")
    p.add_argument("--frames", type=int, metavar="N", help=frames_help)
    p.add_argument("--brightness", type=int, default=255, metavar="0-255", help="global brightness (default 255)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="df2-pi", description="Dance Floor v2 host controller")
    parser.add_argument(
        "--chain",
        action="append",
        metavar="PORT:XDIR_GPIO",
        help="repeatable; defaults to the two-chain hat's ttyAMA0:17 and ttyAMA2:7",
    )
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help=f"baud rate (default: {DEFAULT_BAUDRATE})")
    parser.add_argument("--db", metavar="PATH", help="playlist database (default: $DF2_DB or ~/.local/share/df2/df2.sqlite3)")
    parser.add_argument("--animations", metavar="DIR", help="animation directory (default: the package's animations/)")
    parser.add_argument("-v", "--verbose", action="store_true", help="more output")

    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="run the playlist (or one animation) on the floor or a dev sink")
    play.add_argument("--playlist", metavar="NAME|ID", help="play this playlist instead of the startup one")
    play.add_argument("--animation", metavar="ID", help="play one animation, looping")
    play.add_argument("--param", action="append", default=[], type=parse_param, metavar="KEY=VALUE", help="with --animation; repeatable")
    play.add_argument("--fps", type=float, default=30.0, help="frame rate (default 30)")
    play.add_argument("--realtime", action="store_true", help="raise scheduler priority (needs privileges)")
    play.add_argument("--seed", type=int, help="seed the runner's rng for a reproducible run")
    _add_output_flags(play, frames_help="exit after N frames")
    play.set_defaults(func=_cmd_play)

    animations = sub.add_parser("animations", help="list discovered animations and load errors")
    animations.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="show parameters and descriptions")
    animations.set_defaults(func=_cmd_animations)

    playlists = sub.add_parser("playlists", help="list and edit playlists")
    actions = playlists.add_subparsers(dest="action")
    actions.required = False
    playlists.set_defaults(func=_cmd_playlists, action="list")
    actions.add_parser("list", help="list playlists (* = startup)")
    show = actions.add_parser("show", help="show a playlist's entries")
    show.add_argument("playlist", metavar="NAME|ID")
    create = actions.add_parser("create", help="create an empty playlist")
    create.add_argument("name")
    create.add_argument("--no-loop", action="store_true")
    create.add_argument("--shuffle", action="store_true")
    create.add_argument("--crossfade", type=float, default=0.0, metavar="SECONDS")
    add = actions.add_parser("add", help="append an animation to a playlist")
    add.add_argument("playlist", metavar="NAME|ID")
    add.add_argument("animation", metavar="ID")
    add.add_argument("--duration", type=float, metavar="SECONDS", help="default: the default_entry_duration setting")
    add.add_argument("--param", action="append", default=[], type=parse_param, metavar="KEY=VALUE")
    add.add_argument("--position", type=int, help="insert here instead of appending")
    move = actions.add_parser("move", help="move an entry to a new position")
    move.add_argument("playlist", metavar="NAME|ID")
    move.add_argument("position", type=int)
    move.add_argument("new_position", type=int)
    remove = actions.add_parser("remove", help="remove an entry")
    remove.add_argument("playlist", metavar="NAME|ID")
    remove.add_argument("position", type=int)
    startup = actions.add_parser("set-startup", help="make a playlist the one loaded at startup")
    startup.add_argument("playlist", metavar="NAME|ID")
    delete = actions.add_parser("delete", help="delete a playlist and its entries")
    delete.add_argument("playlist", metavar="NAME|ID")

    ledwalk = sub.add_parser("ledwalk", help="light one LED at a time on a tile to verify LED 0 and the winding")
    ledwalk.add_argument("--row", type=int, default=0, help="row address (default 0)")
    ledwalk.add_argument("--slot", type=int, default=0, help="slot in the row (default 0)")
    ledwalk.add_argument("--tile", type=int, help="tile index 0-63, instead of --row/--slot")
    ledwalk.add_argument("--delay", type=float, default=0.5, metavar="SECONDS", help="per LED (default 0.5)")
    ledwalk.add_argument("--color", type=parse_color, default=(255, 255, 255), metavar="R,G,B")
    _add_output_flags(ledwalk, frames_help="stop after N LEDs")
    ledwalk.set_defaults(func=_cmd_ledwalk)

    tilewalk = sub.add_parser("tilewalk", help="light one tile at a time in floor order to verify the install")
    tilewalk.add_argument("--delay", type=float, default=0.5, metavar="SECONDS", help="per tile (default 0.5)")
    tilewalk.add_argument("--color", type=parse_color, default=(255, 255, 255), metavar="R,G,B")
    _add_output_flags(tilewalk, frames_help="stop after N tiles")
    tilewalk.set_defaults(func=_cmd_tilewalk)

    scan = sub.add_parser("scan", help="find responding row controllers on every chain")
    scan.set_defaults(func=_cmd_scan)
    status = sub.add_parser("status", help="query STATUS from a row controller")
    status.add_argument("row", type=int, help="logical row (0-7)")
    status.set_defaults(func=_cmd_status)
    version = sub.add_parser("version", help="query VERSION from every row/tile and flag anything out of step")
    version.set_defaults(func=_cmd_version)
    blackout = sub.add_parser("blackout", help="black out the whole floor")
    blackout.set_defaults(func=_cmd_blackout)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if getattr(args, "brightness", None) is not None and not 0 <= args.brightness <= 255:
        parser.error("--brightness must be 0..255")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
