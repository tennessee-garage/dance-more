"""Command-line entry point for the Row Bus host controller."""

from __future__ import annotations

import argparse
import sys

from .protocol.constants import Cmd
from .transport.chain_map import RowChainMap
from .transport.floor import ChainConfig, Floor, RowNotResponding, default_chain_configs
from .transport.row_bus import DEFAULT_BAUDRATE

STATUS_STATE_NAMES = {0x00: "idle", 0x01: "discovering", 0x02: "running", 0x03: "error"}


def _open_floor(args: argparse.Namespace) -> Floor:
    if args.chain:
        chains = [_parse_chain(spec, args.baudrate) for spec in args.chain]
    else:
        chains = default_chain_configs(args.baudrate)
    chain_map = (
        RowChainMap.single_chain() if len(chains) == 1 else RowChainMap.alternating(len(chains))
    )
    return Floor(chains=chains, chain_map=chain_map)


def _parse_chain(spec: str, baudrate: int) -> ChainConfig:
    """PORT[:XDIR_GPIO] - e.g. /dev/ttyAMA0:17, or /dev/ttyAMA0:none"""
    port, _, xdir = spec.partition(":")
    if not xdir:
        raise argparse.ArgumentTypeError(f"--chain needs PORT:XDIR_GPIO, got {spec!r}")
    pin = None if xdir.lower() in ("none", "off") else int(xdir)
    return ChainConfig(port, pin, baudrate)


def _cmd_status(args: argparse.Namespace) -> int:
    with _open_floor(args) as floor:
        try:
            frame = floor.request(args.row, Cmd.STATUS)
        except RowNotResponding as exc:
            print(exc, file=sys.stderr)
            return 1
        state, tiles = frame.payload[0], frame.payload[1]
        print(f"row 0x{frame.addr:02X} (chain {floor.chain_map.chain_for(args.row)}): "
              f"state={STATUS_STATE_NAMES.get(state, hex(state))} tiles_found={tiles}")
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


def _cmd_blackout(args: argparse.Namespace) -> int:
    with _open_floor(args) as floor:
        floor.blackout()
        print("blackout sent to all chains")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="df2-pi", description="Dance Floor v2 Row Bus host controller"
    )
    parser.add_argument(
        "--chain",
        action="append",
        metavar="PORT:XDIR_GPIO",
        help="repeatable; defaults to the two-chain hat's ttyAMA0:17 and ttyAMA2:7",
    )
    parser.add_argument(
        "--baudrate", type=int, default=DEFAULT_BAUDRATE, help=f"baud rate (default: {DEFAULT_BAUDRATE})"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="find responding row controllers on every chain")
    scan.set_defaults(func=_cmd_scan)

    status = sub.add_parser("status", help="query STATUS from a row controller")
    status.add_argument("row", type=int, help="logical row (0-7)")
    status.set_defaults(func=_cmd_status)

    blackout = sub.add_parser("blackout", help="black out the whole floor")
    blackout.set_defaults(func=_cmd_blackout)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
