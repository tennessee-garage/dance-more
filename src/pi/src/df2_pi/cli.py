"""Command-line entry point for the Row Bus host controller."""

from __future__ import annotations

import argparse
import sys

from .protocol.constants import Cmd
from .transport.row_bus import DEFAULT_BAUDRATE, DEFAULT_PORT, RowBus


def _cmd_status(args: argparse.Namespace) -> int:
    with RowBus(port=args.port, baudrate=args.baudrate) as bus:
        bus.send(args.row, Cmd.STATUS)
        frame = bus.read_frame(timeout=0.5)
        if frame is None:
            print(f"row {args.row}: no response", file=sys.stderr)
            return 1
        print(f"row {frame.addr}: {frame.payload.hex()}")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="df2-pi", description="Dance Floor v2 Row Bus host controller"
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"serial port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--baudrate", type=int, default=DEFAULT_BAUDRATE, help=f"baud rate (default: {DEFAULT_BAUDRATE})"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="query STATUS from a row controller")
    status.add_argument("row", type=int, help="row address (0-7)")
    status.set_defaults(func=_cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
