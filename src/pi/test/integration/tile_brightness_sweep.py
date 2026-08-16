#!/usr/bin/env python3
"""
Measurement tool: current draw vs. commanded brightness, per color.

Run this on a Raspberry Pi wired to the Row Bus through the Pi hat, with one
row controller and at least one tile attached. Steps SET_COLOR from 0-100%
brightness for red, green, blue, and white, and records the row controller's
INA226 current at each point - see docs/measurements/README.md for what this
is for and how the results are organized, and power_monitor_rp2350.cpp for
the register-level detail this depends on being decoded correctly.

Every reading is a mean of several POWER requests with the first discarded
as a settling sample, and every reading has the blacked-out baseline (row
controller + tile quiescent current, LEDs off) subtracted, so the CSV
isolates what the commanded color/level actually costs.

This targets whatever is plugged into --row - a different tile, a different
LED strip entirely (subject to the row controller still recognizing it as a
Tile Bus device), or a future PCB revision. Re-run it whenever that changes
and give the output a new --label so old and new data don't collide.

Usage:
  # two-chain hat (defaults: ttyAMA0/GPIO17 + ttyAMA2/GPIO7)
  python3 test/integration/tile_brightness_sweep.py --label my-strip

  # single-chain bench board, XDIR still on GPIO23
  python3 test/integration/tile_brightness_sweep.py --chain /dev/ttyAMA0:23 --label my-strip
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from df2_pi.protocol.constants import Cmd, TileCmd
from df2_pi.transport import ChainConfig, Floor, RowChainMap, default_chain_configs
from df2_pi.transport.row_bus import DEFAULT_BAUDRATE

COLORS = {
    "red":   lambda level: (level, 0, 0),
    "green": lambda level: (0, level, 0),
    "blue":  lambda level: (0, 0, level),
    "white": lambda level: (level, level, level),
}

FIELDNAMES = ["color", "pct", "level_0_255", "voltage_mV", "current_mA",
              "current_stdev_mA", "power_mW", "current_delta_mA", "power_delta_mW"]


def color_payload(r: int, g: int, b: int) -> bytes:
    return bytes([TileCmd.SET_COLOR, r, g, b]) * 8


def sample(floor: Floor, row: int, reads: int, gap_s: float) -> tuple[float, float, float, float]:
    """`reads` POWER requests, first discarded as settling, rest averaged."""
    readings = []
    for i in range(reads):
        p = floor.request(row, Cmd.POWER).payload
        v, c, w = (p[0] << 8) | p[1], (p[2] << 8) | p[3], (p[4] << 8) | p[5]
        if i > 0:
            readings.append((v, c, w))
        time.sleep(gap_s)
    vs, cs, ws = zip(*readings)
    return (statistics.mean(vs), statistics.mean(cs), statistics.mean(ws), statistics.pstdev(cs))


def make_row(color: str, pct: int, level: int, v: float, c: float, c_std: float, w: float,
             base_c: float, base_w: float) -> dict:
    return {
        "color": color, "pct": pct, "level_0_255": level,
        "voltage_mV": round(v, 1), "current_mA": round(c, 2),
        "current_stdev_mA": round(c_std, 3), "power_mW": round(w, 1),
        "current_delta_mA": round(c - base_c, 2),
        "power_delta_mW": round(w - base_w, 1),
    }


def run_sweep(floor: Floor, row: int, step_pct: int, settle_s: float,
              reads_per_point: int, read_gap_s: float) -> tuple[list[dict], dict]:
    floor.blackout()
    time.sleep(0.6)
    base_v, base_c, base_w, base_c_std = sample(floor, row, reads_per_point, read_gap_s)
    print(f"baseline: {base_v:.0f} mV  {base_c:.1f} mA (stdev {base_c_std:.2f})  {base_w:.0f} mW")

    rows = [make_row("baseline", 0, 0, base_v, base_c, base_c_std, base_w, base_c, base_w)]

    for color, rgb_fn in COLORS.items():
        for pct in range(0, 101, step_pct):
            level = round(pct / 100 * 255)
            r, g, b = rgb_fn(level)
            floor.send_data(row, color_payload(r, g, b))
            floor.latch()
            time.sleep(settle_s)
            v, c, w, c_std = sample(floor, row, reads_per_point, read_gap_s)
            data_row = make_row(color, pct, level, v, c, c_std, w, base_c, base_w)
            rows.append(data_row)
            print(f"{color:6s} {pct:3d}%  level={level:3d}  "
                  f"current={c:6.1f}mA  delta={c - base_c:6.1f}mA  "
                  f"power_delta={w - base_w:6.1f}mW", flush=True)

    floor.blackout()
    baseline = {"voltage_mV": base_v, "current_mA": base_c, "power_mW": base_w}
    return rows, baseline


def parse_chain(spec: str, baudrate: int) -> ChainConfig:
    port, _, xdir = spec.partition(":")
    if not xdir:
        raise argparse.ArgumentTypeError(f"--chain needs PORT:XDIR_GPIO, got {spec!r}")
    pin = None if xdir.lower() in ("none", "off") else int(xdir)
    return ChainConfig(port, pin, baudrate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep 0-100% brightness per color, recording INA226 current")
    parser.add_argument("--chain", action="append", metavar="PORT:XDIR_GPIO",
                        help="repeatable; defaults to the two-chain hat's ttyAMA0:17 and ttyAMA2:7")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE,
                        help=f"default: {DEFAULT_BAUDRATE}")
    parser.add_argument("--row", type=int, default=0, help="logical row to sweep (0-7)")
    parser.add_argument("--label", required=True,
                        help="identifies what was measured (e.g. 'tile-v1', 'acme-strip'); "
                             "used to name the output file so runs don't collide")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/measurements"),
                        help="default: docs/measurements (relative to cwd)")
    parser.add_argument("--step-pct", type=int, default=5, help="brightness step size (default: 5)")
    parser.add_argument("--settle", type=float, default=0.25,
                        help="seconds to let current settle after LATCH before sampling (default: 0.25)")
    parser.add_argument("--reads-per-point", type=int, default=5,
                        help="POWER requests per data point; first is discarded (default: 5)")
    parser.add_argument("--read-gap", type=float, default=0.08,
                        help="seconds between reads within a point (default: 0.08)")
    args = parser.parse_args()

    chains = ([parse_chain(c, args.baudrate) for c in args.chain]
              if args.chain else default_chain_configs(args.baudrate))
    chain_map = RowChainMap.single_chain() if len(chains) == 1 else RowChainMap.alternating(len(chains))

    with Floor(chains=chains, chain_map=chain_map) as floor:
        try:
            floor.request(args.row, Cmd.STATUS)
        except Exception as exc:
            print(f"row 0x{args.row:02X} is not responding: {exc}", file=sys.stderr)
            return 1

        rows, baseline = run_sweep(floor, args.row, args.step_pct, args.settle,
                                    args.reads_per_point, args.read_gap)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%Y-%m-%d")
    out_path = args.out_dir / f"{date}-brightness-sweep-{args.label}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
