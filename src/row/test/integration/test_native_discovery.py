#!/usr/bin/env python3
"""
Integration test: native row controller vs. the tile project's tile-bus-broker.

Starts tile-bus-broker and 8 native mock-tile processes (both from the tile
project), then runs this project's own native row-controller binary against
the same broker socket, exercising the real TileTransportNative +
RowSenseNative + SenseMapper stack end to end - no hardware, no mocked RC.

Requires (build before running):
  cd ../../../tile && pio run -e native   ->  .pio/build/native/program (mock-tile)
  cd ../../../tile/tools && make          ->  tools/bin/tile-bus-broker
  pio run -e native                       ->  .pio/build/native/program (row controller)

Usage:
  python3 test/integration/test_native_discovery.py
"""

import os
import re
import sys
import time
import subprocess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_ROW_PROJECT  = os.path.abspath(os.path.join(_HERE, '..', '..'))
_TILE_PROJECT = os.path.abspath(os.path.join(_ROW_PROJECT, '..', 'tile'))

BROKER   = os.path.join(_TILE_PROJECT, 'tools', 'bin', 'tile-bus-broker')
TILE_BIN = os.path.join(_TILE_PROJECT, '.pio', 'build', 'native', 'program')
ROW_BIN  = os.path.join(_ROW_PROJECT, '.pio', 'build', 'native', 'program')

ROW  = 0
SOCK = f'/tmp/df2-tile-bus-{ROW}.sock'

# 8 tiles: addr = slot + 1, matching test_sense_mapping.py's convention.
TILES = [(0x01 + i, i) for i in range(8)]

ROW_BIN_TIMEOUT = 5.0  # seconds to let the row controller finish discovery

_SLOT_LINE = re.compile(r'slot (\d+) -> addr 0x([0-9A-Fa-f]+)')


def run_discovery() -> bool:
    proc = subprocess.run(
        [ROW_BIN, str(ROW)],
        capture_output=True,
        text=True,
        timeout=ROW_BIN_TIMEOUT,
    )
    print(proc.stdout, end='')
    if proc.stderr:
        print(proc.stderr, end='', file=sys.stderr)

    found = {int(slot): int(addr, 16) for slot, addr in _SLOT_LINE.findall(proc.stdout)}

    print()
    print(f'Mapping ({len(found)} tile(s) discovered):')
    failures = []
    for expected_addr, slot in TILES:
        actual = found.get(slot)
        status = 'ok' if actual == expected_addr else f'FAIL (expected 0x{expected_addr:02X})'
        print(f'  slot {slot}  ->  addr 0x{actual:02X}  [{status}]' if actual is not None
              else f'  slot {slot}  ->  MISSING  [FAIL]')
        if actual != expected_addr:
            failures.append(f'slot {slot}: expected 0x{expected_addr:02X}, got {actual}')

    extra = set(found) - {slot for _, slot in TILES}
    for slot in extra:
        failures.append(f'unexpected extra slot {slot} (addr 0x{found[slot]:02X})')

    print()
    if failures or proc.returncode != 0:
        if proc.returncode != 0:
            failures.append(f'row controller exited {proc.returncode}')
        for msg in failures:
            print(f'  FAIL: {msg}')
        return False

    print('  PASS')
    return True


def main() -> int:
    for path, hint in [
        (BROKER,   'run: make -C ../tile/tools'),
        (TILE_BIN, 'run: (cd ../tile && pio run -e native)'),
        (ROW_BIN,  'run: pio run -e native'),
    ]:
        if not os.path.isfile(path):
            print(f'ERROR: binary not found: {path}')
            print(f'       {hint}')
            return 1

    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass

    procs: list[subprocess.Popen] = []
    try:
        procs.append(subprocess.Popen(
            [BROKER, str(ROW)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ))
        time.sleep(0.15)  # wait for the Unix socket to be created

        for addr, slot in TILES:
            procs.append(subprocess.Popen(
                [TILE_BIN, f'0x{addr:02X}', str(slot), str(ROW)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ))

        time.sleep(0.5)  # let all 8 tiles connect and IDENTIFY with the broker

        return 0 if run_discovery() else 1

    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
        try:
            os.unlink(SOCK)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    sys.exit(main())
