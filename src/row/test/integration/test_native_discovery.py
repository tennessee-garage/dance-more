#!/usr/bin/env python3
"""
Integration test: native row controller vs. the tile project's tile-bus-broker.

Starts tile-bus-broker and 8 native mock-tile processes (both from the tile
project), then runs this project's own native row-controller binary against
the same broker socket, exercising the real TileTransportNative +
RowSenseNative + SenseMapper stack end to end - no hardware, no mocked RC.

Always rebuilds all three artifacts first (tile-bus-broker via `make`, both
native binaries via `pio run -e native`) rather than just checking they
exist - `pio test -e native` overwrites the same binary path with its own
test binary, so an existing file is not proof it's the right program.
`make`/`pio run` are incremental, so this is cheap when nothing's changed.

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


def ensure_built() -> bool:
    """Always (re)build all three artifacts - never just check for
    presence. `pio test -e native` overwrites the same
    .pio/build/native/program path with its own test binary, so a file that
    merely *exists* isn't proof it's the right one; `make`/`pio run` are
    incremental and near-instant when nothing's actually changed, so paying
    for a real build here is cheap and guarantees freshness either way.
    Returns False (with an error printed) if a build step fails."""
    print('Building tile-bus-broker...')
    result = subprocess.run(['make'], cwd=os.path.join(_TILE_PROJECT, 'tools'))
    if result.returncode != 0 or not os.path.isfile(BROKER):
        print(f'ERROR: failed to build {BROKER}')
        return False

    print('Building tile project native mock (pio run -e native)...')
    result = subprocess.run(['pio', 'run', '-e', 'native'], cwd=_TILE_PROJECT)
    if result.returncode != 0 or not os.path.isfile(TILE_BIN):
        print(f'ERROR: failed to build {TILE_BIN}')
        return False

    print('Building row controller native binary (pio run -e native)...')
    result = subprocess.run(['pio', 'run', '-e', 'native'], cwd=_ROW_PROJECT)
    if result.returncode != 0 or not os.path.isfile(ROW_BIN):
        print(f'ERROR: failed to build {ROW_BIN}')
        return False

    return True


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
    if not ensure_built():
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
