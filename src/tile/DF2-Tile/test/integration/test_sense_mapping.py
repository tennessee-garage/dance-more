#!/usr/bin/env python3
"""
Integration test: SENSE chain auto-mapping.

Starts tile-bus-broker and 8 mock-tile processes, then drives the SENSE
auto-mapping sequence as a mock row controller.

Protocol: the RC passes the "active" SENSE signal down the chain one tile at
a time.  After each tile is identified, the RC must explicitly release the
signal driving that tile's SENSE_IN before advancing, otherwise every
previously-identified tile would continue to respond to DETECT_SENSE.  In
real hardware the RC releases its GPIO; in this simulation it sends a
SENSE_RELEASE (for slot 0) or a unicast CLEAR_SENSE to the upstream tile (for
slots 1-7).

End of chain: after the last tile asserts its SENSE_OUT (with no tile
downstream to receive it), the next DETECT_SENSE times out.  That timeout is
the expected signal that auto-mapping is complete.

Requires (build before running):
  pio run -e native     ->  .pio/build/native/program  (mock-tile)
  make -C tools         ->  tools/bin/tile-bus-broker

Usage:
  python3 test/integration/test_sense_mapping.py
"""

import os
import sys
import time
import socket
import select
import subprocess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, '..', '..'))
BROKER   = os.path.join(_PROJECT, 'tools', 'bin', 'tile-bus-broker')
TILE     = os.path.join(_PROJECT, '.pio', 'build', 'native', 'program')
ROW      = 0
SOCK     = f'/tmp/df2-tile-bus-{ROW}.sock'

# ---------------------------------------------------------------------------
# Protocol constants (mirrors broker_msg.h / protocol.h)
# ---------------------------------------------------------------------------
SYNC1 = 0xAA
SYNC2 = 0x55
ADDR_BROADCAST = 0xFF

CMD_ACTIVATE_SENSE = 0x01
CMD_DETECT_SENSE   = 0x02
CMD_CLEAR_SENSE    = 0x03
CMD_ACK            = 0x80
CMD_DETECT_RESP    = 0x82

MSG_FRAME             = 0x01
MSG_IDENTIFY          = 0x02
MSG_SENSE_ASSERT      = 0x03
MSG_SENSE_RELEASE     = 0x04
MSG_SENSE_IS_ASSERTED = 0x05
MSG_SENSE_DEASSERTED  = 0x06

# ---------------------------------------------------------------------------
# Tile bus frame helpers
# ---------------------------------------------------------------------------
def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
        crc &= 0xFFFF
    return crc

def _encode(addr: int, cmd: int, payload: bytes = b'') -> bytes:
    body = bytes([addr, cmd, len(payload)]) + payload
    c = _crc16(body)
    return bytes([SYNC1, SYNC2]) + body + bytes([c >> 8, c & 0xFF])

def _decode(raw: bytes) -> dict | None:
    if len(raw) < 7 or raw[0] != SYNC1 or raw[1] != SYNC2:
        return None
    addr, cmd, plen = raw[2], raw[3], raw[4]
    if len(raw) != 7 + plen:
        return None
    payload = raw[5:5 + plen]
    expected = _crc16(raw[2:5 + plen])
    received = (raw[5 + plen] << 8) | raw[6 + plen]
    if expected != received:
        return None
    return {'addr': addr, 'cmd': cmd, 'payload': bytes(payload)}

# ---------------------------------------------------------------------------
# Row-controller connection to the broker
# ---------------------------------------------------------------------------
class RowController:
    """Speaks the broker socket protocol as the row controller (slot=0xFF).

    Slot 0xFF causes uint8_t wrap in the broker's sense routing so that
    SENSE_ASSERT from the row controller reaches tile slot 0.
    """

    def __init__(self, sock_path: str):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(sock_path)
        self._buf  = b''
        # Identify as row controller: addr=0x00, slot=0xFF
        self._sock.sendall(bytes([MSG_IDENTIFY, 0x00, 0xFF]))

    # ---- Sideband control ----

    def sense_assert(self):
        """Assert SENSE into the chain; broker routes this to slot 0."""
        self._sock.sendall(bytes([MSG_SENSE_ASSERT]))

    def sense_release(self):
        """Release SENSE; broker sends SENSE_DEASSERTED to slot 0."""
        self._sock.sendall(bytes([MSG_SENSE_RELEASE]))

    # ---- Tile-bus frame TX ----

    def send(self, addr: int, cmd: int, payload: bytes = b''):
        raw = _encode(addr, cmd, payload)
        self._sock.sendall(bytes([MSG_FRAME, len(raw)]) + raw)

    # ---- Tile-bus frame RX ----

    def recv(self, timeout: float) -> dict | None:
        """Return the next tile-bus frame from the broker, or None on timeout.
        Broker sideband messages (SENSE_IS_ASSERTED etc.) are discarded."""
        deadline = time.monotonic() + timeout
        while True:
            f = self._parse()
            if f is not None:
                return f
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            r, _, _ = select.select([self._sock], [], [], remaining)
            if not r:
                return None
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError('broker disconnected')
            self._buf += chunk

    def recv_cmd(self, cmd: int, timeout: float) -> dict | None:
        """Receive frames until one with the expected cmd arrives, or timeout.
        Frames with other commands are discarded (they shouldn't appear in
        normal operation but guards against leftover traffic)."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            f = self.recv(remaining)
            if f is None:
                return None
            if f['cmd'] == cmd:
                return f

    def _parse(self) -> dict | None:
        """Consume broker messages from the internal buffer and return the first
        complete tile-bus frame found, or None if the buffer is not yet ready."""
        while self._buf:
            t = self._buf[0]
            if t == MSG_FRAME:
                if len(self._buf) < 2:
                    return None
                n = self._buf[1]
                if len(self._buf) < 2 + n:
                    return None
                frame_raw   = self._buf[2:2 + n]
                self._buf   = self._buf[2 + n:]
                decoded     = _decode(frame_raw)
                if decoded:
                    return decoded
            elif t in (MSG_SENSE_IS_ASSERTED, MSG_SENSE_DEASSERTED):
                self._buf = self._buf[1:]   # 1-byte message, no payload
            else:
                self._buf = self._buf[1:]   # unknown type, skip
        return None

    def close(self):
        self._sock.close()

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

# 8 tiles: addr = slot + 1 makes the expected mapping unambiguous.
TILES = [(0x01 + i, i) for i in range(8)]   # [(addr, slot), ...]

DETECT_TIMEOUT   = 0.5   # seconds to wait for DETECT_RESP
ACTIVATE_TIMEOUT = 0.5   # seconds to wait for ACTIVATE_SENSE ACK

def run_sense_mapping() -> bool:
    rc = RowController(SOCK)

    # Ensure clean state from any previous run.
    rc.send(ADDR_BROADCAST, CMD_CLEAR_SENSE)
    rc.sense_release()
    time.sleep(0.02)

    print('SENSE auto-mapping sequence')
    print()
    print('  Each round: detect tile, activate sense, release upstream.')
    print('  End of chain: DETECT_SENSE times out (no tile at slot 8).')
    print()

    # The row controller sits at slot 0xFF.  Its SENSE_ASSERT routes to slot 0
    # because uint8_t(0xFF + 1) == 0x00.
    print('  [RC] SENSE_ASSERT  ->  slot 0 (start of chain)')
    rc.sense_assert()
    time.sleep(0.02)   # let tile 0 process MSG_SENSE_IS_ASSERTED

    mapping: list[tuple[int, int]] = []   # (slot_index, tile_addr)
    failures: list[str] = []

    while True:
        slot_idx = len(mapping)

        print(f'  [RC] DETECT_SENSE broadcast'
              f'  (expecting tile at slot {slot_idx})', end='  ', flush=True)

        rc.send(ADDR_BROADCAST, CMD_DETECT_SENSE)
        resp = rc.recv_cmd(CMD_DETECT_RESP, timeout=DETECT_TIMEOUT)

        if resp is None:
            # No response within timeout: end of chain.  This is the expected
            # outcome after the last tile asserts sense out with no downstream
            # tile to receive it.
            print(f'-> timeout  [end of chain after slot {slot_idx - 1}]')
            break

        tile_addr = resp['addr']
        mapping.append((slot_idx, tile_addr))
        print(f'-> DETECT_RESP  addr=0x{tile_addr:02X}')

        # Tell this tile to assert its outgoing SENSE line so the next tile
        # in the chain sees sense_in asserted.
        print(f'  [RC] ACTIVATE_SENSE  ->  0x{tile_addr:02X}', end='  ', flush=True)
        rc.send(tile_addr, CMD_ACTIVATE_SENSE)
        ack = rc.recv_cmd(CMD_ACK | CMD_ACTIVATE_SENSE, timeout=ACTIVATE_TIMEOUT)

        if ack is None:
            msg = f'no ACK from 0x{tile_addr:02X} (slot {slot_idx})'
            print(f'-> {msg}')
            failures.append(msg)
        else:
            print(f'-> ACK  status=0x{ack["payload"][0]:02X}')

        # Release the sense_in for this tile so it won't respond to future
        # DETECT_SENSE broadcasts.
        #
        # Slot 0: the RC drives sense_in directly via SENSE_ASSERT/RELEASE.
        #         Send SENSE_RELEASE to clear it.
        #
        # Slot N>0: sense_in is driven by the upstream tile's (slot N-1)
        #           SENSE_OUT.  Send a unicast CLEAR_SENSE to that upstream
        #           tile so it releases its SENSE_OUT, which causes the broker
        #           to send SENSE_DEASSERTED to this tile, clearing sense_in.
        if slot_idx == 0:
            print(f'  [RC] SENSE_RELEASE  (clear tile slot 0 sense_in)', flush=True)
            rc.sense_release()
        else:
            upstream_addr = mapping[-2][1]   # tile at slot_idx - 1
            print(f'  [RC] CLEAR_SENSE  ->  0x{upstream_addr:02X}'
                  f'  (release upstream sense_out, clears slot {slot_idx} sense_in)',
                  flush=True)
            rc.send(upstream_addr, CMD_CLEAR_SENSE)

        # Give the SENSE_DEASSERTED message time to propagate to the tile
        # before we send the next DETECT_SENSE.
        time.sleep(0.02)

    # Release all remaining sense lines.
    print()
    print('  [RC] CLEAR_SENSE broadcast  (cleanup)')
    rc.send(ADDR_BROADCAST, CMD_CLEAR_SENSE)
    time.sleep(0.02)
    rc.close()

    # ---- Verify results ----
    print()
    print(f'Mapping ({len(mapping)} tile(s) discovered):')
    for slot_idx, addr in mapping:
        expected = TILES[slot_idx][0] if slot_idx < len(TILES) else '?'
        status = 'ok' if addr == expected else f'FAIL (expected 0x{expected:02X})'
        print(f'  slot {slot_idx}  ->  addr 0x{addr:02X}  [{status}]')

    if len(mapping) != len(TILES):
        failures.append(
            f'expected {len(TILES)} tiles, discovered {len(mapping)}'
        )

    for slot_idx, addr in mapping:
        if slot_idx >= len(TILES):
            failures.append(f'unexpected extra slot {slot_idx} (addr 0x{addr:02X})')
            continue
        expected = TILES[slot_idx][0]
        if addr != expected:
            failures.append(
                f'slot {slot_idx}: expected 0x{expected:02X}, got 0x{addr:02X}'
            )

    print()
    if failures:
        for msg in failures:
            print(f'  FAIL: {msg}')
        return False

    print('  PASS')
    return True


def main() -> int:
    for path, hint in [
        (BROKER, 'run: make -C tools'),
        (TILE,   'run: pio run -e native'),
    ]:
        if not os.path.isfile(path):
            print(f'ERROR: binary not found: {path}')
            print(f'       {hint}')
            return 1

    procs: list[subprocess.Popen] = []
    try:
        # Start broker
        procs.append(subprocess.Popen(
            [BROKER, str(ROW)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ))
        time.sleep(0.15)   # wait for the Unix socket to be created

        # Start 8 tile processes
        for addr, slot in TILES:
            procs.append(subprocess.Popen(
                [TILE, f'0x{addr:02X}', str(slot), str(ROW)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ))

        # Wait for all tiles to connect and identify themselves with the broker.
        time.sleep(0.5)

        passed = run_sense_mapping()
        return 0 if passed else 1

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
