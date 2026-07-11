#!/usr/bin/env python3
"""
Integration test: Row Bus multidrop relay (row-bus-broker + PiTransportNative).

Starts row-bus-broker and 8 row-stub processes (one per row address 0x00-
0x07), then acts as a mock Pi: connects to the same broker socket and
exercises the broker's broadcast-and-local-filter model, mirroring how
test_sense_mapping.py drives the tile project's tile-bus-broker.

Verifies:
  - Broadcasting STATUS (addr=0xFF) reaches all 8 row-stubs, each replying
    with its own STATUS_RESP, distinguishable by ADDR.
  - Unicasting STATUS to a single row address is answered only by that row,
    not the other 7 - proving the broker relays to everyone and each client
    filters locally, like real RS-485 multidrop, not point-to-point.

Always rebuilds tools/bin/{row-bus-broker,row-stub} first (via `make -C
tools`) rather than just checking they exist, for the same reason
test_native_discovery.py always rebuilds: a stale binary left over from a
previous build can silently masquerade as current.

Usage:
  python3 test/integration/test_row_bus_broadcast.py
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
TOOLS    = os.path.join(_PROJECT, 'tools')
BROKER   = os.path.join(TOOLS, 'bin', 'row-bus-broker')
STUB     = os.path.join(TOOLS, 'bin', 'row-stub')
SOCK     = '/tmp/df2-row-bus.sock'

ROWS = list(range(8))  # row addresses 0x00-0x07

# ---------------------------------------------------------------------------
# Protocol constants (mirrors row_broker_msg.h / row_bus_protocol.h)
# ---------------------------------------------------------------------------
SYNC1 = 0xAA
SYNC2 = 0x55
ADDR_BROADCAST = 0xFF

CMD_STATUS      = 0x02
CMD_STATUS_RESP = 0x82

MSG_FRAME    = 0x01
MSG_IDENTIFY = 0x02

# ---------------------------------------------------------------------------
# Row Bus frame helpers (2-byte LEN, unlike tile_bus_protocol's 1-byte LEN)
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
    body = bytes([addr, cmd, len(payload) >> 8, len(payload) & 0xFF]) + payload
    c = _crc16(body)
    return bytes([SYNC1, SYNC2]) + body + bytes([c >> 8, c & 0xFF])

def _decode(raw: bytes) -> dict | None:
    if len(raw) < 8 or raw[0] != SYNC1 or raw[1] != SYNC2:
        return None
    addr, cmd = raw[2], raw[3]
    plen = (raw[4] << 8) | raw[5]
    if len(raw) != 8 + plen:
        return None
    payload = raw[6:6 + plen]
    expected = _crc16(raw[2:6 + plen])
    received = (raw[6 + plen] << 8) | raw[7 + plen]
    if expected != received:
        return None
    return {'addr': addr, 'cmd': cmd, 'payload': bytes(payload)}

# ---------------------------------------------------------------------------
# Mock-Pi connection to the broker
# ---------------------------------------------------------------------------
class MockPi:
    """Speaks the broker socket protocol as the Pi (row_addr=0xFF sentinel)."""

    def __init__(self, sock_path: str):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(sock_path)
        self._buf = b''
        self._sock.sendall(bytes([MSG_IDENTIFY, ADDR_BROADCAST]))

    def send(self, addr: int, cmd: int, payload: bytes = b''):
        raw = _encode(addr, cmd, payload)
        n = len(raw)
        self._sock.sendall(bytes([MSG_FRAME, n >> 8, n & 0xFF]) + raw)

    def recv_all(self, timeout: float) -> list[dict]:
        """Collect every Row Bus frame that arrives within timeout."""
        frames = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            r, _, _ = select.select([self._sock], [], [], remaining)
            if not r:
                break
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError('broker disconnected')
            self._buf += chunk
            frames.extend(self._drain())
        return frames

    def _drain(self) -> list[dict]:
        out = []
        while self._buf:
            if self._buf[0] != MSG_FRAME:
                self._buf = self._buf[1:]  # unknown type, skip
                continue
            if len(self._buf) < 3:
                break
            n = (self._buf[1] << 8) | self._buf[2]
            if len(self._buf) < 3 + n:
                break
            frame_raw = self._buf[3:3 + n]
            self._buf = self._buf[3 + n:]
            decoded = _decode(frame_raw)
            if decoded:
                out.append(decoded)
        return out

    def close(self):
        self._sock.close()

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
RESPONSE_TIMEOUT = 0.5  # seconds to collect replies

def run_test() -> bool:
    pi = MockPi(SOCK)
    failures: list[str] = []

    # ---- Broadcast STATUS: all 8 rows should reply ----
    print('[Pi] STATUS broadcast (addr=0xFF)')
    pi.send(ADDR_BROADCAST, CMD_STATUS)
    frames = pi.recv_all(RESPONSE_TIMEOUT)
    replies = {f['addr'] for f in frames if f['cmd'] == CMD_STATUS_RESP}
    print(f'  replies from: {sorted(hex(a) for a in replies)}')

    missing = set(ROWS) - replies
    extra = replies - set(ROWS)
    if missing:
        failures.append(f'broadcast: no reply from row(s) {sorted(hex(a) for a in missing)}')
    if extra:
        failures.append(f'broadcast: unexpected reply from {sorted(hex(a) for a in extra)}')

    # ---- Unicast STATUS to row 0x03: only that row should reply ----
    target = 0x03
    print(f'[Pi] STATUS unicast -> 0x{target:02X}')
    pi.send(target, CMD_STATUS)
    frames = pi.recv_all(RESPONSE_TIMEOUT)
    replies = {f['addr'] for f in frames if f['cmd'] == CMD_STATUS_RESP}
    print(f'  replies from: {sorted(hex(a) for a in replies)}')

    if replies != {target}:
        failures.append(f'unicast to 0x{target:02X}: expected reply only from that row, got {sorted(hex(a) for a in replies)}')

    pi.close()

    print()
    if failures:
        for msg in failures:
            print(f'  FAIL: {msg}')
        return False

    print('  PASS')
    return True


def main() -> int:
    print('Building row-bus-broker and row-stub...')
    result = subprocess.run(['make'], cwd=TOOLS)
    if result.returncode != 0:
        print('ERROR: build failed')
        return 1
    for path in (BROKER, STUB):
        if not os.path.isfile(path):
            print(f'ERROR: binary not found after build: {path}')
            return 1

    procs: list[subprocess.Popen] = []
    try:
        procs.append(subprocess.Popen(
            [BROKER], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ))
        time.sleep(0.15)  # wait for the Unix socket to be created

        for row in ROWS:
            procs.append(subprocess.Popen(
                [STUB, str(row)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            ))

        time.sleep(0.5)  # let all stubs connect and identify

        passed = run_test()
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
