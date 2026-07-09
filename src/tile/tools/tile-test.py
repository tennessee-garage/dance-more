#!/usr/bin/env python3
"""
tools/tile-test.py — drive a single native tile process for visual testing.

Modes (mutually exclusive, one required):
  --color R,G,B | #RRGGBB   set solid color on all 40 LEDs
  --stdin                    read exactly 40 colors from stdin
                               (one per line, R,G,B or #RRGGBB)
  --test                     animated test pattern cycling through solid
                               colors and animation effects

Requires (build before running):
  pio run -e native   ->  .pio/build/native/program
  make -C tools       ->  tools/bin/tile-bus-broker

Usage:
  python3 tools/tile-test.py --color 255,0,0
  python3 tools/tile-test.py --color '#FF0000'
  printf '255,0,0\\n...' | python3 tools/tile-test.py --stdin
  python3 tools/tile-test.py --test
"""

import argparse
import colorsys
import math
import os
import socket
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, '..'))
BROKER   = os.path.join(_PROJECT, 'tools', 'bin', 'tile-bus-broker')
TILE     = os.path.join(_PROJECT, '.pio', 'build', 'native', 'program')

TILE_ADDR = 0x01
TILE_SLOT = 0
ROW       = 0
SOCK      = f'/tmp/df2-tile-bus-{ROW}.sock'

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
SYNC1, SYNC2   = 0xAA, 0x55
ADDR_BROADCAST = 0xFF

CMD_SET_COLOR  = 0x10
CMD_SET_LEDS   = 0x12
CMD_LATCH      = 0x13

MSG_FRAME    = 0x01
MSG_IDENTIFY = 0x02

NUM_LEDS = 40

# ---------------------------------------------------------------------------
# Frame helpers
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

# ---------------------------------------------------------------------------
# Row-controller connection
# ---------------------------------------------------------------------------
class RC:
    def __init__(self, sock_path: str):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(sock_path)
        self._sock.sendall(bytes([MSG_IDENTIFY, 0x00, 0xFF]))

    def _send_frame(self, addr: int, cmd: int, payload: bytes = b''):
        raw = _encode(addr, cmd, payload)
        self._sock.sendall(bytes([MSG_FRAME, len(raw)]) + raw)

    def push_solid(self, r: int, g: int, b: int):
        self._send_frame(ADDR_BROADCAST, CMD_SET_COLOR, bytes([r, g, b]))
        self._send_frame(ADDR_BROADCAST, CMD_LATCH)

    def push_leds(self, leds: list[tuple[int, int, int]]):
        payload = bytes(v for rgb in leds for v in rgb)
        self._send_frame(ADDR_BROADCAST, CMD_SET_LEDS, payload)
        self._send_frame(ADDR_BROADCAST, CMD_LATCH)

    def close(self):
        self._sock.close()

# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------
# The square mirrors led_driver_native.cpp's physical layout.
# Clockwise LED order:
#   top row    left→right  leds[0..9]
#   right col  top→bottom  leds[10..19]
#   bottom row right→left  leds[20..29]
#   left col   bottom→top  leds[30..39]
#
# Grid: 12 rows × 21 cols
#   row 0:    ' ' [L0] ' ' [L1] ' ' … [L9] ' '
#   rows 1-10: [Lleft] <19 spaces> [Lright]
#   row 11:   ' ' [L29] ' ' [L28] ' ' … [L20] ' '

def _block(r: int, g: int, b: int) -> str:
    return f'\033[38;2;{r};{g};{b}m█\033[0m'

def draw_square(leds: list[tuple[int, int, int]]) -> None:
    """Clear the screen and draw the 12-row LED square."""
    out = ['\033[2J\033[H']   # clear + home cursor

    # Top row: leds[0..9]
    row = ' '
    for i in range(10):
        row += _block(*leds[i]) + ' '
    out.append(row)

    # Middle rows 1-10
    for r in range(1, 11):
        left  = _block(*leds[40 - r])   # leds[39] down to leds[30]
        right = _block(*leds[9 + r])    # leds[10] up to leds[19]
        out.append(left + ' ' * 19 + right)

    # Bottom row: leds[29..20]
    row = ' '
    for i in range(29, 19, -1):
        row += _block(*leds[i]) + ' '
    out.append(row)

    print('\n'.join(out), flush=True)

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def parse_color(s: str) -> tuple[int, int, int]:
    s = s.strip()
    if s.startswith('#'):
        h = s[1:]
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    r, g, b = s.split(',')
    return (int(r), int(g), int(b))

def solid(r: int, g: int, b: int) -> list[tuple[int, int, int]]:
    return [(r, g, b)] * NUM_LEDS

def hsv(h: float, s: float = 1.0, v: float = 1.0) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

# ---------------------------------------------------------------------------
# Animation frame generators
# ---------------------------------------------------------------------------
def frame_chase(tick: int, color: tuple = (255, 255, 255),
                trail: int = 4) -> list[tuple[int, int, int]]:
    """Single bright dot with a fading trail moving clockwise."""
    leds: list[tuple[int, int, int]] = [(0, 0, 0)] * NUM_LEDS
    for i in range(trail + 1):
        idx = (tick - i) % NUM_LEDS
        fade = 1.0 - i / (trail + 1)
        r, g, b = color
        leds[idx] = (int(r * fade), int(g * fade), int(b * fade))
    return leds

def frame_rainbow(tick: int) -> list[tuple[int, int, int]]:
    """Rotating rainbow gradient across all 40 positions."""
    return [hsv(((i + tick) % NUM_LEDS) / NUM_LEDS) for i in range(NUM_LEDS)]

def frame_pulse(tick: int,
                color: tuple = (0, 128, 255)) -> list[tuple[int, int, int]]:
    """All LEDs fade in and out together."""
    brightness = (math.sin(tick * math.pi / 20) + 1.0) / 2.0
    r, g, b = color
    c = (int(r * brightness), int(g * brightness), int(b * brightness))
    return [c] * NUM_LEDS

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def mode_color(rc: RC, color: tuple[int, int, int]) -> None:
    rc.push_solid(*color)
    time.sleep(0.1)
    draw_square(solid(*color))

def mode_stdin(rc: RC) -> None:
    lines = [ln for ln in sys.stdin.read().splitlines() if ln.strip()]
    if len(lines) != NUM_LEDS:
        sys.exit(f'ERROR: expected {NUM_LEDS} colors, got {len(lines)}')
    leds = [parse_color(ln) for ln in lines]
    rc.push_leds(leds)
    time.sleep(0.1)
    draw_square(leds)

def mode_test(rc: RC) -> None:
    FPS   = 25
    FRAME = 1.0 / FPS

    def tick(leds: list[tuple[int, int, int]]) -> None:
        t0 = time.monotonic()
        rc.push_leds(leds)
        draw_square(leds)
        elapsed = time.monotonic() - t0
        rem = FRAME - elapsed
        if rem > 0:
            time.sleep(rem)

    # Solid color cards, ~1 second each
    for rgb in [
        (255,   0,   0),   # red
        (  0, 255,   0),   # green
        (  0,   0, 255),   # blue
        (255, 128,   0),   # amber
        (255, 255, 255),   # white
        (  0,   0,   0),   # off
    ]:
        leds = solid(*rgb)
        for _ in range(FPS):
            tick(leds)

    # Chase: white dot with trail, 3 laps
    for t in range(NUM_LEDS * 3):
        tick(frame_chase(t))

    # Coloured chases: red, green, blue, one lap each
    for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        for t in range(NUM_LEDS):
            tick(frame_chase(t, color=color))

    # Rainbow sweep, 3 full rotations
    for t in range(NUM_LEDS * 3):
        tick(frame_rainbow(t))

    # Pulse (cyan), 3 full cycles (40 ticks per cycle)
    for t in range(40 * 3):
        tick(frame_pulse(t, color=(0, 200, 255)))

    # End: clear
    tick(solid(0, 0, 0))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description='Drive a single native tile for visual testing.')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--color', metavar='R,G,B|#RRGGBB',
                     help='solid color on all 40 LEDs')
    grp.add_argument('--stdin', action='store_true',
                     help='read 40 colors from stdin (one per line)')
    grp.add_argument('--test', action='store_true',
                     help='animated test pattern')
    args = ap.parse_args()

    for path, hint in [
        (BROKER, 'make -C tools'),
        (TILE,   'pio run -e native'),
    ]:
        if not os.path.isfile(path):
            sys.exit(f'ERROR: binary not found: {path}\n       hint: {hint}')

    procs: list[subprocess.Popen] = []
    try:
        procs.append(subprocess.Popen(
            [BROKER, str(ROW)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(0.15)

        procs.append(subprocess.Popen(
            [TILE, f'0x{TILE_ADDR:02X}', str(TILE_SLOT), str(ROW)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(0.3)

        rc = RC(SOCK)

        if args.color:
            mode_color(rc, parse_color(args.color))
        elif args.stdin:
            mode_stdin(rc)
        elif args.test:
            mode_test(rc)

        rc.close()

    except KeyboardInterrupt:
        pass
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
    main()
