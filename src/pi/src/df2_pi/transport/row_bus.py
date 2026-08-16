"""Serial transport for the Row Bus (Raspberry Pi <-> row controllers).

Talks through the Pi hat's RS-485 transceiver over the Pi's UART (GPIO14
TXD0 / GPIO15 RXD0), with a GPIO output driving the transceiver's DE/RE
direction pin - LOW = RX (default, listening), HIGH = TX - since nothing in
the kernel UART driver does this for us. See docs/row-bus-protocol.md for
the wire format and docs/communication.md for the physical layer this
rides on.

Deliberately NOT /dev/serial0: on this Pi 5, that symlink resolves to
ttyAMA10, the BCM2712 SoC's own internal "console" UART (device tree alias
`console`/`uart10`), which is electrically disconnected from the 40-pin
header entirely. The header UART (GPIO14/15, RP1-attached) is a separate
peripheral, aliased `serial0`/`uart0` in the device tree but exposed at
/dev/ttyAMA0 - confirmed via `cat /proc/device-tree/aliases/serial0` and
cross-checking each ttyAMA's sysfs of_node. Getting this wrong doesn't
error - both ports open and accept writes fine - it just silently talks to
the wrong hardware.
"""

from __future__ import annotations

import time

import serial
from gpiozero import OutputDevice

from ..protocol.constants import ADDR_BROADCAST
from ..protocol.frame import Frame, FrameParser

DEFAULT_PORT = "/dev/ttyAMA0"
# The Pi's RP1 UART is clocked at 50 MHz and a PL011 needs a divisor >= 1, so
# 50e6/16 = 3.125 Mbps is this link's hard ceiling (docs/row-bus-protocol.md
# §1). It divides exactly on both ends. Requesting more doesn't raise - the
# kernel silently clamps and the row controller then sees only framing
# errors, which its UART driver discards uncounted, so the bus goes mute.
DEFAULT_BAUDRATE = 3_125_000
DEFAULT_XDIR_PIN = 23  # BCM numbering; matches the pi-hat's XDIR wiring

# Extra time to hold XDIR after the frame's computed on-wire duration, so a
# late start inside the kernel can't truncate the last stop bit. Must stay
# well under the row controller's turnaround guard (TURNAROUND_GUARD_US in
# pi_transport_rp2350.cpp) or we'll still be driving when it replies.
DEFAULT_TX_RELEASE_MARGIN_S = 40e-6

# 8N1: one start bit + 8 data + one stop = 10 bit-times per byte.
_BITS_PER_BYTE = 10


class RowBus:
    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        read_timeout: float = 0.05,
        xdir_pin: int | None = DEFAULT_XDIR_PIN,
        tx_release_margin_s: float = DEFAULT_TX_RELEASE_MARGIN_S,
    ) -> None:
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=read_timeout)
        self._parser = FrameParser()
        self._tx_release_margin_s = tx_release_margin_s
        # xdir_pin=None disables direction control entirely, e.g. for a
        # transport that turns out to be full-duplex or auto-direction.
        self._xdir = OutputDevice(xdir_pin, initial_value=False) if xdir_pin is not None else None

    def close(self) -> None:
        self._serial.close()
        if self._xdir is not None:
            self._xdir.close()

    def __enter__(self) -> RowBus:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def send(self, addr: int, cmd: int, payload: bytes = b"") -> None:
        self._write(Frame(addr, cmd, payload).encode())

    def broadcast(self, cmd: int, payload: bytes = b"") -> None:
        self.send(ADDR_BROADCAST, cmd, payload)

    def send_raw(self, data: bytes) -> None:
        """Write pre-built bytes directly, bypassing Frame.encode().

        For protocol-robustness testing (e.g. sending a deliberately
        corrupt frame) where the caller needs to put specific bytes on the
        wire rather than a well-formed Frame.
        """
        self._write(data)

    def start_write(self, data: bytes) -> float:
        """Key up and write, returning the deadline at which the frame has
        cleared the wire. The caller must busy-wait to that deadline and
        then call finish_write().

        Split out of _write() so a caller driving several chains can start
        them all before waiting on any, which keeps a broadcast's skew
        across chains down to the gap between write() calls (microseconds)
        instead of one full frame time per chain. See Floor.latch().
        """
        if self._xdir is None:
            self._serial.write(data)
            self._serial.flush()
            return time.perf_counter()

        # Deliberately NOT serial.flush() (tcdrain) to decide when the frame
        # has cleared the wire: on the Pi 5's PL011 tcdrain overshoots the
        # true transmit time by ~6 ms regardless of frame size. That's ~60x
        # the row controller's 100 us turnaround guard, so the Pi would
        # still be driving the bus - with its own receiver disabled, since
        # DE and ~RE share XDIR - when the reply arrived. Every response was
        # lost to that collision.
        #
        # Instead, hold XDIR for the frame's computed on-wire duration plus
        # a small margin. Busy-wait rather than time.sleep(), whose ~50 us+
        # granularity is the same order as the window we're trying to hit.
        tx_seconds = len(data) * _BITS_PER_BYTE / self._serial.baudrate

        self._xdir.on()  # switch transceiver to TX
        start = time.perf_counter()
        self._serial.write(data)
        return start + tx_seconds + self._tx_release_margin_s

    def finish_write(self) -> None:
        """Release the bus after start_write()'s deadline has passed."""
        if self._xdir is not None:
            self._xdir.off()  # back to RX

    def _write(self, data: bytes) -> None:
        deadline = self.start_write(data)
        while time.perf_counter() < deadline:
            pass
        self.finish_write()

    def read_frame(self, timeout: float | None = None) -> Frame | None:
        """Block until a CRC-valid frame arrives, or timeout elapses."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            byte = self._serial.read(1)
            if not byte:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                continue
            frame = self._parser.feed(byte[0])
            if frame is not None:
                return frame
            if deadline is not None and time.monotonic() >= deadline:
                return None
