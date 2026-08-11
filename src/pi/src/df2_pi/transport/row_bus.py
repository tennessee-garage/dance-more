"""Serial transport for the Row Bus (Raspberry Pi <-> row controllers).

Talks through the Pi hat's RS-485 transceiver over the Pi's UART. See
docs/row-bus-protocol.md for the wire format and docs/communication.md for
the physical layer this rides on.
"""

from __future__ import annotations

import time

import serial

from ..protocol.constants import ADDR_BROADCAST
from ..protocol.frame import Frame, FrameParser

DEFAULT_PORT = "/dev/serial0"
DEFAULT_BAUDRATE = 4_000_000  # 4 Mbps minimum per row-bus-protocol.md; 8 Mbps recommended


class RowBus:
    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        read_timeout: float = 0.05,
    ) -> None:
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=read_timeout)
        self._parser = FrameParser()

    def close(self) -> None:
        self._serial.close()

    def __enter__(self) -> RowBus:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def send(self, addr: int, cmd: int, payload: bytes = b"") -> None:
        self._serial.write(Frame(addr, cmd, payload).encode())

    def broadcast(self, cmd: int, payload: bytes = b"") -> None:
        self.send(ADDR_BROADCAST, cmd, payload)

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
