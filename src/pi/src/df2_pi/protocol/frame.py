"""Row Bus frame encoding and streaming decode.

Frame layout (docs/row-bus-protocol.md #2):

    SYNC1 SYNC2 ADDR CMD LEN_H LEN_L PAYLOAD[N] CRC_H CRC_L
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto

from .constants import MAX_PAYLOAD, SYNC1, SYNC2
from .crc import crc16_ccitt


@dataclass(frozen=True)
class Frame:
    addr: int
    cmd: int
    payload: bytes = b""

    def encode(self) -> bytes:
        if not (0 <= self.addr <= 0xFF):
            raise ValueError(f"addr out of range: {self.addr}")
        if not (0 <= self.cmd <= 0xFF):
            raise ValueError(f"cmd out of range: {self.cmd}")
        if len(self.payload) > MAX_PAYLOAD:
            raise ValueError(f"payload too large: {len(self.payload)} > {MAX_PAYLOAD}")

        body = bytes(
            [self.addr, self.cmd, len(self.payload) >> 8, len(self.payload) & 0xFF]
        ) + self.payload
        crc = crc16_ccitt(body)
        return bytes([SYNC1, SYNC2]) + body + bytes([crc >> 8, crc & 0xFF])


class _State(IntEnum):
    SYNC1 = auto()
    SYNC2 = auto()
    ADDR = auto()
    CMD = auto()
    LEN_H = auto()
    LEN_L = auto()
    PAYLOAD = auto()
    CRC_H = auto()
    CRC_L = auto()


class FrameParser:
    """Streaming Row Bus decoder: feed it bytes as they arrive off the wire.

    Returns a decoded Frame once a full, CRC-valid frame has been
    assembled. Frames that fail CRC are silently dropped and the parser
    resyncs on the next SYNC1/SYNC2 pair, matching the firmware's receiver
    framing rules (docs/row-bus-protocol.md "Receiver framing").
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state = _State.SYNC1
        self._addr = 0
        self._cmd = 0
        self._len = 0
        self._payload = bytearray()
        self._crc = 0

    def feed(self, byte: int) -> Frame | None:
        state = self._state

        if state == _State.SYNC1:
            self._state = _State.SYNC2 if byte == SYNC1 else _State.SYNC1
        elif state == _State.SYNC2:
            if byte == SYNC2:
                self._state = _State.ADDR
            elif byte != SYNC1:
                self._state = _State.SYNC1
        elif state == _State.ADDR:
            self._addr = byte
            self._state = _State.CMD
        elif state == _State.CMD:
            self._cmd = byte
            self._state = _State.LEN_H
        elif state == _State.LEN_H:
            self._len = byte << 8
            self._state = _State.LEN_L
        elif state == _State.LEN_L:
            self._len |= byte
            self._payload = bytearray()
            self._state = _State.PAYLOAD if self._len else _State.CRC_H
        elif state == _State.PAYLOAD:
            self._payload.append(byte)
            if len(self._payload) == self._len:
                self._state = _State.CRC_H
        elif state == _State.CRC_H:
            self._crc = byte << 8
            self._state = _State.CRC_L
        elif state == _State.CRC_L:
            self._crc |= byte
            frame = self._finish()
            self.reset()
            return frame

        return None

    def feed_bytes(self, data: bytes):
        for byte in data:
            frame = self.feed(byte)
            if frame is not None:
                yield frame

    def _finish(self) -> Frame | None:
        body = bytes(
            [self._addr, self._cmd, self._len >> 8, self._len & 0xFF]
        ) + bytes(self._payload)
        if crc16_ccitt(body) != self._crc:
            return None
        return Frame(self._addr, self._cmd, bytes(self._payload))
