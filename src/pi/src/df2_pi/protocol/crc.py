"""CRC-16/CCITT-FALSE, matching the row/tile firmware's checksum.

Polynomial 0x1021, init 0xFFFF, no reflect, no xorout - see
docs/row-bus-protocol.md and src/common/tile_bus_protocol/protocol.cpp.

Computed by `binascii.crc_hqx`, which is this exact CRC in C. It matters:
the bit-at-a-time Python loop it replaced cost 2.5 ms per 1448-byte
SEND_DATA payload on the Pi 5 - 20 ms of a 33 ms frame budget just to
checksum eight rows (docs/measurements/2026-09-15-eight-row-bus-bringup.md).
"""

from binascii import crc_hqx


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    return crc_hqx(data, crc)
