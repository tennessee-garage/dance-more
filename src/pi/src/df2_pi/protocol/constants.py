"""Row Bus wire-format constants.

Mirrors docs/row-bus-protocol.md. Row Bus is a 2-byte-LEN variant of the
same frame shape used on Tile Bus (see src/common/tile_bus_protocol/).
"""

from enum import IntEnum

SYNC1 = 0xAA
SYNC2 = 0x55

ADDR_BROADCAST = 0xFF
MIN_ROW_ADDR = 0x00
MAX_ROW_ADDR = 0x07

MAX_PAYLOAD = 968
FRAME_OVERHEAD = 8  # SYNC1 SYNC2 ADDR CMD LEN_H LEN_L .. CRC_H CRC_L
MAX_FRAME_SIZE = FRAME_OVERHEAD + MAX_PAYLOAD

RESPONSE_BIT = 0x80


class Cmd(IntEnum):
    """Pi -> row controller commands."""

    TEST = 0x01
    STATUS = 0x02
    POWER = 0x03
    RE_DISCOVER = 0x04
    ERROR_LOG = 0x05
    SEND_DATA = 0x10
    LATCH = 0x11
    BLACKOUT = 0x12


class Resp(IntEnum):
    """Row controller -> Pi response codes."""

    TEST_RESP = 0x81
    STATUS_RESP = 0x82
    POWER_RESP = 0x83
    RE_DISCOVER_RESP = 0x84
    ERROR_LOG_RESP = 0x85


class TileCmd(IntEnum):
    """Tile Bus command codes, as embedded in a SEND_DATA per-tile entry."""

    SET_COLOR = 0x10
    SET_PATTERN = 0x11
    SET_LEDS = 0x12


# Per-tile SEND_DATA entry size in bytes, including the 1-byte tile_cmd header.
TILE_ENTRY_SIZE = {
    TileCmd.SET_COLOR: 4,
    TileCmd.SET_PATTERN: 6,
    TileCmd.SET_LEDS: 121,
}
