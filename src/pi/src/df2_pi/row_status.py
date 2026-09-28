"""A row controller's STATUS and POWER replies, decoded
(docs/row-bus-protocol.md).

Shared by `df2-pi status` and the admin page's floor status.
"""

from __future__ import annotations

from dataclasses import dataclass

STATE_NAMES = {0x00: "idle", 0x01: "discovering", 0x02: "running", 0x03: "error"}
TILE_SLOTS = 8


@dataclass(frozen=True)
class RowStatus:
    state: int
    tiles_found: int
    tile_status: tuple[int, ...]  # one byte per slot
    uptime_s: int | None  # None: firmware older than the uptime field

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.state, hex(self.state))

    @classmethod
    def decode(cls, payload: bytes) -> RowStatus:
        if len(payload) < 2 + TILE_SLOTS:
            raise ValueError(f"STATUS_RESP payload too short: {len(payload)} bytes")
        uptime = None
        # Uptime follows the tile-status bytes; a shorter payload is
        # firmware that predates the field, not a malformed reply.
        if len(payload) >= 2 + TILE_SLOTS + 4:
            p = payload[2 + TILE_SLOTS :]
            uptime = (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]
        return cls(payload[0], payload[1], tuple(payload[2 : 2 + TILE_SLOTS]), uptime)

    def format_uptime(self) -> str:
        if self.uptime_s is None:
            return ""
        h, rem = divmod(self.uptime_s, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"


@dataclass(frozen=True)
class RowPower:
    """A POWER reply: the row's 12 V rail as its power monitor measures it."""

    voltage_mV: int
    current_mA: int
    power_mW: int

    @classmethod
    def decode(cls, payload: bytes) -> RowPower:
        if len(payload) < 6:
            raise ValueError(f"POWER_RESP payload too short: {len(payload)} bytes")
        p = payload
        return cls((p[0] << 8) | p[1], (p[2] << 8) | p[3], (p[4] << 8) | p[5])
