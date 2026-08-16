"""Firmware version identity, mirroring src/common/tile_bus_protocol/firmware_version.h.

Wire encoding (docs/row-bus-protocol.md and docs/tile-bus-protocol.md's
VERSION_RESP): 7 bytes big-endian - version_H version_L sha[0..3] flags.
"""

from __future__ import annotations

from dataclasses import dataclass

WIRE_SIZE = 7
FLAG_DIRTY = 0x01


@dataclass(frozen=True)
class FirmwareVersion:
    version: int
    git_sha: int
    flags: int

    @property
    def dirty(self) -> bool:
        return bool(self.flags & FLAG_DIRTY)

    def encode(self) -> bytes:
        if not (0 <= self.version <= 0xFFFF):
            raise ValueError(f"version out of range: {self.version}")
        if not (0 <= self.git_sha <= 0xFFFFFFFF):
            raise ValueError(f"git_sha out of range: {self.git_sha}")
        if not (0 <= self.flags <= 0xFF):
            raise ValueError(f"flags out of range: {self.flags}")
        return bytes(
            [
                self.version >> 8,
                self.version & 0xFF,
                (self.git_sha >> 24) & 0xFF,
                (self.git_sha >> 16) & 0xFF,
                (self.git_sha >> 8) & 0xFF,
                self.git_sha & 0xFF,
                self.flags,
            ]
        )

    @classmethod
    def decode(cls, data: bytes) -> FirmwareVersion:
        if len(data) != WIRE_SIZE:
            raise ValueError(f"expected {WIRE_SIZE} bytes, got {len(data)}")
        flags = data[6]
        if flags & ~FLAG_DIRTY:
            raise ValueError(f"reserved flag bits set: {flags:#x}")
        version = (data[0] << 8) | data[1]
        git_sha = (data[2] << 24) | (data[3] << 16) | (data[4] << 8) | data[5]
        return cls(version, git_sha, flags)


def format_version(v: FirmwareVersion) -> str:
    """Render as `v12+2b5c293c`, or `v12+2b5c293c-dirty` when built dirty."""
    text = f"v{v.version}+{v.git_sha:08x}"
    return f"{text}-dirty" if v.dirty else text
