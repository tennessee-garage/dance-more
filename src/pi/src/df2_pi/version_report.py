"""Aggregates VERSION responses from every row into a floor-wide report.

Pure logic - no Floor/serial access here, so the majority/mismatch rules
are testable without mocking hardware. See cli.py's `version` subcommand
for how a report gets built from a live floor.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .protocol.firmware_version import WIRE_SIZE, FirmwareVersion, format_version

NUM_TILE_SLOTS = 8
# row FirmwareVersion + tiles_valid bitmask + 8 tile FirmwareVersion entries
# (docs/row-bus-protocol.md's VERSION_RESP).
VERSION_RESP_SIZE = WIRE_SIZE + 1 + NUM_TILE_SLOTS * WIRE_SIZE  # 64


@dataclass(frozen=True)
class RowVersionReport:
    row: FirmwareVersion
    tiles: tuple[FirmwareVersion | None, ...]  # len NUM_TILE_SLOTS; None = cache miss

    @classmethod
    def decode(cls, payload: bytes) -> RowVersionReport:
        if len(payload) != VERSION_RESP_SIZE:
            raise ValueError(
                f"expected {VERSION_RESP_SIZE}-byte VERSION_RESP payload, got {len(payload)}"
            )
        row = FirmwareVersion.decode(payload[:WIRE_SIZE])
        tiles_valid = payload[WIRE_SIZE]
        tiles = []
        for slot in range(NUM_TILE_SLOTS):
            offset = WIRE_SIZE + 1 + slot * WIRE_SIZE
            if tiles_valid & (1 << slot):
                tiles.append(FirmwareVersion.decode(payload[offset : offset + WIRE_SIZE]))
            else:
                tiles.append(None)
        return cls(row, tuple(tiles))


def _majority(values) -> FirmwareVersion | None:
    counts = Counter(values)
    return counts.most_common(1)[0][0] if counts else None


def format_version_report(row_reports: dict[int, RowVersionReport | None]) -> tuple[str, bool]:
    """Render a table and report whether the floor is consistent.

    A row or tile is flagged if it didn't respond, was built dirty, or its
    version differs from whatever most of the floor is running - there's no
    notion of an "expected" version, only "is everything in step".
    """
    row_majority = _majority(r.row for r in row_reports.values() if r is not None)
    tile_majority = _majority(
        t for r in row_reports.values() if r is not None for t in r.tiles if t is not None
    )

    lines = []
    ok = True
    for row in sorted(row_reports):
        report = row_reports[row]
        if report is None:
            lines.append(f"row {row}  NOT RESPONDING")
            ok = False
            continue

        row_bad = report.row.dirty or report.row != row_majority
        ok = ok and not row_bad
        row_text = format_version(report.row) + (" *" if row_bad else "")

        tile_lines = []
        for slot, tile in enumerate(report.tiles):
            if tile is None:
                tile_lines.append(f"slot {slot}: no version")
                ok = False
            elif tile.dirty or tile != tile_majority:
                tile_lines.append(f"slot {slot}: {format_version(tile)} *")
                ok = False

        present = [t for t in report.tiles if t is not None]
        if tile_lines:
            tile_text = "; ".join(tile_lines)
        elif present:
            tile_text = f"all {format_version(present[0])}"
        else:
            tile_text = "(no tiles)"

        lines.append(
            f"row {row}  {row_text}  tiles {len(present)}/{NUM_TILE_SLOTS}  {tile_text}"
        )

    if not ok:
        lines.append("* mismatch, dirty build, missing version, or non-responding")
    return "\n".join(lines), ok
