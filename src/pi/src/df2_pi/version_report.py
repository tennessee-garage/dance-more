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


@dataclass(frozen=True)
class TileVersion:
    slot: int
    version: FirmwareVersion | None  # None: the row has no version cached for it
    out_of_step: bool


@dataclass(frozen=True)
class RowVersion:
    row: int
    version: FirmwareVersion | None  # None: the row did not respond
    out_of_step: bool
    tiles: tuple[TileVersion, ...]


@dataclass(frozen=True)
class VersionAssessment:
    rows: tuple[RowVersion, ...]
    ok: bool


def assess_versions(row_reports: dict[int, RowVersionReport | None]) -> VersionAssessment:
    """Flag every row and tile that is out of step: not responding, built
    dirty, missing a version, or differing from what most of the floor
    runs. There is no notion of an "expected" version, only "is everything
    in step"."""
    row_majority = _majority(r.row for r in row_reports.values() if r is not None)
    tile_majority = _majority(
        t for r in row_reports.values() if r is not None for t in r.tiles if t is not None
    )
    rows = []
    for row in sorted(row_reports):
        report = row_reports[row]
        if report is None:
            rows.append(RowVersion(row, None, True, ()))
            continue
        tiles = tuple(
            TileVersion(slot, tile, tile is None or tile.dirty or tile != tile_majority)
            for slot, tile in enumerate(report.tiles)
        )
        rows.append(RowVersion(row, report.row, report.row.dirty or report.row != row_majority, tiles))
    ok = all(not r.out_of_step and not any(t.out_of_step for t in r.tiles) for r in rows)
    return VersionAssessment(tuple(rows), ok)


def format_version_report(row_reports: dict[int, RowVersionReport | None]) -> tuple[str, bool]:
    """Render `assess_versions` as a table, and whether the floor is consistent."""
    assessment = assess_versions(row_reports)
    lines = []
    for row in assessment.rows:
        if row.version is None:
            lines.append(f"row {row.row}  NOT RESPONDING")
            continue
        row_text = format_version(row.version) + (" *" if row.out_of_step else "")
        tile_lines = [
            f"slot {t.slot}: no version" if t.version is None else f"slot {t.slot}: {format_version(t.version)} *"
            for t in row.tiles
            if t.out_of_step
        ]
        present = [t.version for t in row.tiles if t.version is not None]
        if tile_lines:
            tile_text = "; ".join(tile_lines)
        elif present:
            tile_text = f"all {format_version(present[0])}"
        else:
            tile_text = "(no tiles)"
        lines.append(f"row {row.row}  {row_text}  tiles {len(present)}/{NUM_TILE_SLOTS}  {tile_text}")

    if not assessment.ok:
        lines.append("* mismatch, dirty build, missing version, or non-responding")
    return "\n".join(lines), assessment.ok
