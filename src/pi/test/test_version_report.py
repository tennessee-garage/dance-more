import pytest

from df2_pi.protocol.firmware_version import FirmwareVersion
from df2_pi.version_report import (
    NUM_TILE_SLOTS,
    VERSION_RESP_SIZE,
    RowVersionReport,
    assess_versions,
    format_version_report,
)

ROW_V = FirmwareVersion(version=12, git_sha=0x2B5C293C, flags=0)
TILE_V = FirmwareVersion(version=7, git_sha=0x2B5C293C, flags=0)


def _encode_resp(row: FirmwareVersion, tiles: list[FirmwareVersion | None]) -> bytes:
    assert len(tiles) == NUM_TILE_SLOTS
    tiles_valid = 0
    body = bytearray(row.encode())
    entries = bytearray()
    for slot, tile in enumerate(tiles):
        if tile is not None:
            tiles_valid |= 1 << slot
            entries += tile.encode()
        else:
            entries += bytes(7)
    return bytes(body) + bytes([tiles_valid]) + bytes(entries)


def test_decode_round_trip_all_tiles_present():
    tiles = [FirmwareVersion(version=7, git_sha=0x1000 + i, flags=0) for i in range(8)]
    payload = _encode_resp(ROW_V, tiles)
    assert len(payload) == VERSION_RESP_SIZE

    report = RowVersionReport.decode(payload)
    assert report.row == ROW_V
    assert report.tiles == tuple(tiles)


def test_decode_leaves_missing_slots_as_none():
    tiles = [TILE_V if slot != 3 else None for slot in range(8)]
    payload = _encode_resp(ROW_V, tiles)

    report = RowVersionReport.decode(payload)
    assert report.tiles[3] is None
    assert all(report.tiles[i] == TILE_V for i in range(8) if i != 3)


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError):
        RowVersionReport.decode(bytes(VERSION_RESP_SIZE - 1))


def _all_clean_report() -> RowVersionReport:
    return RowVersionReport(ROW_V, tuple(TILE_V for _ in range(8)))


def test_report_all_clean_and_consistent_is_ok():
    reports = {0: _all_clean_report(), 2: _all_clean_report()}
    text, ok = format_version_report(reports)
    assert ok
    assert "all v7+2b5c293c" in text
    assert "*" not in text


def test_report_flags_dirty_row():
    dirty_row = FirmwareVersion(ROW_V.version, ROW_V.git_sha, flags=0x01)
    reports = {0: _all_clean_report(), 2: RowVersionReport(dirty_row, _all_clean_report().tiles)}
    text, ok = format_version_report(reports)
    assert not ok
    assert "v12+2b5c293c-dirty *" in text


def test_report_flags_minority_row_version():
    behind_row = FirmwareVersion(version=11, git_sha=0xA783A93F, flags=0)
    reports = {
        0: _all_clean_report(),
        2: _all_clean_report(),
        4: RowVersionReport(behind_row, _all_clean_report().tiles),
    }
    text, ok = format_version_report(reports)
    assert not ok
    assert "v11+a783a93f *" in text


def test_report_flags_missing_tile_version():
    tiles = list(_all_clean_report().tiles)
    tiles[3] = None
    reports = {0: RowVersionReport(ROW_V, tuple(tiles))}
    text, ok = format_version_report(reports)
    assert not ok
    assert "slot 3: no version" in text


def test_slot_status_labels_a_missing_version_and_an_empty_slot_is_in_step():
    tiles = list(_all_clean_report().tiles)
    tiles[3] = tiles[4] = None
    reports = {0: RowVersionReport(ROW_V, tuple(tiles))}
    statuses = {0: (1, 1, 1, 0x00, 0x02, 1, 1, 1)}  # slot 3 empty, slot 4 not responding
    assessment = assess_versions(reports, statuses)
    by_slot = {t.slot: t for t in assessment.rows[0].tiles}
    assert (by_slot[3].state, by_slot[3].out_of_step) == ("empty", False)
    assert (by_slot[4].state, by_slot[4].out_of_step) == ("not responding", True)
    assert not assessment.ok
    statuses = {0: (1, 1, 1, 0x00, 0x00, 1, 1, 1)}  # both empty: nothing out of step
    assert assess_versions(reports, statuses).ok


def test_assessment_names_the_majorities():
    odd = FirmwareVersion(version=6, git_sha=0x2B5C293C, flags=0)
    reports = {0: _all_clean_report(), 1: _all_clean_report(), 2: RowVersionReport(odd, _all_clean_report().tiles)}
    assessment = assess_versions(reports)
    assert (assessment.row_majority, assessment.tile_majority) == (ROW_V, TILE_V)
    assert [r.out_of_step for r in assessment.rows] == [False, False, True]


def test_report_flags_minority_tile_version():
    tiles = list(_all_clean_report().tiles)
    tiles[5] = FirmwareVersion(version=6, git_sha=0x11111111, flags=0)
    reports = {0: _all_clean_report(), 2: RowVersionReport(ROW_V, tuple(tiles))}
    text, ok = format_version_report(reports)
    assert not ok
    assert "slot 5: v6+11111111 *" in text


def test_report_flags_non_responding_row():
    reports = {0: _all_clean_report(), 2: None}
    text, ok = format_version_report(reports)
    assert not ok
    assert "row 2  NOT RESPONDING" in text


def test_report_single_row_is_trivially_its_own_majority():
    reports = {0: _all_clean_report()}
    _, ok = format_version_report(reports)
    assert ok
