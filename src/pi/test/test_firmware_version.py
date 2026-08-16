import pytest

from df2_pi.protocol.firmware_version import FirmwareVersion, format_version


def test_encode_wire_layout():
    v = FirmwareVersion(version=0x0102, git_sha=0x11223344, flags=0x01)
    assert v.encode() == bytes([0x01, 0x02, 0x11, 0x22, 0x33, 0x44, 0x01])


def test_decode_round_trip():
    v = FirmwareVersion(version=12, git_sha=0x2B5C293C, flags=0)
    assert FirmwareVersion.decode(v.encode()) == v


def test_decode_zero_entry():
    # An undiscovered/non-responsive tile slot is transmitted as all zeros
    # (docs/row-bus-protocol.md's VERSION_RESP) and must decode cleanly.
    assert FirmwareVersion.decode(bytes(7)) == FirmwareVersion(0, 0, 0)


def test_decode_rejects_reserved_flag_bits():
    data = bytes([0x00, 0x01, 0, 0, 0, 0, 0x02])  # bit 1 set
    with pytest.raises(ValueError):
        FirmwareVersion.decode(data)


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError):
        FirmwareVersion.decode(bytes(6))


def test_dirty_flag():
    assert not FirmwareVersion(1, 0, 0).dirty
    assert FirmwareVersion(1, 0, 0x01).dirty


def test_format_version_clean():
    v = FirmwareVersion(version=12, git_sha=0x2B5C293C, flags=0)
    assert format_version(v) == "v12+2b5c293c"


def test_format_version_dirty():
    v = FirmwareVersion(version=12, git_sha=0x2B5C293C, flags=0x01)
    assert format_version(v) == "v12+2b5c293c-dirty"
