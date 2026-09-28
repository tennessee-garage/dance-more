import pytest

from df2_pi.row_status import RowPower, RowStatus


def test_status_decodes_state_tiles_slots_and_uptime():
    s = RowStatus.decode(bytes([0x02, 7]) + bytes([1] * 7 + [0]) + (3725).to_bytes(4, "big"))
    assert (s.state_name, s.tiles_found, s.tile_status, s.uptime_s) == ("running", 7, (1,) * 7 + (0,), 3725)
    assert s.format_uptime() == "1h02m05s"


def test_status_from_firmware_without_uptime():
    s = RowStatus.decode(bytes([0x00, 0]) + bytes(8))
    assert (s.state_name, s.uptime_s, s.format_uptime()) == ("idle", None, "")


def test_power_decodes_millivolts_milliamps_milliwatts():
    p = RowPower.decode((12034).to_bytes(2, "big") + (1480).to_bytes(2, "big") + (17810).to_bytes(2, "big"))
    assert (p.voltage_mV, p.current_mA, p.power_mW) == (12034, 1480, 17810)


@pytest.mark.parametrize("decode, payload", [(RowStatus.decode, bytes(9)), (RowPower.decode, bytes(5))])
def test_a_short_reply_is_refused(decode, payload):
    with pytest.raises(ValueError, match="too short"):
        decode(payload)
