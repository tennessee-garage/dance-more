from df2_pi.protocol.crc import crc16_ccitt


def test_check_value():
    # Standard CRC-16/CCITT-FALSE check value (poly 0x1021, init 0xFFFF)
    # for the ASCII string "123456789".
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_empty_input():
    assert crc16_ccitt(b"") == 0xFFFF
