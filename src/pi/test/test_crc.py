from df2_pi.protocol.crc import crc16_ccitt


def test_check_value():
    # Standard CRC-16/CCITT-FALSE check value (poly 0x1021, init 0xFFFF)
    # for the ASCII string "123456789".
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_empty_input():
    assert crc16_ccitt(b"") == 0xFFFF


def _reference(data: bytes, crc: int = 0xFFFF) -> int:
    """Bit-at-a-time CRC-16/CCITT-FALSE, the definition the fast path must match."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def test_matches_reference_on_full_size_payloads():
    import random

    rng = random.Random(0x1021)
    for _ in range(20):
        data = bytes(rng.getrandbits(8) for _ in range(1456))
        assert crc16_ccitt(data) == _reference(data)


def test_seed_is_honoured():
    assert crc16_ccitt(b"abc", 0x0000) == _reference(b"abc", 0x0000)
    assert crc16_ccitt(b"abc", 0x1234) == _reference(b"abc", 0x1234)
