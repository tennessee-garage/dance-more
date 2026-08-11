from df2_pi.protocol.constants import Cmd
from df2_pi.protocol.frame import Frame, FrameParser


def test_round_trip_no_payload():
    frame = Frame(addr=0x03, cmd=Cmd.STATUS)
    parser = FrameParser()
    decoded = list(parser.feed_bytes(frame.encode()))
    assert decoded == [frame]


def test_round_trip_with_payload():
    frame = Frame(addr=0xFF, cmd=Cmd.SEND_DATA, payload=bytes(range(32)))
    parser = FrameParser()
    decoded = list(parser.feed_bytes(frame.encode()))
    assert decoded == [frame]


def test_corrupt_crc_is_dropped_and_resyncs():
    good = Frame(addr=0x01, cmd=Cmd.STATUS).encode()
    corrupt = bytearray(good)
    corrupt[-1] ^= 0xFF  # flip CRC_L

    parser = FrameParser()
    decoded = list(parser.feed_bytes(bytes(corrupt) + good))
    assert decoded == [Frame(addr=0x01, cmd=Cmd.STATUS)]


def test_junk_before_sync_is_ignored():
    frame = Frame(addr=0x02, cmd=Cmd.POWER)
    parser = FrameParser()
    decoded = list(parser.feed_bytes(b"\x00\x11\xaa" + frame.encode()))
    assert decoded == [frame]
