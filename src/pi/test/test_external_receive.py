import socket
import struct
import time

import numpy as np
import pytest

from df2_pi.interfacing.packets import (
    OP_POLL_REPLY,
    Dmx,
    Poll,
    Sync,
    artdmx,
    artpoll,
    artsync,
    parse_artnet,
    parse_sacn,
    poll_replies,
    sacn_data,
    sacn_group,
    sacn_sync,
)
from df2_pi.interfacing.receiver import SYNC_TIMEOUT_S, Assembler, Layout, Receiver

# ---- Art-Net packets -------------------------------------------------------------------------


def test_artdmx_round_trips_with_a_15_bit_port_address():
    data = bytes(range(200))
    parsed = parse_artnet(artdmx(0x1234, data))
    assert parsed == Dmx(universe=0x1234, data=data)


def test_artdmx_fields_are_where_the_spec_puts_them():
    packet = artdmx(0x0102, b"\x07\x08", sequence=9)
    assert packet[:8] == b"Art-Net\x00"
    assert struct.unpack_from("<H", packet, 8)[0] == 0x5000
    assert packet[12] == 9 and packet[14] == 0x02 and packet[15] == 0x01  # sequence, SubUni, Net
    assert struct.unpack_from(">H", packet, 16)[0] == 2 and packet[18:] == b"\x07\x08"


def test_sync_poll_and_junk():
    assert parse_artnet(artsync()) == Sync()
    assert parse_artnet(artpoll()) == Poll()
    assert parse_artnet(b"Art-Net\x00\x00\x99" + b"\x00" * 10) is None  # an opcode we ignore
    assert parse_artnet(b"not art-net at all") is None
    truncated = artdmx(0, bytes(100))[:-10]
    assert parse_artnet(truncated) is None


def test_poll_replies_group_universes_by_subnet_four_ports_each():
    replies = poll_replies("10.0.0.5", list(range(14, 24)))
    # 14,15 | 16..19 | 20..23: split at the sub-net boundary (16), then by four
    assert len(replies) == 3
    for index, reply in enumerate(replies, start=1):
        assert len(reply) == 239
        assert struct.unpack_from("<H", reply, 8)[0] == OP_POLL_REPLY
        assert reply[10:14] == bytes((10, 0, 0, 5)) and reply[211] == index
    first, second = replies[0], replies[1]
    assert struct.unpack_from(">H", first, 172)[0] == 2 and first[19] == 0 and first[190:192] == bytes((14, 15))
    assert struct.unpack_from(">H", second, 172)[0] == 4 and second[19] == 1 and second[190:194] == bytes((0, 1, 2, 3))
    assert first[26:37] == b"Dance Floor"


# ---- sACN packets -----------------------------------------------------------------------------


def test_sacn_data_and_sync_round_trip():
    data = bytes(range(255)) * 2
    assert parse_sacn(sacn_data(7, data, sync=3)) == Dmx(universe=7, data=data, sync=3)
    assert parse_sacn(sacn_sync(3)) == Sync(address=3)


def test_sacn_preview_terminated_and_non_dmx_are_ignored():
    assert parse_sacn(sacn_data(1, b"\x01\x02", options=0x80)) is None
    assert parse_sacn(sacn_data(1, b"\x01\x02", options=0x40)) is None
    packet = bytearray(sacn_data(1, b"\x01\x02"))
    packet[125] = 0xDD  # a non-zero start code (RDM, text, ...)
    assert parse_sacn(bytes(packet)) is None
    assert parse_sacn(artdmx(1, b"\x00\x00")) is None


def test_sacn_multicast_group():
    assert sacn_group(1) == "239.255.0.1" and sacn_group(0x0203) == "239.255.2.3"


# ---- assembling frames ------------------------------------------------------------------------


def universes(layout: Layout, first: int = 0) -> list[Dmx]:
    return [Dmx(first + u, bytes([u + 1]) * 512) for u in range(layout.universes)]


def test_layout_universe_counts():
    assert Layout("tile").universes == 1
    assert Layout("grid", 34, 34).universes == 7
    assert Layout("raw").universes == 23


def test_a_frame_is_complete_when_every_universe_has_arrived():
    layout = Layout("raw")
    a = Assembler("artnet", 10, layout)
    packets = universes(layout, 10)
    for packet in packets[:-1]:
        assert a.dmx(packet, "h", 0.0) is None
    frame = a.dmx(packets[-1], "h", 0.0)
    assert frame.shape == (23, 512) and frame[22, 0] == 23 and frame[0, 0] == 1
    assert a.dmx(Dmx(99, b"\x00\x00"), "h", 0.0) is None  # not ours


def test_a_repeated_universe_publishes_what_there_is():
    layout = Layout("grid", 34, 34)  # 7 universes
    a = Assembler("artnet", 0, layout)
    packets = universes(layout)
    for packet in packets[:5]:
        a.dmx(packet, "h", 0.0)
    frame = a.dmx(packets[0], "h", 0.0)  # the sender started over without universes 5 and 6
    assert frame is not None and frame[4, 0] == 5 and frame[5, 0] == 0


def test_with_sync_only_the_sync_completes_a_frame_until_it_times_out():
    layout = Layout("grid", 34, 34)
    a = Assembler("artnet", 0, layout)
    a.sync(0.0)  # the sender uses sync
    for packet in universes(layout):
        assert a.dmx(packet, "h", 0.1) is None
    assert a.sync(0.2)[6, 0] == 7
    later = SYNC_TIMEOUT_S + 1.0  # no sync for a while: back to completion
    packets = universes(layout)
    for packet in packets[:-1]:
        assert a.dmx(packet, "h", later) is None
    assert a.dmx(packets[-1], "h", later) is not None


# ---- the receiver, over loopback ----------------------------------------------------------------


@pytest.fixture
def receiver():
    r = Receiver(Layout("tile"), artnet_port=0, sacn_port=0, bind="127.0.0.1")
    r.start()
    yield r
    r.stop()


def send(port: int, packet: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(packet, ("127.0.0.1", port))


def wait_for(predicate, timeout: float = 2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("timed out")


def test_artnet_and_sacn_frames_are_received(receiver):
    ports = receiver.ports()
    send(ports["artnet"], artdmx(0, bytes([200]) * 192))
    frame = wait_for(receiver.latest)
    assert (frame.protocol, frame.sender, frame.layout.mode) == ("artnet", "127.0.0.1", "tile")
    assert frame.channels[0, 0] == 200
    send(ports["sacn"], sacn_data(1, bytes([50]) * 192))
    frame = wait_for(lambda: receiver.latest() if receiver.latest().protocol == "sacn" else None)
    assert frame.channels[0, 0] == 50 and frame.seq == 2


def test_artpoll_gets_a_reply(receiver):
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(2.0)
    receiver._reply_port = listener.getsockname()[1]
    try:
        listener.sendto(artpoll(), ("127.0.0.1", receiver.ports()["artnet"]))
        reply, _ = listener.recvfrom(1024)
    finally:
        listener.close()
    assert len(reply) == 239 and struct.unpack_from("<H", reply, 8)[0] == OP_POLL_REPLY
    assert b"tile mode, 1 universe" in reply[108:172]


def test_reconfiguring_drops_the_old_frame_and_closing_a_protocol_closes_its_socket(receiver):
    send(receiver.ports()["artnet"], artdmx(0, bytes([9]) * 192))
    wait_for(receiver.latest)
    receiver.configure(layout=Layout("raw"), sacn=False)
    assert receiver.latest() is None and receiver.layout.mode == "raw"
    assert set(receiver.ports()) == {"artnet"}
