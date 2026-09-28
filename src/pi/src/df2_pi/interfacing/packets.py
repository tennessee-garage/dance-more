"""Art-Net 4 and sACN (E1.31) packets: parsing what a media server sends,
and building the one reply the floor owes.

Pure functions over bytes - no sockets here (receiver.py has those) - so
every packet shape is unit-testable byte for byte.

Art-Net (UDP 6454). Every packet starts "Art-Net\\0" and a little-endian
opcode. The floor handles three:

    ArtDmx   0x5000  one universe of DMX data (up to 512 channels)
    ArtSync  0x5200  "every universe of this frame has been sent"
    ArtPoll  0x2000  discovery: answered with an ArtPollReply (0x2100) per
                     group of up to 4 universes, which is how the floor
                     appears in Resolume's and TouchDesigner's node lists

An Art-Net universe is a 15-bit port-address: Net (7 bits), Sub-Net (4),
Universe (4). ArtDmx carries it as SubUni (low byte) and Net (high byte).

sACN / E1.31 (UDP 5568, multicast 239.255.<hi>.<lo> per universe). Root
layer, framing layer, DMP layer; the floor reads data packets (root vector
4, framing vector 2) and synchronisation packets (root vector 8, framing
vector 1). sACN universes are 1..63999. A data packet whose sync address
is non-zero is held until a sync packet with that address, the sACN
equivalent of ArtSync.

Anything else - other opcodes, discovery, preview data, a non-zero DMX
start code - parses to None and is ignored.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

ARTNET_PORT = 6454
SACN_PORT = 5568

ARTNET_ID = b"Art-Net\x00"
OP_POLL = 0x2000
OP_POLL_REPLY = 0x2100
OP_DMX = 0x5000
OP_SYNC = 0x5200
ARTNET_PROTOCOL = 14

SACN_ID = b"ASC-E1.17\x00\x00\x00"
VECTOR_ROOT_DATA = 0x00000004
VECTOR_ROOT_EXTENDED = 0x00000008
VECTOR_FRAMING_DATA = 0x00000002
VECTOR_FRAMING_SYNC = 0x00000001
SACN_OPTION_PREVIEW = 0x80
SACN_OPTION_TERMINATED = 0x40

UNIVERSE_CHANNELS = 512
PIXELS_PER_UNIVERSE = 170  # 510 channels of RGB; pixels never straddle a universe

ESTA_PROTOTYPE = 0x7FF0  # the ESTA manufacturer code reserved for prototypes
OEM_UNKNOWN = 0x00FF


@dataclass(frozen=True)
class Dmx:
    """One universe of channel data, from either protocol. `sync` is the
    sACN sync address (0: none); Art-Net signals sync out of band."""

    universe: int
    data: bytes
    sync: int = 0


@dataclass(frozen=True)
class Sync:
    """End of a frame: every universe before this belongs to it. `address`
    is sACN's sync address; 0 for ArtSync."""

    address: int = 0


@dataclass(frozen=True)
class Poll:
    """An Art-Net ArtPoll: a controller asking who is out there."""


# ---- Art-Net ----------------------------------------------------------------------------


def parse_artnet(packet: bytes) -> Dmx | Sync | Poll | None:
    if len(packet) < 12 or packet[:8] != ARTNET_ID:
        return None
    opcode = struct.unpack_from("<H", packet, 8)[0]
    if opcode == OP_DMX:
        if len(packet) < 18:
            return None
        sub_uni, net = packet[14], packet[15]
        length = struct.unpack_from(">H", packet, 16)[0]
        if not 2 <= length <= UNIVERSE_CHANNELS or len(packet) < 18 + length:
            return None
        return Dmx(universe=((net & 0x7F) << 8) | sub_uni, data=bytes(packet[18 : 18 + length]))
    if opcode == OP_SYNC:
        return Sync()
    if opcode == OP_POLL:
        return Poll()
    return None


def artdmx(universe: int, data: bytes, sequence: int = 0) -> bytes:
    """An ArtDmx packet - what a sender builds; here for tests and tools."""
    if len(data) % 2:
        data = data + b"\x00"  # the length must be even
    return (
        ARTNET_ID
        + struct.pack("<H", OP_DMX)
        + bytes((0, ARTNET_PROTOCOL, sequence & 0xFF, 0, universe & 0xFF, (universe >> 8) & 0x7F))
        + struct.pack(">H", len(data))
        + data
    )


def artsync() -> bytes:
    return ARTNET_ID + struct.pack("<H", OP_SYNC) + bytes((0, ARTNET_PROTOCOL, 0, 0))


def artpoll() -> bytes:
    return ARTNET_ID + struct.pack("<H", OP_POLL) + bytes((0, ARTNET_PROTOCOL, 0, 0))


def poll_replies(
    ip: str,
    universes: list[int],
    *,
    short_name: str = "Dance Floor",
    long_name: str = "Dance Floor v2 - 8x8 LED tiles",
    report: str = "",
    mac: bytes = b"\x00" * 6,
) -> list[bytes]:
    """The ArtPollReply packets describing `universes`: one per group of up
    to 4 that share a Net and Sub-Net, numbered by BindIndex from 1. Each
    port is an output port (Art-Net in, light out)."""
    groups: list[list[int]] = []
    for universe in sorted(set(universes)):
        if groups and len(groups[-1]) < 4 and groups[-1][0] >> 4 == universe >> 4:
            groups[-1].append(universe)
        else:
            groups.append([universe])
    ip_bytes = bytes(int(part) for part in ip.split("."))
    replies = []
    for index, group in enumerate(groups, start=1):
        first = group[0]
        ports = len(group)
        body = bytearray(239)
        body[0:8] = ARTNET_ID
        struct.pack_into("<H", body, 8, OP_POLL_REPLY)
        body[10:14] = ip_bytes
        struct.pack_into("<H", body, 14, ARTNET_PORT)
        body[16:18] = (0, 1)  # firmware version
        body[18] = (first >> 8) & 0x7F  # NetSwitch
        body[19] = (first >> 4) & 0x0F  # SubSwitch
        struct.pack_into(">H", body, 20, OEM_UNKNOWN)
        body[23] = 0xE0  # Status1: indicators normal (7-6 = 11), addresses set from the network (5-4 = 10)
        struct.pack_into("<H", body, 24, ESTA_PROTOTYPE)
        body[26:44] = _text(short_name, 18)
        body[44:108] = _text(long_name, 64)
        body[108:172] = _text(report, 64)
        struct.pack_into(">H", body, 172, ports)
        for p, universe in enumerate(group):
            body[174 + p] = 0x80  # PortTypes: can output DMX512 from Art-Net
            body[182 + p] = 0x80  # GoodOutputA: data being output
            body[190 + p] = universe & 0x0F  # SwOut
        body[200] = 0x00  # Style: StNode
        body[201:207] = mac[:6].ljust(6, b"\x00")
        body[207:211] = ip_bytes  # BindIp
        body[211] = index  # BindIndex
        body[212] = 0x08  # Status2: 15-bit port-addresses
        replies.append(bytes(body))
    return replies


def _text(text: str, size: int) -> bytes:
    """NUL-terminated ASCII in a fixed field."""
    return text.encode("ascii", "replace")[: size - 1].ljust(size, b"\x00")


# ---- sACN -------------------------------------------------------------------------------


def parse_sacn(packet: bytes) -> Dmx | Sync | None:
    if len(packet) < 38 or packet[4:16] != SACN_ID:
        return None
    root_vector = struct.unpack_from(">I", packet, 18)[0]
    if root_vector == VECTOR_ROOT_DATA:
        if len(packet) < 126:
            return None
        if struct.unpack_from(">I", packet, 40)[0] != VECTOR_FRAMING_DATA:
            return None
        options = packet[112]
        if options & (SACN_OPTION_PREVIEW | SACN_OPTION_TERMINATED):
            return None
        sync = struct.unpack_from(">H", packet, 109)[0]
        universe = struct.unpack_from(">H", packet, 113)[0]
        count = struct.unpack_from(">H", packet, 123)[0]  # start code + channels
        if packet[117] != 0x02 or packet[125] != 0x00 or count < 1:
            return None  # not DMP set-property, or not a DMX (start code 0) packet
        channels = min(count - 1, UNIVERSE_CHANNELS, len(packet) - 126)
        return Dmx(universe=universe, data=bytes(packet[126 : 126 + channels]), sync=sync)
    if root_vector == VECTOR_ROOT_EXTENDED:
        if len(packet) < 49 or struct.unpack_from(">I", packet, 40)[0] != VECTOR_FRAMING_SYNC:
            return None
        return Sync(address=struct.unpack_from(">H", packet, 45)[0])
    return None


def sacn_group(universe: int) -> str:
    """The multicast group a universe's data is sent to."""
    return f"239.255.{(universe >> 8) & 0xFF}.{universe & 0xFF}"


def sacn_data(universe: int, data: bytes, *, sync: int = 0, sequence: int = 0, options: int = 0) -> bytes:
    """An E1.31 data packet - what a sender builds; here for tests and tools."""
    channels = len(data)
    dmp = struct.pack(">HBBHHH", 0x7000 | (11 + channels), 0x02, 0xA1, 0, 1, channels + 1) + b"\x00" + data
    framing = (
        struct.pack(">HI", 0x7000 | (77 + len(dmp)), VECTOR_FRAMING_DATA)
        + _text("test", 64)
        + struct.pack(">BHBBH", 100, sync, sequence & 0xFF, options, universe)
        + dmp
    )
    root = struct.pack(">HH", 0x0010, 0) + SACN_ID + struct.pack(">HI", 0x7000 | (22 + len(framing)), VECTOR_ROOT_DATA)
    return root + b"\x00" * 16 + framing


def sacn_sync(address: int, sequence: int = 0) -> bytes:
    framing = struct.pack(">HIBHH", 0x7000 | 11, VECTOR_FRAMING_SYNC, sequence & 0xFF, address, 0)
    root = struct.pack(">HH", 0x0010, 0) + SACN_ID + struct.pack(">HI", 0x7000 | (22 + len(framing)), VECTOR_ROOT_EXTENDED)
    return root + b"\x00" * 16 + framing
