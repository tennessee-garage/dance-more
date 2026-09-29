"""The Art-Net / sACN receiver: UDP in, complete frames out.

    receiver = Receiver(Layout("grid", 34, 34), artnet_universe=0, sacn_universe=1)
    receiver.start()
    frame = receiver.latest()      # ExternalFrame or None, from any thread
    receiver.stop()

A frame of floor pixels spans one or more universes (`Layout.universes`),
so the receiver's job is to know when a frame is COMPLETE - latching half
of one frame and half of the next tears the picture - and to publish only
complete ones. `Assembler` does that, per protocol:

- Once a sender uses sync (ArtSync, or sACN sync packets) the frame is
  complete when the sync arrives. With no sync for `SYNC_TIMEOUT_S`, it
  falls back to the next rule, as Art-Net 4 specifies.
- Without sync, the frame is complete when every universe has arrived
  since the last one. A universe arriving twice first means the sender
  has started a new frame without sending them all (a misconfigured
  universe count, say): what there is gets published, missing universes
  keeping their last data, rather than never showing anything.

Published frames are latest-wins: a single reference swapped atomically,
read by the render thread at its own 30 fps whatever rate the sender
runs at, never queued.

The receiver answers ArtPoll with ArtPollReply, so the floor shows up in
a sender's node list. sACN is received both unicast and on the multicast
group of each configured universe.

A DMX control block (dmx_control.py) can ride along: with
`configure_control()` set, every data packet for the control universe of
its protocol that covers the whole block is handed to `on_control` - on
this thread, as it arrives, whether or not that universe also carries
pixels. A packet that stops short of the block is ignored rather than
padded, since 0 on the first channel means "dimmer off".

Sockets are opened with SO_REUSEADDR (and SO_REUSEPORT where there is
one) so another Art-Net program on the same machine can share the port.
"""

from __future__ import annotations

import logging
import selectors
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from df2_pi.pixels import default_geometry
from df2_pi.interfacing.packets import (
    ARTNET_PORT,
    PIXELS_PER_UNIVERSE,
    SACN_PORT,
    UNIVERSE_CHANNELS,
    Dmx,
    Poll,
    Sync,
    parse_artnet,
    parse_sacn,
    poll_replies,
    sacn_group,
)

log = logging.getLogger(__name__)

SYNC_TIMEOUT_S = 4.0
MODES = ("tile", "grid", "raw")


@dataclass(frozen=True)
class Layout:
    """How channels map onto the floor. `tile`: 64 RGB tiles, raster order
    from the top-left of the canonical view. `grid`: a `width` x `height`
    RGB image, raster order from the top-left, sampled at each LED.
    `raw`: every LED in chain order. Pixels pack 170 to a universe."""

    mode: str = "tile"
    width: int = 34
    height: int = 34

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not (1 <= self.width <= 136 and 1 <= self.height <= 136):
            raise ValueError(f"grid must be 1..136 on each side, got {self.width}x{self.height}")

    @property
    def pixels(self) -> int:
        geometry = default_geometry()
        if self.mode == "tile":
            return geometry.tiles
        if self.mode == "grid":
            return self.width * self.height
        return geometry.led_count

    @property
    def universes(self) -> int:
        return -(-self.pixels // PIXELS_PER_UNIVERSE)


@dataclass(frozen=True)
class ExternalFrame:
    """One complete frame: `channels` is (universes, 512) uint8, `seq`
    counts frames published by this receiver."""

    channels: np.ndarray
    layout: Layout
    protocol: str
    sender: str
    received_at: float
    seq: int


@dataclass
class Assembler:
    """Universes -> complete frames, for one protocol. Pure: time is passed
    in, nothing here touches a socket."""

    protocol: str
    first: int
    layout: Layout
    _buffers: np.ndarray = field(init=False)
    _received: set[int] = field(init=False, default_factory=set)
    _last_sync: float | None = field(init=False, default=None)
    _sender: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self._buffers = np.zeros((self.layout.universes, UNIVERSE_CHANNELS), dtype=np.uint8)

    def reconfigure(self, first: int, layout: Layout) -> None:
        if first != self.first or layout != self.layout:
            self.first, self.layout = first, layout
            self.__post_init__()
            self._received.clear()

    def synced(self, now: float) -> bool:
        return self._last_sync is not None and now - self._last_sync < SYNC_TIMEOUT_S

    def dmx(self, packet: Dmx, sender: str, now: float) -> np.ndarray | None:
        """Take one universe; the complete frame's channels if this finished one."""
        index = packet.universe - self.first
        if not 0 <= index < len(self._buffers):
            return None
        done = None
        synced = self.synced(now)
        if index in self._received and not synced:
            done = self._publish()  # a new frame started before the last was whole
        self._buffers[index, : len(packet.data)] = np.frombuffer(packet.data, dtype=np.uint8)
        self._received.add(index)
        self._sender = sender
        if done is None and not synced and len(self._received) == len(self._buffers):
            done = self._publish()
        return done

    def sync(self, now: float) -> np.ndarray | None:
        self._last_sync = now
        return self._publish() if self._received else None

    @property
    def sender(self) -> str:
        return self._sender

    def _publish(self) -> np.ndarray:
        self._received.clear()
        return self._buffers.copy()


class Receiver:
    def __init__(
        self,
        layout: Layout,
        *,
        artnet_universe: int = 0,
        sacn_universe: int = 1,
        artnet: bool = True,
        sacn: bool = True,
        artnet_port: int = ARTNET_PORT,
        sacn_port: int = SACN_PORT,
        reply_port: int = ARTNET_PORT,
        bind: str = "",
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._now = now
        self._bind = bind
        self._ports = {"artnet": artnet_port, "sacn": sacn_port}
        self._reply_port = reply_port
        self._enabled = {"artnet": artnet, "sacn": sacn}
        self._assemblers = {
            "artnet": Assembler("artnet", artnet_universe, layout),
            "sacn": Assembler("sacn", sacn_universe, layout),
        }
        self._lock = threading.Lock()  # guards the assemblers against configure()
        self._latest: ExternalFrame | None = None
        self._seq = 0
        self._sockets: dict[str, socket.socket] = {}
        self._joined: set[str] = set()
        self._thread: threading.Thread | None = None
        self._running = False
        self.errors: dict[str, str] = {}  # protocol -> why its socket is not open
        self.packets = 0
        self.polls = 0
        # The DMX control block: protocol -> universe, 1-based start address, width.
        self._control: dict[str, int] | None = None
        self._control_address = 1
        self._control_width = 0
        self.on_control: Callable[[bytes], None] | None = None

    # ---- any thread ---------------------------------------------------------------------

    def latest(self) -> ExternalFrame | None:
        return self._latest

    @property
    def layout(self) -> Layout:
        return self._assemblers["artnet"].layout

    def ports(self) -> dict[str, int]:
        """The bound port per open protocol (the real one, when 0 was asked for)."""
        return {name: sock.getsockname()[1] for name, sock in self._sockets.items()}

    def configure(
        self,
        *,
        layout: Layout | None = None,
        artnet_universe: int | None = None,
        sacn_universe: int | None = None,
        artnet: bool | None = None,
        sacn: bool | None = None,
    ) -> None:
        """Change what is listened for, live. Opening or closing a protocol
        takes effect immediately when running."""
        with self._lock:
            art, sac = self._assemblers["artnet"], self._assemblers["sacn"]
            layout = layout if layout is not None else art.layout
            art.reconfigure(art.first if artnet_universe is None else artnet_universe, layout)
            sac.reconfigure(sac.first if sacn_universe is None else sacn_universe, layout)
            if artnet is not None:
                self._enabled["artnet"] = artnet
            if sacn is not None:
                self._enabled["sacn"] = sacn
            self._latest = None  # a frame in the old layout is no longer decodable as the new one
        if self._running:
            self._open_sockets()

    def configure_control(self, *, artnet_universe: int, sacn_universe: int, address: int, width: int, enabled: bool = True) -> None:
        """Where the DMX control block is, or `enabled=False` for nowhere."""
        self._control = {"artnet": artnet_universe, "sacn": sacn_universe} if enabled else None
        self._control_address, self._control_width = address, width
        if self._running and "sacn" in self._sockets:
            self._join_sacn_groups()

    def status(self) -> dict:
        latest = self._latest
        return {
            "listening": sorted(self._sockets),
            "errors": dict(self.errors),
            "packets": self.packets,
            "polls": self.polls,
            "frames": self._seq,
            "last_frame_at": None if latest is None else latest.received_at,
            "protocol": None if latest is None else latest.protocol,
            "sender": None if latest is None else latest.sender,
        }

    def start(self) -> None:
        self._running = True
        self._open_sockets()
        self._thread = threading.Thread(target=self._run, name="external-in", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(2.0)
        for name in list(self._sockets):
            self._close(name)

    # ---- sockets ------------------------------------------------------------------------

    def _open_sockets(self) -> None:
        for name in ("artnet", "sacn"):
            if self._enabled[name] and name not in self._sockets:
                try:
                    self._sockets[name] = self._open(self._ports[name])
                    self.errors.pop(name, None)
                except OSError as exc:
                    self.errors[name] = str(exc)
                    log.warning("%s: cannot listen on UDP %d: %s", name, self._ports[name], exc)
            elif not self._enabled[name] and name in self._sockets:
                self._close(name)
        if "sacn" in self._sockets:
            self._join_sacn_groups()

    def _open(self, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((self._bind, port))
        sock.setblocking(False)
        return sock

    def _close(self, name: str) -> None:
        sock = self._sockets.pop(name, None)
        if sock is not None:
            sock.close()
        if name == "sacn":
            self._joined.clear()

    def _join_sacn_groups(self) -> None:
        sock = self._sockets["sacn"]
        sac = self._assemblers["sacn"]
        wanted = {sacn_group(sac.first + i) for i in range(sac.layout.universes)}
        if self._control is not None:
            wanted.add(sacn_group(self._control["sacn"]))
        for group in self._joined - wanted:
            self._membership(sock, socket.IP_DROP_MEMBERSHIP, group)
        for group in wanted - self._joined:
            self._membership(sock, socket.IP_ADD_MEMBERSHIP, group)
        self._joined = wanted

    @staticmethod
    def _membership(sock: socket.socket, option: int, group: str) -> None:
        try:
            sock.setsockopt(socket.IPPROTO_IP, option, struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton("0.0.0.0")))
        except OSError as exc:
            log.debug("sACN multicast %s %s: %s", "join" if option == socket.IP_ADD_MEMBERSHIP else "leave", group, exc)

    # ---- the receive thread -------------------------------------------------------------

    def _run(self) -> None:
        selector = selectors.DefaultSelector()
        registered: dict[socket.socket, str] = {}
        while self._running:
            current = {sock: name for name, sock in self._sockets.items()}
            if current != registered:
                for sock in registered:
                    try:
                        selector.unregister(sock)
                    except (KeyError, ValueError):
                        pass
                for sock, name in current.items():
                    selector.register(sock, selectors.EVENT_READ, name)
                registered = current
            if not registered:
                time.sleep(0.25)
                continue
            for key, _ in selector.select(timeout=0.25):
                sock = key.fileobj
                try:
                    packet, (host, port) = sock.recvfrom(1024)
                except (BlockingIOError, OSError):
                    continue
                self.packets += 1
                try:
                    self._handle(key.data, packet, host, sock)
                except Exception:
                    log.exception("bad %s packet from %s", key.data, host)
        selector.close()

    def _handle(self, protocol: str, packet: bytes, host: str, sock: socket.socket) -> None:
        parsed = parse_artnet(packet) if protocol == "artnet" else parse_sacn(packet)
        if parsed is None:
            return
        if isinstance(parsed, Poll):
            self.polls += 1
            self._reply_to_poll(host, sock)
            return
        control, callback = self._control, self.on_control
        if control is not None and callback is not None and isinstance(parsed, Dmx) and parsed.universe == control[protocol]:
            start = self._control_address - 1
            if len(parsed.data) >= start + self._control_width:
                callback(parsed.data[start : start + self._control_width])
        now = self._now()
        with self._lock:
            assembler = self._assemblers[protocol]
            channels = assembler.dmx(parsed, host, now) if isinstance(parsed, Dmx) else assembler.sync(now)
            if channels is None:
                return
            self._seq += 1
            self._latest = ExternalFrame(channels, assembler.layout, protocol, assembler.sender, now, self._seq)

    def _reply_to_poll(self, host: str, sock: socket.socket) -> None:
        art = self._assemblers["artnet"]
        universes = [art.first + i for i in range(art.layout.universes)]
        report = f"#0001 [{self._seq:04d}] {art.layout.mode} mode, {len(universes)} universe(s)"
        for reply in poll_replies(_local_ip_toward(host), universes, report=report, mac=_mac()):
            try:
                sock.sendto(reply, (host, self._reply_port))
            except OSError as exc:
                log.debug("ArtPollReply to %s: %s", host, exc)


def _local_ip_toward(host: str) -> str:
    """This machine's address on the route to `host` - what an ArtPollReply
    must carry. No packet is sent."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((host, ARTNET_PORT))
        return probe.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        probe.close()


def _mac() -> bytes:
    return uuid.getnode().to_bytes(6, "big")
