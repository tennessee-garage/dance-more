"""The preview wire format, `PreviewSink`, and `RecorderSink`.

The floor preview in the admin UI is THE secondary sink in practice: a VJ
has it open on a laptop beside the floor for the whole session. So it gets
its own sink rather than an ad-hoc callback. Everything above the socket
lives here - serialisation, per-subscriber rate and format, fan-out with a
latest-wins mailbox per subscriber so one stalled browser cannot affect
another. The web layer supplies only the WebSocket transport: a `send(
bytes)` callable per connection.

Wire format - binary, never JSON or base64:

     0        1        2        3 .. 6      7 ..
    +--------+--------+--------+-----------+-------------+
    | VER    | FORMAT | FLAGS  | FRAME_NO  |   PAYLOAD   |
    | 0x01   |        |        | uint32 BE |             |
    +--------+--------+--------+-----------+-------------+

    FORMAT 0x01 "full"   every LED x RGB, canonical chain order   11,520 B
    FORMAT 0x02 "tiles"  one RGB per tile (PixelFrame.to_tiles)      192 B
    FLAGS  bit 0         the animation produced a TileFrame (a "full"
                         payload is then that colour repeated per LED)

Full at 30 FPS is ~230 KB/s per client - fine on the LAN the floor lives
on, too much over a phone hotspot, which is what "tiles" is for.
FRAME_NO is the clock's frame index, so a client can see drops.

The payload is the frame AS THE ANIMATION PRODUCED IT: global brightness
is reported in the state snapshot and not applied to the preview.

`RecorderSink` appends the same records to a file, each prefixed with a
length and the frame's `t`, for replay and for the GIF/mp4 export that
builds on it. `read_recording()` is the inverse.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Mapping

import numpy as np

from df2_pi.effects import Effect
from df2_pi.engine.clock import FrameInfo
from df2_pi.geometry import FloorGeometry
from df2_pi.output.sink import DEFAULT_MAX_FAILURES, Mailbox, ThreadedSink
from df2_pi.pixels import CHANNELS, Frame, PixelFrame, TileFrame, default_geometry

log = logging.getLogger(__name__)

PREVIEW_VERSION = 0x01
FORMAT_FULL = 0x01
FORMAT_TILES = 0x02
FORMATS = {"full": FORMAT_FULL, "tiles": FORMAT_TILES}
FLAG_TILE_SOURCE = 0x01
HEADER = struct.Struct(">BBBI")  # VER FORMAT FLAGS FRAME_NO
HEADER_SIZE = HEADER.size


def encode_preview(frame: Frame, frame_no: int, fmt: str) -> bytes:
    """One preview record for `frame`."""
    code = FORMATS[fmt]
    flags = FLAG_TILE_SOURCE if isinstance(frame, TileFrame) else 0
    if code == FORMAT_FULL:
        pixels = frame.to_pixels() if isinstance(frame, TileFrame) else frame
        payload = pixels.data.tobytes()
    else:
        tiles = frame.to_tiles() if isinstance(frame, PixelFrame) else frame
        payload = tiles.data.tobytes()
    return HEADER.pack(PREVIEW_VERSION, code, flags, frame_no & 0xFFFFFFFF) + payload


@dataclass(frozen=True)
class PreviewRecord:
    frame_no: int
    fmt: str
    tile_source: bool
    frame: Frame


def decode_preview(record: bytes, geometry: FloorGeometry | None = None) -> PreviewRecord:
    """The inverse of `encode_preview`."""
    geometry = geometry if geometry is not None else default_geometry()
    if len(record) < HEADER_SIZE:
        raise ValueError("preview record shorter than its header")
    version, code, flags, frame_no = HEADER.unpack_from(record)
    if version != PREVIEW_VERSION:
        raise ValueError(f"unsupported preview version {version:#x}")
    payload = np.frombuffer(record, dtype=np.uint8, offset=HEADER_SIZE)
    if code == FORMAT_FULL:
        frame: Frame = PixelFrame(payload.reshape(PixelFrame.shape_for(geometry)).copy(), geometry)
        fmt = "full"
    elif code == FORMAT_TILES:
        frame = TileFrame(payload.reshape(TileFrame.shape_for(geometry)).copy(), geometry)
        fmt = "tiles"
    else:
        raise ValueError(f"unknown preview format {code:#x}")
    return PreviewRecord(frame_no, fmt, bool(flags & FLAG_TILE_SOURCE), frame)


# ---- PreviewSink -------------------------------------------------------------------------


class Subscriber:
    """One remote viewer: a `send(bytes)` callable behind its own mailbox
    and thread, with its own format and rate cap."""

    def __init__(
        self,
        send: Callable[[bytes], None],
        fmt: str,
        max_fps: float | None,
        *,
        name: str,
        max_failures: int,
        on_drop: Callable[[Subscriber, str], None],
    ) -> None:
        if fmt not in FORMATS:
            raise ValueError(f"format must be one of {sorted(FORMATS)}, got {fmt!r}")
        if max_fps is not None and max_fps <= 0:
            raise ValueError(f"max_fps must be positive, got {max_fps}")
        self.send = send
        self.fmt = fmt
        self.min_interval = 0.0 if max_fps is None else 1.0 / max_fps
        self.name = name
        self.max_failures = max_failures
        self.mailbox = Mailbox()
        self.sent = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.last_t: float | None = None
        self._on_drop = on_drop
        self._thread = threading.Thread(target=self._run, name=f"preview:{name}", daemon=True)
        self._thread.start()

    def offer(self, t: float, record: bytes) -> bool:
        """Rate-limit against the frame's `t`, then mailbox the record."""
        if self.last_t is not None and t - self.last_t < self.min_interval - 1e-9:
            return False
        self.last_t = t
        self.mailbox.put(record)
        return True

    @property
    def dropped(self) -> int:
        return self.mailbox.replaced

    def close(self) -> None:
        self.mailbox.close()

    def _run(self) -> None:
        while True:
            record = self.mailbox.take()
            if record is None:
                return
            try:
                self.send(record)
            except Exception as exc:
                self.failures += 1
                self.consecutive_failures += 1
                reason = f"{type(exc).__name__}: {exc}"
                log.warning("preview subscriber %s: send failed: %s", self.name, reason)
                if self.consecutive_failures >= self.max_failures:
                    self.mailbox.close()
                    self._on_drop(self, reason)
                    return
            else:
                self.sent += 1
                self.consecutive_failures = 0


class PreviewSink(ThreadedSink):
    """Serialise each frame once per format in use and fan it out to every
    subscriber's mailbox. Subscribe and unsubscribe at runtime."""

    def __init__(self, name: str = "preview", *, max_failures: int = DEFAULT_MAX_FAILURES) -> None:
        super().__init__(name, max_failures=max_failures)
        self._lock = threading.Lock()
        self._subscribers: list[Subscriber] = []
        self._counter = 0
        self.dropped_subscribers: dict[str, str] = {}

    def subscribe(
        self,
        send: Callable[[bytes], None],
        fmt: str = "full",
        max_fps: float | None = None,
        *,
        name: str | None = None,
        max_failures: int = DEFAULT_MAX_FAILURES,
    ) -> Subscriber:
        """Add a viewer. `send` is called on the subscriber's own thread with
        each record; raising `max_failures` times in a row drops it."""
        with self._lock:
            self._counter += 1
            sub = Subscriber(
                send,
                fmt,
                max_fps,
                name=name or f"sub{self._counter}",
                max_failures=max_failures,
                on_drop=self._dropped,
            )
            self._subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)
        sub.close()

    @property
    def subscribers(self) -> tuple[Subscriber, ...]:
        with self._lock:
            return tuple(self._subscribers)

    def handle(self, frame: Frame, info: FrameInfo) -> None:
        subs = self.subscribers
        if not subs:
            return
        records: dict[str, bytes] = {}
        for sub in subs:
            if sub.mailbox.closed:
                continue
            if sub.fmt not in records:
                records[sub.fmt] = encode_preview(frame, info.n, sub.fmt)
            sub.offer(info.t, records[sub.fmt])

    def on_close(self) -> None:
        for sub in self.subscribers:
            sub.close()

    def _dropped(self, sub: Subscriber, reason: str) -> None:
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)
        self.dropped_subscribers[sub.name] = reason
        log.error("preview subscriber %s dropped: %s", sub.name, reason)


# ---- RecorderSink --------------------------------------------------------------------------

RECORDING_MAGIC = b"DF2REC\x01\x00"
RECORD_PREFIX = struct.Struct(">Id")  # record length, frame t


class RecorderSink(ThreadedSink):
    """Append every frame it gets to `path`: the magic, then per frame a
    `(uint32 length, float64 t)` prefix and a preview record. Latest-wins
    like every observer, so a slow disk drops frames rather than stalling
    the floor; FRAME_NO in each record says where."""

    def __init__(
        self,
        path: Path | str,
        fmt: str = "full",
        *,
        name: str = "recorder",
        max_failures: int = DEFAULT_MAX_FAILURES,
    ) -> None:
        if fmt not in FORMATS:
            raise ValueError(f"format must be one of {sorted(FORMATS)}, got {fmt!r}")
        super().__init__(name, max_failures=max_failures)
        self.path = Path(path)
        self.fmt = fmt
        self._file: BinaryIO | None = None
        self.records = 0

    def handle(self, frame: Frame, info: FrameInfo) -> None:
        if self._file is None:
            self._file = open(self.path, "wb")
            self._file.write(RECORDING_MAGIC)
        record = encode_preview(frame, info.n, self.fmt)
        self._file.write(RECORD_PREFIX.pack(len(record), info.t))
        self._file.write(record)
        self.records += 1

    def on_close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


@dataclass(frozen=True)
class RecordedFrame:
    t: float
    record: PreviewRecord

    @property
    def frame(self) -> Frame:
        return self.record.frame

    @property
    def frame_no(self) -> int:
        return self.record.frame_no


def read_recording(path: Path | str, geometry: FloorGeometry | None = None) -> Iterator[RecordedFrame]:
    """Replay a `RecorderSink` file."""
    with open(path, "rb") as f:
        if f.read(len(RECORDING_MAGIC)) != RECORDING_MAGIC:
            raise ValueError(f"{path} is not a df2 recording")
        while True:
            prefix = f.read(RECORD_PREFIX.size)
            if not prefix:
                return
            if len(prefix) < RECORD_PREFIX.size:
                raise ValueError(f"{path}: truncated record prefix")
            length, t = RECORD_PREFIX.unpack(prefix)
            record = f.read(length)
            if len(record) < length:
                raise ValueError(f"{path}: truncated record")
            yield RecordedFrame(t, decode_preview(record, geometry))


def wait_for(predicate: Callable[[], bool], timeout: float = 2.0, interval: float = 0.001) -> bool:
    """Poll `predicate` until true or `timeout`. For tests and shutdown."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
