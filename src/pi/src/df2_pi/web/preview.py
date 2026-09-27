"""The preview stream: the floor's pixels over a WebSocket, per browser.

    WS  /ws/preview?format=full|tiles&fps=30    binary preview records
    GET /api/preview/geometry                   what a renderer must not hardcode

Each binary message is one record exactly as `encode_preview` produced it
(`VER FORMAT FLAGS FRAME_NO payload`, see output/preview.py): nothing is
re-encoded and nothing is JSON. Omit `fps` for every frame the runner
renders; `tiles` at 10 fps is the phone path.

Backpressure never reaches the runner. The sink gives each subscriber its
own thread and a latest-wins mailbox; this handler bridges that thread to
the event loop through a one-slot, latest-wins queue - the same rule once
more at the socket.

FLOW CONTROL IS BY ACK, not by the socket. The kernel's socket buffers
accept megabytes, so a send completes long before a slow browser has the
frame: without acks a slow client lags seconds behind instead of skipping.
So the client acknowledges with a text message `{"ack": n}`, which covers
every record up to and including FRAME_NO n, and the server keeps at most
`ACK_WINDOW` records unacknowledged. Past that it sends nothing and the
slot keeps only the newest frame: a slow browser sees fewer frames (FRAME_NO
gaps say how many), always current ones, and nothing behind it waits. A
client that never acks gets `ACK_WINDOW` records and then silence.

The client changes format or rate mid-stream with a text message,
`{"format": "tiles"}` or `{"fps": 10}` (`null` for every frame); the handler
re-subscribes. A malformed message gets a text `{"error": ...}` reply and
the stream carries on.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from df2_pi.output.preview import FORMATS, HEADER, Subscriber

if TYPE_CHECKING:
    from df2_pi.web.app import AppContext

log = logging.getLogger(__name__)

CLOSE_POLICY = 1008  # a bad query string: the request itself is refused
CLOSE_GOING_AWAY = 1001  # the sink dropped us or closed: the server is going
SINK_CHECK_S = 1.0  # how often an idle stream checks its subscription is alive
# Records sent and not yet acknowledged. Two keeps a LAN at full rate (an
# ack is back within a frame) and a 50 ms phone link near 40 fps.
ACK_WINDOW = 2


class Display(BaseModel):
    flip_y: bool = Field(description="Canonical y=0 is nearest the Pi and is drawn at the BOTTOM.")
    rule: str


class PreviewGeometry(BaseModel):
    tile_rows: int
    tile_cols: int
    tiles: int
    leds_per_side: int
    leds_per_tile: int
    led_count: int
    cell_size: int = Field(description="Cells per tile side: a tile is cell_size x cell_size, corners dark.")
    width: int = Field(description="Cells across the floor.")
    height: int = Field(description="Cells up the floor.")
    led_to_cell: list[int] = Field(
        description=(
            "Canonical (y, x) of every LED in chain order - the order of a `full` "
            "payload - flattened: [y0, x0, y1, x1, ...], 2 * led_count values."
        )
    )
    display: Display


def parse_client_message(text: str) -> dict[str, Any]:
    """A client text message: `{"ack": n}`, or a change of `format` and/or
    `fps`. Raises ValueError saying what is wrong."""
    request = json.loads(text)
    if not isinstance(request, dict) or not request:
        raise ValueError('expected {"ack": n}, or {"format": ...} and/or {"fps": ...}')
    if "ack" in request:
        ack = request["ack"]
        if set(request) != {"ack"} or isinstance(ack, bool) or not isinstance(ack, int) or ack < 0:
            raise ValueError(f'an ack is {{"ack": FRAME_NO}} and nothing else, got {text}')
        return request
    if not set(request) <= {"format", "fps"}:
        raise ValueError('expected {"ack": n}, or {"format": ...} and/or {"fps": ...}')
    return request


def validate_settings(fmt: Any, fps: Any) -> str | None:
    """Why this format/rate cannot be served, or None if it can."""
    if fmt not in FORMATS:
        return f"format must be one of {sorted(FORMATS)}, got {fmt!r}"
    if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not fps > 0):
        return f"fps must be a positive number or null, got {fps!r}"
    return None


def preview_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    preview = ctx.preview

    @router.get("/api/preview/geometry", response_model=PreviewGeometry, tags=["preview"])
    def preview_geometry() -> PreviewGeometry:
        """The floor's shape, fetched once by a renderer."""
        geometry = ctx.runner.geometry
        return PreviewGeometry(
            tile_rows=geometry.tile_rows,
            tile_cols=geometry.tile_cols,
            tiles=geometry.tiles,
            leds_per_side=geometry.leds_per_side,
            leds_per_tile=geometry.leds_per_tile,
            led_count=geometry.led_count,
            cell_size=geometry.cell_size,
            width=geometry.width,
            height=geometry.height,
            led_to_cell=geometry.led_to_cell.reshape(-1).tolist(),
            display=Display(flip_y=True, rule="display_y = height - 1 - y; x unchanged"),
        )

    @router.websocket("/ws/preview")
    async def preview_socket(websocket: WebSocket, format: str = "full", fps: float | None = None) -> None:
        problem = validate_settings(format, fps)
        if problem is not None:
            await websocket.close(code=CLOSE_POLICY, reason=problem)
            return
        await websocket.accept()

        loop = asyncio.get_running_loop()
        slot: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

        def put_latest(record: bytes) -> None:
            if slot.full():
                slot.get_nowait()  # latest wins: the unsent frame is stale
            slot.put_nowait(record)

        def send(record: bytes) -> None:
            # On the subscriber's thread: hand over and return at once.
            loop.call_soon_threadsafe(put_latest, record)

        client = websocket.client
        name = f"ws:{client.host}:{client.port}" if client else "ws"
        settings = {"format": format, "fps": fps}
        subscriber: Subscriber = preview.subscribe(send, format, fps, name=name)

        unacked: deque[int] = deque()  # FRAME_NOs sent and not yet acknowledged
        credit = asyncio.Event()  # set by an ack

        async def pump() -> None:
            while True:
                record = None
                with anyio.move_on_after(SINK_CHECK_S):
                    if len(unacked) >= ACK_WINDOW:
                        credit.clear()
                        await credit.wait()  # the slot keeps taking the newest meanwhile
                    else:
                        record = await slot.get()
                if record is None:
                    if subscriber.mailbox.closed:  # dropped by the sink, or the sink closed
                        await websocket.close(code=CLOSE_GOING_AWAY)
                        return
                    continue
                await websocket.send_bytes(record)
                unacked.append(HEADER.unpack_from(record)[3])

        async def listen() -> None:
            nonlocal subscriber
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                text = message.get("text")
                if text is None:
                    continue
                try:
                    request = parse_client_message(text)
                except ValueError as exc:
                    await websocket.send_text(json.dumps({"error": str(exc)}))
                    continue
                if "ack" in request:
                    while unacked and unacked[0] <= request["ack"]:
                        unacked.popleft()
                    credit.set()
                    continue
                wanted = {**settings, **request}
                problem = validate_settings(wanted["format"], wanted["fps"])
                if problem is not None:
                    await websocket.send_text(json.dumps({"error": problem}))
                    continue
                settings.update(wanted)
                preview.unsubscribe(subscriber)
                subscriber = preview.subscribe(send, settings["format"], settings["fps"], name=name)

        try:
            async with anyio.create_task_group() as group:

                async def until_done(loop_fn) -> None:
                    # Whichever side ends - the client went, or the sink
                    # dropped us - ends the other.
                    try:
                        await loop_fn()
                    except (WebSocketDisconnect, RuntimeError):  # sending on a closed socket
                        pass
                    group.cancel_scope.cancel()

                group.start_soon(until_done, pump)
                group.start_soon(until_done, listen)
        finally:
            preview.unsubscribe(subscriber)

    return router
