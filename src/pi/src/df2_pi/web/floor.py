"""The floor's own view of itself: Row Bus admin requests, for diagnostics.

    GET  /api/floor/status     STATUS from every row: state, tiles found, uptime
    GET  /api/floor/version    VERSION (and STATUS) from every row, with what is out of step
    POST /api/floor/blackout   the BLACKOUT broadcast itself

The same queries as `df2-pi status` / `df2-pi version`. Each one waits for
a row's reply, so none may overlap a frame on the wire: every request runs
on the render thread at a frame boundary (`Runner.call`), ONE ROW PER
BOUNDARY, so a row that answers costs a frame a millisecond or two of its
slack and a row that does not (three 20 ms timeouts) costs at most one
frame. They are for a button, never a poll.

Without a floor (`--no-hardware`) they are 503.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from df2_pi.protocol.constants import Cmd
from df2_pi.protocol.firmware_version import FirmwareVersion, format_version
from df2_pi.row_status import RowStatus
from df2_pi.version_report import RowVersionReport, assess_versions

if TYPE_CHECKING:
    from df2_pi.web.app import AppContext

# The render thread takes a call at its next frame boundary: normally within
# a frame. Longer than this means it is stuck, or stopped under us.
RENDER_THREAD_TIMEOUT_S = 5.0


class RowStatusInfo(BaseModel):
    row: int
    chain: int
    responding: bool
    state: str | None = None
    tiles_found: int | None = None
    tile_status: list[int] | None = Field(default=None, description="One status byte per tile slot.")
    uptime_s: int | None = Field(default=None, description="Null for firmware that predates the field.")
    error: str | None = None


class FloorStatus(BaseModel):
    rows: list[RowStatusInfo]


class VersionInfo(BaseModel):
    version: int
    git_sha: str
    dirty: bool
    text: str


class TileVersionInfo(BaseModel):
    slot: int
    version: VersionInfo | None = Field(description="Null: the row has no version for this slot; `state` says why.")
    state: str = Field(
        description='"ok" with a version. Without: "empty" (no tile found), "no version" (found, never answered '
        'VERSION), "not responding" or "test failed" - from the row\'s STATUS for that slot.'
    )
    out_of_step: bool = Field(description="Differs from the majority, dirty, or missing a version. An empty slot is not.")


class RowVersionInfo(BaseModel):
    row: int
    responding: bool
    version: VersionInfo | None
    out_of_step: bool = Field(description="Not responding, dirty, or differing from most of the floor.")
    tiles: list[TileVersionInfo]


class FloorVersions(BaseModel):
    rows: list[RowVersionInfo]
    ok: bool = Field(description="Every row and tile responding, clean, and in step.")
    row_version: VersionInfo | None = Field(description="What most rows run: what each row is compared against.")
    tile_version: VersionInfo | None = Field(description="What most tiles run: what each tile is compared against.")


def version_info(v: FirmwareVersion | None) -> VersionInfo | None:
    if v is None:
        return None
    return VersionInfo(version=v.version, git_sha=f"{v.git_sha:08x}", dirty=v.dirty, text=format_version(v))


def floor_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/floor", tags=["floor"])

    def the_floor():
        sink = ctx.fanout.get("hardware")
        if sink is None or getattr(sink, "floor", None) is None:
            raise HTTPException(503, "no floor attached: the server was started with --no-hardware")
        return sink.floor

    async def between_frames(fn: Callable[[], Any]) -> Any:
        """`fn` on the render thread at the next frame boundary."""
        if not ctx.runner.alive:
            raise HTTPException(503, "the runner is not running")
        future = ctx.runner.call(fn)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), RENDER_THREAD_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            future.cancel()
            raise HTTPException(503, "the render thread did not take the request") from exc

    def ask(floor, row: int, cmd: int):
        """One admin request; the reply frame, or the exception it raised."""

        def request():
            try:
                return floor.request(row, cmd)
            except Exception as exc:  # RowNotResponding, a serial error: a fact about that row
                return exc

        return between_frames(request)

    @router.get("/status", response_model=FloorStatus)
    async def status() -> FloorStatus:
        """STATUS from every row, one row per frame boundary."""
        floor = the_floor()
        rows = []
        for row, chain in floor.chain_map.items():
            reply = await ask(floor, row, Cmd.STATUS)
            if isinstance(reply, Exception):
                rows.append(RowStatusInfo(row=row, chain=chain, responding=False, error=str(reply)))
                continue
            try:
                s = RowStatus.decode(reply.payload)
            except ValueError as exc:
                rows.append(RowStatusInfo(row=row, chain=chain, responding=True, error=str(exc)))
                continue
            rows.append(
                RowStatusInfo(
                    row=row,
                    chain=chain,
                    responding=True,
                    state=s.state_name,
                    tiles_found=s.tiles_found,
                    tile_status=list(s.tile_status),
                    uptime_s=s.uptime_s,
                )
            )
        return FloorStatus(rows=rows)

    @router.get("/version", response_model=FloorVersions)
    async def version() -> FloorVersions:
        """VERSION from every row, and what is out of step against the
        majority - `df2-pi version`'s verdict. Each row's STATUS as well, so
        a tile with no version is told apart: an empty slot, or a tile that
        is there and did not answer. One request per frame boundary."""
        floor = the_floor()
        reports: dict[int, RowVersionReport | None] = {}
        slot_status: dict[int, tuple[int, ...]] = {}
        for row, _chain in floor.chain_map.items():
            reply = await ask(floor, row, Cmd.VERSION)
            try:
                reports[row] = None if isinstance(reply, Exception) else RowVersionReport.decode(reply.payload)
            except ValueError:
                reports[row] = None
            if reports[row] is None:
                continue
            status = await ask(floor, row, Cmd.STATUS)
            if not isinstance(status, Exception):
                try:
                    slot_status[row] = RowStatus.decode(status.payload).tile_status
                except ValueError:
                    pass  # unlabelled: its missing versions stay "no version"
        assessment = assess_versions(reports, slot_status)
        return FloorVersions(
            rows=[
                RowVersionInfo(
                    row=r.row,
                    responding=r.version is not None,
                    version=version_info(r.version),
                    out_of_step=r.out_of_step,
                    tiles=[
                        TileVersionInfo(slot=t.slot, version=version_info(t.version), state=t.state, out_of_step=t.out_of_step)
                        for t in r.tiles
                    ],
                )
                for r in assessment.rows
            ],
            ok=assessment.ok,
            row_version=version_info(assessment.row_majority),
            tile_version=version_info(assessment.tile_majority),
        )

    @router.post("/blackout", status_code=204)
    async def blackout() -> None:
        """The BLACKOUT broadcast: every tile's pixels and effect register
        cleared. Unlike the transport blackout this mutes nothing - while
        the runner plays, the next frame lights the floor again."""
        floor = the_floor()
        await between_frames(floor.blackout)

    return router
