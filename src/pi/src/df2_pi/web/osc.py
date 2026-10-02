"""The OSC control surface's settings and status.

    GET   /api/osc     settings + status: listening, port, counts, subscribers,
                       and the most recent messages (the External tab's monitor)
    PATCH /api/osc     {"enabled": bool, "port": int}: applied now (the server
                       restarts) and stored

The OSC addresses themselves are in interfacing/osc.py. With no OSC control
(a test app) both answer 503.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from df2_pi.interfacing.osc import OscControl
    from df2_pi.web.app import AppContext


class OscSettings(BaseModel):
    enabled: bool
    port: int = Field(description="UDP port the floor listens on for OSC.")


class OscChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = None
    port: int = Field(default=None, ge=1, le=65535)


class OscInfo(BaseModel):
    settings: OscSettings
    status: dict[str, Any] = Field(description="listening, port, error, received, errors, subscribers, recent")


def osc_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/osc", tags=["osc"])

    def control() -> OscControl:
        if ctx.osc is None:
            raise HTTPException(503, "OSC is not running")
        return ctx.osc

    @router.get("", response_model=OscInfo)
    def get() -> OscInfo:
        osc = control()
        return OscInfo(settings=OscSettings(**osc.settings()), status=osc.status())

    @router.patch("", response_model=OscInfo)
    def patch(body: OscChanges) -> OscInfo:
        osc = control()
        try:
            osc.configure(**body.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return OscInfo(settings=OscSettings(**osc.settings()), status=osc.status())

    return router
