"""The External input API: Art-Net / sACN status and settings.

    GET   /api/external     settings + live status
    PATCH /api/external     change settings; applied now and stored

Settings are validated as a whole before anything changes. With the
receiver not running (`df2-pi serve --no-external`, or a test app) both
answer 503.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from df2_pi.web.app import AppContext


class ExternalSettings(BaseModel):
    artnet_enabled: bool
    sacn_enabled: bool
    mode: Literal["tile", "grid", "raw"] = Field(description="tile: 64 tile colours; grid: a W x H image sampled at each LED; raw: every LED in chain order.")
    grid_width: int
    grid_height: int
    artnet_universe: int = Field(description="The first Art-Net port-address (0-based) listened to.")
    sacn_universe: int = Field(description="The first sACN universe (1-based) listened to.")
    source: Literal["internal", "external", "mix"] = Field(description="What the floor shows while there is signal.")
    mix: float = Field(description="External's amount over the playlist in mix.")
    timeout_s: float = Field(description="Seconds without a frame before the floor takes its own show back.")
    dmx_enabled: bool = Field(description="Listen for the 18-channel DMX control block.")
    dmx_artnet_universe: int = Field(description="The control block's Art-Net universe (0-based).")
    dmx_sacn_universe: int = Field(description="The control block's sACN universe (1-based).")
    dmx_address: int = Field(description="The control block's first channel (1-based).")


class ExternalChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artnet_enabled: bool = None
    sacn_enabled: bool = None
    mode: Literal["tile", "grid", "raw"] = None
    grid_width: int = Field(default=None, ge=1, le=136)
    grid_height: int = Field(default=None, ge=1, le=136)
    artnet_universe: int = Field(default=None, ge=0)
    sacn_universe: int = Field(default=None, ge=1)
    source: Literal["internal", "external", "mix"] = None
    mix: float = Field(default=None, ge=0, le=1)
    timeout_s: float = Field(default=None, ge=0.2, le=60)
    dmx_enabled: bool = None
    dmx_artnet_universe: int = Field(default=None, ge=0, le=32767)
    dmx_sacn_universe: int = Field(default=None, ge=1, le=63999)
    dmx_address: int = Field(default=None, ge=1, le=497)


class ExternalInfo(BaseModel):
    settings: ExternalSettings
    status: dict[str, Any] = Field(
        description="live, applied source, frames, fps, age_s, protocol, sender, universes, listening, ports, errors, packets, polls, dmx {live, packets, values}."
    )


def external_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/external", tags=["external"])

    def external():
        if ctx.external is None:
            raise HTTPException(503, "external input is not running")
        return ctx.external

    def info() -> ExternalInfo:
        ext = external()
        return ExternalInfo(settings=ExternalSettings(**ext.settings()), status=ext.status())

    @router.get("", response_model=ExternalInfo)
    def get_external() -> ExternalInfo:
        return info()

    @router.patch("", response_model=ExternalInfo, responses={422: {"description": "A value, or the combination, is invalid"}})
    def update_external(body: ExternalChanges) -> ExternalInfo:
        """Validated as a whole before anything changes; applied now, and stored."""
        try:
            external().update(**body.model_dump(exclude_unset=True))
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return info()

    return router
