"""The beat sync API.

    GET   /api/beat          settings + live status
    PATCH /api/beat          change settings; applied now and stored
    POST  /api/beat/tap      a tap, for tap tempo
    POST  /api/beat/resync   the next beat is a downbeat
    POST  /api/beat/nudge    {"ms": 10}: shift the beat later on the floor (negative: earlier)

With beat sync not running (a test app) every route answers 503.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from df2_pi.interfacing.beat_service import BeatService
    from df2_pi.web.app import AppContext

Source = Literal["off", "link", "tap"]
Quantum = Literal["off", "beat", "bar"]
Multiplier = Literal[0.5, 1.0, 2.0]


class BeatSettings(BaseModel):
    source: Source = Field(description="off; link: Ableton Link on the local network; tap: tap tempo.")
    beats_per_bar: int = Field(description="Bar length; also Ableton Link's quantum.")
    multiplier: Multiplier = Field(description="1/2x, 1x or 2x the source's tempo.")
    offset_ms: float = Field(description="Read the music this much later than each frame's latch: the rest of the way to the LEDs.")
    launch_quantum: Quantum = Field(description="goto/next/previous/loads/one-offs wait for the next beat or bar line.")


class BeatChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source = None
    beats_per_bar: int = Field(default=None, ge=1, le=16)
    multiplier: Multiplier = None
    offset_ms: float = Field(default=None, ge=-500, le=500)
    launch_quantum: Quantum = None


class BeatInfoOut(BaseModel):
    settings: BeatSettings
    status: dict[str, Any] = Field(description="active, tempo, beat, phase, bar_phase, bar_known; peers for Link, taps for tap.")


class Nudge(BaseModel):
    ms: float = Field(ge=-1000, le=1000, allow_inf_nan=False)


def beat_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/beat", tags=["beat"])

    def service() -> BeatService:
        if ctx.beat is None:
            raise HTTPException(503, "beat sync is not running")
        return ctx.beat

    def info(beat: BeatService) -> BeatInfoOut:
        return BeatInfoOut(settings=BeatSettings(**beat.settings()), status=beat.status())

    @router.get("", response_model=BeatInfoOut)
    def get() -> BeatInfoOut:
        return info(service())

    @router.patch("", response_model=BeatInfoOut, responses={422: {"description": "A value is out of range"}})
    def patch(body: BeatChanges) -> BeatInfoOut:
        beat = service()
        try:
            beat.update(**body.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return info(beat)

    @router.post("/tap", response_model=BeatInfoOut)
    def tap() -> BeatInfoOut:
        beat = service()
        beat.tap()
        return info(beat)

    @router.post("/resync", response_model=BeatInfoOut)
    def resync() -> BeatInfoOut:
        beat = service()
        beat.resync()
        return info(beat)

    @router.post("/nudge", response_model=BeatInfoOut)
    def nudge(body: Nudge) -> BeatInfoOut:
        beat = service()
        beat.nudge(body.ms)
        return info(beat)

    return router
