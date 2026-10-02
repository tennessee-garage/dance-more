"""The palette API (#128).

    GET    /api/palettes            every palette, and which is active
    POST   /api/palettes/active     {"name": ...}: make one the floor's palette
    PUT    /api/palettes/{name}     {"stops": ["ff0000", ...]}: create or replace a user palette
    DELETE /api/palettes/{name}     remove a user palette

Built-ins can be made active but not changed or deleted. With no palette
book (a test app) every route answers 503.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from df2_pi.palette import PaletteBook
    from df2_pi.web.app import AppContext


class PaletteOut(BaseModel):
    name: str
    stops: list[str] = Field(description="Hex colours, 2..8, spaced evenly round a loop.")
    builtin: bool


class PalettesOut(BaseModel):
    active: str
    palettes: list[PaletteOut] = Field(description="Built-ins in library order, then the user's: the order DMX channel 18 counts.")


class Activate(BaseModel):
    name: str


class Stops(BaseModel):
    stops: list[str] = Field(min_length=2, max_length=8, description="Hex colours like 'ff8000'.")


def palettes_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/palettes", tags=["palettes"])

    def book() -> PaletteBook:
        if ctx.palettes is None:
            raise HTTPException(503, "no palette book")
        return ctx.palettes

    def listing(b: PaletteBook) -> PalettesOut:
        return PalettesOut(active=b.active, palettes=[PaletteOut(**e) for e in b.entries()])

    @router.get("", response_model=PalettesOut)
    def get() -> PalettesOut:
        return listing(book())

    @router.post("/active", response_model=PalettesOut, responses={404: {"description": "No such palette"}})
    def activate(body: Activate) -> PalettesOut:
        b = book()
        try:
            b.activate(body.name)
        except KeyError:
            raise HTTPException(404, f"no palette {body.name!r}") from None
        return listing(b)

    @router.put("/{name}", response_model=PalettesOut, responses={422: {"description": "A bad name, a built-in's name, or bad stops"}})
    def save(name: str, body: Stops) -> PalettesOut:
        b = book()
        try:
            b.save(name, body.stops)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return listing(b)

    @router.delete("/{name}", response_model=PalettesOut, responses={404: {"description": "No such palette"}, 422: {"description": "A built-in"}})
    def delete(name: str) -> PalettesOut:
        b = book()
        try:
            b.delete(name)
        except KeyError:
            raise HTTPException(404, f"no palette {name!r}") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return listing(b)

    return router
