"""The MIDI input's settings, mapping and learn.

    GET    /api/midi                  status (ports, counts, recent messages, learning) and the mapping
    PATCH  /api/midi                  {"enabled": bool}: applied now (ports reopened) and stored
    PATCH  /api/midi/map              {"program_change": bool, "clock": bool}
    POST   /api/midi/learn            {"to": "/floor/...", "toggle": bool, "value": f, "this_port_only": bool}:
                                      the next note or CC moved becomes this binding
    DELETE /api/midi/learn            stop waiting
    DELETE /api/midi/bindings/{i}     remove binding i
    POST   /api/midi/default          replace the mapping with the APC mini default

Every change to the mapping is written to its YAML file at once. With no
MIDI input (a test app) every route answers 503.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from df2_pi.interfacing.midi import MidiControl
    from df2_pi.web.app import AppContext


class MidiInfo(BaseModel):
    status: dict[str, Any] = Field(description="enabled, ports, port_errors, received, clock_received, errors, recent, learning, map_path, map_error")
    map: dict[str, Any] = Field(description="program_change, clock, and bindings: each {note|cc, to, channel?, port?, toggle?, value?, control}")


class Enabled(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


class MapOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    program_change: bool = None
    clock: bool = None


class Learn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: str = Field(description="A /floor/... address, as for OSC.")
    toggle: bool = Field(default=False, description="Each press flips it on or off.")
    value: float | None = Field(default=None, description="Send this on a press instead of the control's value.")
    this_port_only: bool = Field(default=False, description="Bind only for the port the control is on.")


def midi_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/midi", tags=["midi"])

    def midi() -> MidiControl:
        if ctx.midi is None:
            raise HTTPException(503, "MIDI is not running")
        return ctx.midi

    def info(m: MidiControl) -> MidiInfo:
        return MidiInfo(status=m.status(), map=m.mapping())

    @router.get("", response_model=MidiInfo)
    def get() -> MidiInfo:
        return info(midi())

    @router.patch("", response_model=MidiInfo)
    def enable(body: Enabled) -> MidiInfo:
        m = midi()
        m.configure(enabled=body.enabled)
        return info(m)

    @router.patch("/map", response_model=MidiInfo)
    def options(body: MapOptions) -> MidiInfo:
        m = midi()
        m.set_options(**body.model_dump(exclude_unset=True))
        return info(m)

    @router.post("/learn", response_model=MidiInfo, responses={422: {"description": "Not a /floor/... address"}})
    def learn(body: Learn) -> MidiInfo:
        m = midi()
        try:
            m.learn(body.to, toggle=body.toggle, value=body.value, this_port_only=body.this_port_only)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return info(m)

    @router.delete("/learn", response_model=MidiInfo)
    def cancel() -> MidiInfo:
        m = midi()
        m.cancel_learn()
        return info(m)

    @router.delete("/bindings/{index}", response_model=MidiInfo, responses={404: {"description": "No such binding"}})
    def remove(index: int) -> MidiInfo:
        m = midi()
        try:
            m.remove_binding(index)
        except IndexError:
            raise HTTPException(404, f"no binding {index}") from None
        return info(m)

    @router.post("/default", response_model=MidiInfo)
    def default() -> MidiInfo:
        m = midi()
        m.reset_to_default()
        return info(m)

    return router
