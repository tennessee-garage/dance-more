"""The transport API: the runner's control API and its state over HTTP.

    GET  /api/state                   the RunnerState snapshot
    POST /api/transport/<command>     queue a command; returns {queued, state}

JSON in, JSON out; the admin page and the MIDI bridge (#71) both drive
this. Every body is validated here - index, brightness, animation id,
params coerced through the animation's `Param` specs - so invalid input
never reaches the runner.

COMMANDS ARE ASYNCHRONOUS. The runner applies queued commands at the next
frame boundary, so the state a POST returns may not reflect the command
yet. A client re-polls `/api/state`; it does not read the change from the
POST's response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from df2_pi.engine import RunnerState

if TYPE_CHECKING:
    from df2_pi.web.app import AppContext

ASYNC_NOTE = (
    "Queued, and applied by the runner at the next frame boundary: the `state` "
    "returned here may not reflect the command yet. Re-poll `GET /api/state`."
)


def mark_percentiles_nullable(schema: dict[str, Any]) -> None:
    """Telemetry percentiles are NaN until their window has a sample, and
    JSON has no NaN: they are sent as null. The dataclass says `float`, so
    patch the published schema to say what is actually on the wire."""
    percentiles = schema.get("components", {}).get("schemas", {}).get("Percentiles")
    if percentiles is None:
        return
    for name in ("p50", "p95", "max"):
        prop = percentiles["properties"][name]
        prop.pop("type", None)
        prop["anyOf"] = [{"type": "number"}, {"type": "null"}]
        prop["description"] = "Milliseconds; null while the window is empty."


class Queued(BaseModel):
    queued: bool = True
    state: RunnerState


class Goto(BaseModel):
    index: int = Field(ge=0, description="An index into the playlist's entries; an unplayable one moves on to the next playable.")


class Load(BaseModel):
    playlist: int | str = Field(description="A playlist id (a JSON number) or name (a JSON string).")


class PlayAnimation(BaseModel):
    id: str
    params: dict[str, Any] = Field(default_factory=dict, description="Overrides, coerced through the animation's Param specs.")
    hold: float | None = Field(default=None, gt=0, description="Seconds before returning to the playlist; null holds until `next`.")


class Brightness(BaseModel):
    value: int = Field(ge=0, le=255)


class Blackout(BaseModel):
    on: bool


def transport_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api")
    runner = ctx.runner

    def queued(command: Callable[[], None]) -> Queued:
        command()
        return Queued(state=runner.state)

    @router.get("/state", response_model=RunnerState, tags=["state"])
    def state() -> RunnerState:
        """The runner's snapshot, replaced every frame. Never carries pixels:
        those are the preview socket's."""
        return runner.state

    def simple(name: str, command: Callable[[], None], summary: str) -> None:
        router.add_api_route(
            f"/transport/{name}",
            lambda: queued(command),
            methods=["POST"],
            response_model=Queued,
            tags=["transport"],
            summary=summary,
            description=ASYNC_NOTE,
            name=name,
        )

    simple("play", runner.play, "Start, or restart from the top if the playlist had ended; resumes if paused")
    simple("pause", runner.pause, "Hold the current frame lit; nothing advances")
    simple("resume", runner.resume, "Resume after a pause")
    simple("next", runner.next, "Skip to the next entry, or end a one-off")
    simple("previous", runner.previous, "Back to the previous entry")
    simple("restart", runner.restart, "Restart the current entry from its first frame")

    @router.post("/transport/goto", response_model=Queued, tags=["transport"], description=ASYNC_NOTE)
    def goto(body: Goto) -> Queued:
        """Jump to a playlist entry."""
        return queued(lambda: runner.goto(body.index))

    @router.post(
        "/transport/load",
        response_model=Queued,
        tags=["transport"],
        description=ASYNC_NOTE,
        responses={404: {"description": "No such playlist"}},
    )
    def load(body: Load) -> Queued:
        """Swap playlists without stopping the clock."""
        if ctx.store is None:
            raise HTTPException(404, "no playlist store")
        try:
            resolved = ctx.store.resolve(body.playlist, ctx.registry)
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0]) if exc.args else "no such playlist") from exc
        return queued(lambda: runner.load_playlist(resolved))

    @router.post(
        "/transport/animation",
        response_model=Queued,
        tags=["transport"],
        description=ASYNC_NOTE,
        responses={404: {"description": "No such animation"}, 422: {"description": "A param is unknown or out of range"}},
    )
    def play_animation(body: PlayAnimation) -> Queued:
        """Play one animation as a one-off, then return to the playlist."""
        definition = ctx.registry.get(body.id)
        if definition is None:
            raise HTTPException(404, f"no animation {body.id!r}")
        try:
            params = definition.meta.resolve_params(body.params)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return queued(lambda: runner.play_animation(body.id, params, body.hold))

    @router.post("/transport/brightness", response_model=Queued, tags=["transport"], description=ASYNC_NOTE)
    def brightness(body: Brightness) -> Queued:
        """Global brightness, 0-255. Also stored as the brightness setting,
        so the floor comes back at it after a restart."""
        response = queued(lambda: runner.set_brightness(body.value))
        if ctx.store is not None:
            ctx.store.set_setting("brightness", body.value)
        return response

    @router.post(
        "/transport/params",
        response_model=Queued,
        tags=["transport"],
        description=ASYNC_NOTE,
        responses={
            409: {"description": "What is playing has no parameters to set (the built-in idle animation)"},
            422: {"description": 'A param is unknown or out of range: `detail` is `{"param": name, "message": why}`'},
        },
    )
    def set_params(body: dict[str, Any]) -> Queued:
        """Live-tune the running animation: `{"speed": 2.0}`. Coerced through
        its Param specs; keys not given keep their current value."""
        playing = runner.state.animation
        definition = ctx.registry.get(playing[0]) if playing else None
        if definition is None:
            raise HTTPException(409, "the running animation has no parameters to set")
        specs = definition.meta.params
        coerced: dict[str, Any] = {}
        for name, value in body.items():
            spec = specs.get(name)
            if spec is None:
                raise HTTPException(
                    422, {"param": name, "message": f"{definition.id} has no parameter {name!r}; it declares {sorted(specs)}"}
                )
            try:
                coerced[name] = spec.coerce(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(422, {"param": name, "message": str(exc)}) from exc
        return queued(lambda: runner.set_params(**coerced))

    @router.post("/transport/blackout", response_model=Queued, tags=["transport"], description=ASYNC_NOTE)
    def blackout(body: Blackout) -> Queued:
        """Black out the floor (playback continues underneath), or lift it."""
        return queued(runner.blackout if body.on else runner.unblackout)

    return router
