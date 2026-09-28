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

from typing import TYPE_CHECKING, Any, Callable, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from df2_pi.engine import RunnerState
from df2_pi.engine.overlays import DECAY_MAX_S, SATURATION_MAX, SPEED_MAX

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


# ---- show controls ----


class Speed(BaseModel):
    value: float = Field(ge=0, le=SPEED_MAX, description="Scales the time animations see; 1 is as written.")


class Freeze(BaseModel):
    on: bool


class Bump(BaseModel):
    level: float = Field(default=1.0, ge=0, le=1, description="How far toward white.")
    decay_s: float = Field(default=0.25, ge=0.01, le=DECAY_MAX_S, description="Seconds to fade back.")


class Strobe(BaseModel):
    rate_hz: float = Field(ge=0, description="0 is off; held to the strobe_max_hz setting.")


class Tint(BaseModel):
    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)
    amount: float = Field(ge=0, le=1, description="0 is off; 1 is fully the tint's colour.")


class HueShift(BaseModel):
    value: float = Field(description="Turns; 1.0 is all the way round, wrapped.", allow_inf_nan=False)


BlendMode = Literal["add", "max", "multiply", "mix"]  # pixels.BLEND_MODES


class Layer(BaseModel):
    id: str
    params: dict[str, Any] = Field(default_factory=dict, description="Overrides, coerced through the animation's Param specs.")
    mode: BlendMode = Field(default="add", description="add: light on light; max: the brighter; multiply: a mask; mix: replace.")
    amount: float = Field(default=1.0, ge=0, le=1, description="0 is the base untouched; 1 is the full blend.")


class LayerBlend(BaseModel):
    mode: BlendMode | None = None
    amount: float | None = Field(default=None, ge=0, le=1)


class Saturation(BaseModel):
    value: float = Field(ge=0, le=SATURATION_MAX, description="0 is grey; 1 is unchanged.")


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

    # ---- show controls: act on whatever is playing; not stored ----

    @router.post("/transport/speed", response_model=Queued, tags=["show"], description=ASYNC_NOTE)
    def speed(body: Speed) -> Queued:
        """Scale the time animations see, 0..4."""
        return queued(lambda: runner.set_speed(body.value))

    @router.post("/transport/freeze", response_model=Queued, tags=["show"], description=ASYNC_NOTE)
    def freeze(body: Freeze) -> Queued:
        """Hold the picture while animations keep running underneath, or let it go."""
        return queued(lambda: runner.freeze(body.on))

    @router.post("/transport/bump", response_model=Queued, tags=["show"], description=ASYNC_NOTE)
    def bump(body: Bump = Bump()) -> Queued:
        """A flash toward white, fading back. At level 1 it is a full-white frame."""
        return queued(lambda: runner.bump(body.level, body.decay_s))

    @router.post("/transport/strobe", response_model=Queued, tags=["show"], description=ASYNC_NOTE)
    def strobe(body: Strobe) -> Queued:
        """Shutter the picture at `rate_hz`; 0 is off."""
        return queued(lambda: runner.set_strobe(body.rate_hz))

    @router.post("/transport/tint", response_model=Queued, tags=["show"], description=ASYNC_NOTE)
    def tint(body: Tint) -> Queued:
        """Colourise toward a colour by `amount`; black stays black."""
        return queued(lambda: runner.set_tint(body.r, body.g, body.b, body.amount))

    @router.post("/transport/hue_shift", response_model=Queued, tags=["show"], description=ASYNC_NOTE)
    def hue_shift(body: HueShift) -> Queued:
        """Rotate every hue, in turns."""
        return queued(lambda: runner.set_hue_shift(body.value))

    @router.post("/transport/saturation", response_model=Queued, tags=["show"], description=ASYNC_NOTE)
    def saturation(body: Saturation) -> Queued:
        """Scale saturation, 0..2."""
        return queued(lambda: runner.set_saturation(body.value))

    simple("reset_show", runner.reset_show, "Every show control back to where it does nothing")

    # ---- the layer: one animation over whatever plays ----

    @router.post(
        "/transport/layer",
        response_model=Queued,
        tags=["layer"],
        description=ASYNC_NOTE,
        responses={404: {"description": "No such animation"}, 422: {"description": "A param is unknown or out of range"}},
    )
    def set_layer(body: Layer) -> Queued:
        """Run an animation as a layer over whatever plays, replacing any layer."""
        definition = ctx.registry.get(body.id)
        if definition is None:
            raise HTTPException(404, f"no animation {body.id!r}")
        try:
            params = definition.meta.resolve_params(body.params)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return queued(lambda: runner.set_layer(body.id, params, body.mode, body.amount))

    @router.post("/transport/layer_blend", response_model=Queued, tags=["layer"], description=ASYNC_NOTE)
    def layer_blend(body: LayerBlend) -> Queued:
        """Change how the layer combines, without restarting it."""
        return queued(lambda: runner.set_layer_blend(body.mode, body.amount))

    simple("clear_layer", runner.clear_layer, "Remove the layer")

    return router
