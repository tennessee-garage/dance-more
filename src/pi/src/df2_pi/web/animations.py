"""The animations API: what the registry knows, served to the page.

    GET  /api/animations          every animation, and every file that failed
    GET  /api/animations/{id}     one; 404 carrying the load error if it failed
    POST /api/animations/reload   re-import what changed on disk

`params` are the `Param` specs with the type as a name (int / float / str /
bool): what the live controls (#93) are built from and what the playlist
editor (#90) validates overrides against before the server does. `extra`
is served verbatim. Paths are relative to their animations directory; an
absolute path never leaves the Pi.

An id can be in `animations` and `errors` at once: a reload that fails
keeps the last good version live, and the error says why the edit did not
take.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from df2_pi.animation.loader import AnimationDef, LoadError

if TYPE_CHECKING:
    from df2_pi.web.app import AppContext


class ParamSpec(BaseModel):
    type: str = Field(description='The Python type as a name: "int", "float", "str" or "bool".')
    default: Any
    min: Any = None
    max: Any = None
    choices: list[Any] | None = None
    label: str | None = None
    help: str | None = None


class LoadErrorInfo(BaseModel):
    stage: str = Field(description='"syntax", "import" (the module body raised) or "validate".')
    message: str
    path: str = Field(description="Relative to the animations directory.")


class AnimationInfo(BaseModel):
    id: str
    name: str
    description: str
    author: str
    format: str = Field(description='"tile" or "pixel".')
    tags: list[str]
    period: float | None = Field(description="Seconds; advisory, never read by the runner.")
    params: dict[str, ParamSpec]
    extra: dict[str, Any] = Field(description="Every @animation keyword the decorator did not recognise, verbatim.")
    path: str = Field(description="Relative to the animations directory.")
    error: LoadErrorInfo | None = Field(
        default=None, description="Set when a reload failed and this last good version is still the live one."
    )


class AnimationList(BaseModel):
    animations: list[AnimationInfo]
    errors: dict[str, LoadErrorInfo]


class Reloaded(BaseModel):
    changed: list[str] = Field(description="Ids loaded, reloaded, newly failing or removed.")
    errors: dict[str, LoadErrorInfo]


def _jsonable(value: Any) -> Any:
    """`value` as JSON can carry it; anything JSON cannot (a set, an
    object) becomes its repr rather than failing the whole response."""
    try:
        return json.loads(json.dumps(value, default=repr))
    except (TypeError, ValueError):
        return repr(value)


def animations_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/animations", tags=["animations"])
    registry = ctx.registry

    def relative(path: Path) -> str:
        for directory in registry.paths:
            try:
                return path.resolve().relative_to(directory.resolve()).as_posix()
            except ValueError:
                continue
        return path.name

    def scrub(message: str) -> str:
        """Drop the animations directories from a message: the duplicate-id
        error names the file already loaded by its absolute path."""
        for directory in registry.paths:
            for form in {str(directory), str(directory.resolve())}:
                message = message.replace(form + os.sep, "")
        return message

    def error_info(error: LoadError) -> LoadErrorInfo:
        return LoadErrorInfo(stage=error.stage, message=scrub(error.message), path=relative(error.path))

    def info(definition: AnimationDef, errors: dict[str, LoadError]) -> AnimationInfo:
        meta = definition.meta
        error = errors.get(definition.id)
        return AnimationInfo(
            id=definition.id,
            name=meta.name,
            description=meta.description,
            author=meta.author,
            format=meta.format,
            tags=list(meta.tags),
            period=meta.period,
            params={
                key: ParamSpec(
                    type=spec.type.__name__,
                    default=_jsonable(spec.default),
                    min=_jsonable(spec.min),
                    max=_jsonable(spec.max),
                    choices=None if spec.choices is None else [_jsonable(c) for c in spec.choices],
                    label=spec.label,
                    help=spec.help,
                )
                for key, spec in meta.params.items()
            },
            extra={str(k): _jsonable(v) for k, v in meta.extra.items()},
            path=relative(definition.path),
            error=error_info(error) if error is not None else None,
        )

    @router.get("", response_model=AnimationList)
    def list_animations() -> AnimationList:
        """Every loaded animation, by name, and every file that failed."""
        # One read of each attribute: reload() replaces them, never mutates.
        animations, errors = registry.animations, registry.errors
        return AnimationList(
            animations=[info(d, errors) for d in sorted(animations.values(), key=lambda d: (d.meta.name.lower(), d.id))],
            errors={animation_id: error_info(e) for animation_id, e in sorted(errors.items())},
        )

    @router.get(
        "/{animation_id}",
        response_model=AnimationInfo,
        responses={404: {"description": "No such animation; if its file failed to load, `error` says why."}},
    )
    def get_animation(animation_id: str):
        animations, errors = registry.animations, registry.errors
        definition = animations.get(animation_id)
        if definition is not None:
            return info(definition, errors)
        error = errors.get(animation_id)
        if error is None:
            return JSONResponse({"detail": f"no animation {animation_id!r}"}, status_code=404)
        return JSONResponse(
            {"detail": f"{animation_id!r} failed to load", "error": error_info(error).model_dump()},
            status_code=404,
        )

    @router.post("/reload", response_model=Reloaded)
    def reload() -> Reloaded:
        """Re-import files whose mtime changed, pick up new ones and drop
        deleted ones. A file that now fails keeps its last good version live.
        Playlists already loaded keep the definitions they resolved; reload
        the playlist (`POST /api/transport/load`) to play an edit."""
        changed = registry.reload()
        return Reloaded(changed=changed, errors={k: error_info(e) for k, e in sorted(registry.errors.items())})

    return router
