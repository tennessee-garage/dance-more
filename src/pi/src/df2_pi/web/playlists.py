"""The playlists API: full CRUD over the PlaylistStore, and the settings.

    GET/POST        /api/playlists
    GET/PATCH/DELETE /api/playlists/{id}
    POST            /api/playlists/{id}/entries
    PATCH/DELETE    /api/playlists/{id}/entries/{entry_id}
    POST            /api/playlists/{id}/entries/{entry_id}/move
    POST            /api/playlists/{id}/startup
    GET/PATCH       /api/settings     brightness, rotation, strobe_max_hz, default_entry_duration, startup_playlist

A playlist is always served RESOLVED: each entry with its animation's name,
format and period when the registry knows it, `unresolved` and the reason
when it does not - never dropped - and the full params it would play with,
beside the overrides actually stored (only the diffs from the defaults).
Every write answers with the whole playlist as it now stands.

The runner holds its own resolved copy of whatever it loaded, so editing
the loaded playlist changes the database only; such a response carries
`"loaded": true`, and `POST /api/transport/load` puts the edit on the floor.

Store errors map to status codes: no such playlist or entry 404, a name
already taken 409, anything else the store refuses 422 with its message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from df2_pi.engine.overlays import STROBE_MAX_HZ_LIMIT
from df2_pi.playlists.store import DuplicatePlaylistName, Playlist, ResolvedEntry

if TYPE_CHECKING:
    from df2_pi.web.app import AppContext


class AnimationRef(BaseModel):
    name: str
    format: str
    period: float | None


class EntryInfo(BaseModel):
    id: int
    position: int
    animation_id: str
    duration_s: float
    enabled: bool
    params: dict[str, Any] = Field(description="The stored overrides: only what differs from the animation's defaults.")
    resolved_params: dict[str, Any] = Field(description="What it plays with: the defaults plus the overrides, clamped to the specs.")
    warnings: list[str] = Field(description="Overrides clamped or ignored because the specs changed under them.")
    animation: AnimationRef | None = Field(description="Null when the animation is missing or failed to load.")
    unresolved: bool
    error: str | None = Field(description="Why it is unresolved.")
    playable: bool = Field(description="Enabled and resolved: the runner plays it.")


class PlaylistSummary(BaseModel):
    id: int
    name: str
    description: str
    loop: bool
    shuffle: bool
    crossfade_s: float
    entry_count: int
    total_duration_s: float
    startup: bool = Field(description="Loaded when the floor starts.")
    loaded: bool = Field(description="The playlist the runner has loaded now.")


class PlaylistDetail(PlaylistSummary):
    created_at: str
    updated_at: str
    entries: list[EntryInfo]


class NewPlaylist(BaseModel):
    name: str
    description: str = ""
    loop: bool = True
    shuffle: bool = False
    crossfade_s: float = Field(default=0.0, ge=0)


class PlaylistChanges(BaseModel):
    name: str | None = None
    description: str | None = None
    loop: bool | None = None
    shuffle: bool | None = None
    crossfade_s: float | None = Field(default=None, ge=0)


class NewEntry(BaseModel):
    animation_id: str
    duration_s: float | None = Field(default=None, gt=0, description="Omitted: the default_entry_duration setting.")
    params: dict[str, Any] = Field(default_factory=dict)
    position: int | None = Field(default=None, ge=0, description="Omitted: appended.")
    enabled: bool = True


class EntryChanges(BaseModel):
    animation_id: str | None = None
    duration_s: float | None = Field(default=None, gt=0)
    params: dict[str, Any] | None = Field(
        default=None, description="Replaces the overrides wholesale. Changing animation_id without it clears them."
    )
    enabled: bool | None = None


class Move(BaseModel):
    position: int = Field(ge=0)


Rotation = Literal[0, 90, 180, 270]


class Settings(BaseModel):
    brightness: int = Field(ge=0, le=255, description="What the floor starts at; changing it also applies it now.")
    rotation: Rotation = Field(description="Degrees clockwise the picture is turned; changing it also applies it now.")
    strobe_max_hz: float = Field(description="The strobe show control's cap; changing it also applies it now.")
    default_entry_duration: float = Field(gt=0, description="Seconds, for an entry added without a duration.")
    startup_playlist: int | None = Field(description="Loaded when the floor starts.")


class SettingsChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a setting that is not here is refused, not ignored

    # Omitted means unchanged; null is refused except where clearing means something.
    brightness: int = Field(default=None, ge=0, le=255)
    rotation: Rotation = Field(default=None)
    strobe_max_hz: float = Field(default=None, ge=0, le=STROBE_MAX_HZ_LIMIT)
    default_entry_duration: float = Field(default=None, gt=0)
    startup_playlist: int | None = Field(default=None, description="A playlist id; send null explicitly to clear it.")


def playlists_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["playlists"])

    def store():
        if ctx.store is None:
            raise HTTPException(503, "no playlist store")
        return ctx.store

    def loaded_id() -> int | None:
        playing = ctx.runner.state.playlist
        return playing[0] if playing else None

    def startup_id() -> int | None:
        return store().get_int("startup_playlist")

    def not_found(exc: KeyError) -> HTTPException:
        return HTTPException(404, str(exc.args[0]) if exc.args else "not found")

    def refused(exc: ValueError | TypeError) -> HTTPException:
        return HTTPException(409 if isinstance(exc, DuplicatePlaylistName) else 422, str(exc))

    def summary_fields(pl: Playlist) -> dict[str, Any]:
        return dict(
            id=pl.id,
            name=pl.name,
            description=pl.description,
            loop=pl.loop,
            shuffle=pl.shuffle,
            crossfade_s=pl.crossfade_s,
            entry_count=len(pl.entries),
            total_duration_s=sum(e.duration_s for e in pl.entries),
            startup=pl.id == startup_id(),
            loaded=pl.id == loaded_id(),
        )

    def entry_info(resolved: ResolvedEntry) -> EntryInfo:
        entry, definition = resolved.entry, resolved.definition
        return EntryInfo(
            id=entry.id,
            position=entry.position,
            animation_id=entry.animation_id,
            duration_s=entry.duration_s,
            enabled=entry.enabled,
            params=dict(entry.params),
            resolved_params=dict(resolved.params),
            warnings=list(resolved.warnings),
            animation=None
            if definition is None
            else AnimationRef(name=definition.meta.name, format=definition.meta.format, period=definition.meta.period),
            unresolved=resolved.unresolved,
            error=resolved.error,
            playable=resolved.playable,
        )

    def detail(pid: int) -> PlaylistDetail:
        try:
            resolved = store().resolve(pid, ctx.registry)
        except KeyError as exc:
            raise not_found(exc) from exc
        pl = resolved.playlist
        return PlaylistDetail(
            **summary_fields(pl),
            created_at=pl.created_at,
            updated_at=pl.updated_at,
            entries=[entry_info(e) for e in resolved.entries],
        )

    def owned_entry(pid: int, entry_id: int):
        """The entry, if it belongs to playlist `pid`; 404 otherwise."""
        try:
            store().playlist(pid)
            entry = store().entry(entry_id)
        except KeyError as exc:
            raise not_found(exc) from exc
        if entry.playlist_id != pid:
            raise HTTPException(404, f"no entry {entry_id} in playlist {pid}")
        return entry

    # ---- playlists -----------------------------------------------------------------------

    @router.get("/playlists", response_model=list[PlaylistSummary])
    def list_playlists() -> list[PlaylistSummary]:
        """Every playlist, by name."""
        return [PlaylistSummary(**summary_fields(pl)) for pl in store().playlists()]

    @router.post("/playlists", response_model=PlaylistDetail, status_code=201)
    def create_playlist(body: NewPlaylist) -> PlaylistDetail:
        try:
            pl = store().create_playlist(
                body.name, description=body.description, loop=body.loop, shuffle=body.shuffle, crossfade_s=body.crossfade_s
            )
        except (ValueError, TypeError) as exc:
            raise refused(exc) from exc
        return detail(pl.id)

    @router.get("/playlists/{pid}", response_model=PlaylistDetail)
    def get_playlist(pid: int) -> PlaylistDetail:
        """One playlist, entries resolved against the animations."""
        return detail(pid)

    @router.patch("/playlists/{pid}", response_model=PlaylistDetail)
    def update_playlist(pid: int, body: PlaylistChanges) -> PlaylistDetail:
        try:
            store().update_playlist(pid, **body.model_dump(exclude_unset=True))
        except KeyError as exc:
            raise not_found(exc) from exc
        except (ValueError, TypeError) as exc:
            raise refused(exc) from exc
        return detail(pid)

    @router.delete("/playlists/{pid}", status_code=204)
    def delete_playlist(pid: int) -> Response:
        """Delete it and its entries. If it was the startup playlist, there
        is no startup playlist any more; if it is loaded, the runner plays
        on from its own copy."""
        was_startup = pid == startup_id()
        try:
            store().delete_playlist(pid)
        except KeyError as exc:
            raise not_found(exc) from exc
        if was_startup:
            store().set_setting("startup_playlist", None)
        return Response(status_code=204)

    @router.post("/playlists/{pid}/startup", response_model=PlaylistDetail)
    def make_startup(pid: int) -> PlaylistDetail:
        """Make this the playlist loaded when the floor starts."""
        response = detail(pid)  # 404 first
        store().set_setting("startup_playlist", pid)
        response.startup = True
        return response

    # ---- entries -------------------------------------------------------------------------

    @router.post("/playlists/{pid}/entries", response_model=PlaylistDetail, status_code=201)
    def add_entry(pid: int, body: NewEntry) -> PlaylistDetail:
        try:
            store().add_entry(
                pid, body.animation_id, duration_s=body.duration_s, params=body.params, position=body.position, enabled=body.enabled
            )
        except KeyError as exc:
            raise not_found(exc) from exc
        except (ValueError, TypeError) as exc:
            raise refused(exc) from exc
        return detail(pid)

    @router.patch("/playlists/{pid}/entries/{entry_id}", response_model=PlaylistDetail)
    def update_entry(pid: int, entry_id: int, body: EntryChanges) -> PlaylistDetail:
        entry = owned_entry(pid, entry_id)
        changes = body.model_dump(exclude_unset=True)
        if "animation_id" in changes and changes["animation_id"] != entry.animation_id and "params" not in changes:
            changes["params"] = {}  # the old animation's overrides mean nothing to the new one
        try:
            store().update_entry(entry_id, **changes)
        except (ValueError, TypeError) as exc:
            raise refused(exc) from exc
        return detail(pid)

    @router.delete("/playlists/{pid}/entries/{entry_id}", response_model=PlaylistDetail)
    def remove_entry(pid: int, entry_id: int) -> PlaylistDetail:
        owned_entry(pid, entry_id)
        store().remove_entry(entry_id)
        return detail(pid)

    @router.post("/playlists/{pid}/entries/{entry_id}/move", response_model=PlaylistDetail)
    def move_entry(pid: int, entry_id: int, body: Move) -> PlaylistDetail:
        """To `position` (clamped to the end); the rest close up around it."""
        owned_entry(pid, entry_id)
        store().move_entry(entry_id, body.position)
        return detail(pid)

    # ---- settings ------------------------------------------------------------------------

    def settings() -> Settings:
        s = store()
        return Settings(
            brightness=s.get_int("brightness"),
            rotation=s.get_rotation(),
            strobe_max_hz=s.get_strobe_max_hz(),
            default_entry_duration=s.get_float("default_entry_duration"),
            startup_playlist=s.get_int("startup_playlist"),
        )

    @router.get("/settings", response_model=Settings, tags=["settings"])
    def get_settings() -> Settings:
        return settings()

    @router.patch("/settings", response_model=Settings, tags=["settings"])
    def update_settings(body: SettingsChanges) -> Settings:
        """Validated before anything is written: all of it lands, or none.
        A new brightness, rotation or strobe cap is applied to the floor
        now as well as stored."""
        changes = body.model_dump(exclude_unset=True)
        if changes.get("startup_playlist") is not None:
            try:
                store().playlist(changes["startup_playlist"])
            except KeyError as exc:
                raise HTTPException(422, f"no playlist {changes['startup_playlist']}") from exc
        for key, value in changes.items():
            store().set_setting(key, value)
        if "brightness" in changes:
            ctx.runner.set_brightness(changes["brightness"])
        if "rotation" in changes:
            ctx.runner.set_rotation(changes["rotation"])
        if "strobe_max_hz" in changes:
            ctx.runner.set_strobe_max(changes["strobe_max_hz"])
        return settings()

    return router
