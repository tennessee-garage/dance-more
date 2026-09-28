"""The FastAPI app: the runner's lifecycle, the page shell, `/healthz`.

    ctx = AppContext(registry, store, fanout, runner, preview)
    app = create_app(ctx)                  # nothing starts until the lifespan runs
    uvicorn.run(app, host="0.0.0.0", port=8000)

The runner lives on a daemon thread of this process. Its control API is
queued and thread-safe, so a request handler puts a command on the queue
and reads the state snapshot; there is no IPC and no lock here.

Lifespan. Startup loads the startup playlist exactly as `df2-pi play`
does and starts the render thread. Shutdown - a deploy restart, Ctrl-C,
SIGTERM from systemd - is `stop()` then `join()`, so the runner blacks the
floor out and latches before the process exits rather than leaving the
tiles frozen on a stale frame. uvicorn owns the signals; the runner's own
handlers are never installed.

Everything the app touches arrives in `AppContext`, never module globals,
so tests build one against a fake clock and a `NullSink`, and the CLI builds
one against the floor.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from df2_pi.web.animations import animations_router
from df2_pi.web.playlists import playlists_router
from df2_pi.web.preview import preview_router
from df2_pi.web.transport import mark_percentiles_nullable, transport_router

if TYPE_CHECKING:
    from df2_pi.animation import AnimationRegistry
    from df2_pi.engine import Runner
    from df2_pi.output import FanOut, PreviewSink
    from df2_pi.playlists import PlaylistStore

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SHUTDOWN_TIMEOUT_S = 5.0  # the final blackout and latch take a frame or two


@dataclass
class AppContext:
    """What the app drives. `store` is None only for a run with no
    playlist database (tests); the idle animation then plays."""

    registry: AnimationRegistry
    store: PlaylistStore | None
    fanout: FanOut
    runner: Runner
    preview: PreviewSink


class _RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate on every load. A deploy
    restarts the server with new `app.js`; without this a browser can run
    last week's copy from its heuristic cache. The ETag makes an unchanged
    file a cheap 304."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def load_startup_playlist(ctx: AppContext) -> None:
    """Queue the startup playlist on the runner, seeding a Default one into
    an empty database - the same start `df2-pi play` makes."""
    if ctx.store is None:
        return
    if ctx.store.seed_default(ctx.registry) is not None:
        log.info("seeded a Default playlist")
    startup = ctx.store.startup_playlist()
    ctx.runner.load_playlist(ctx.store.resolve(startup, ctx.registry) if startup is not None else None)


def create_app(ctx: AppContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_startup_playlist(ctx)
        ctx.runner.start()
        try:
            yield
        finally:
            ctx.runner.stop()
            ctx.runner.join(SHUTDOWN_TIMEOUT_S)
            if ctx.runner.alive:
                log.error("render thread still running %.0f s after stop()", SHUTDOWN_TIMEOUT_S)

    app = FastAPI(
        title="Dance Floor",
        description=(
            "Control API for the Dance Floor v2 runner. Transport commands are queued and "
            "applied at the next frame boundary, so the `state` a POST returns may not "
            "reflect the command yet: re-poll `GET /api/state`."
        ),
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    app.include_router(transport_router(ctx))
    app.include_router(animations_router(ctx))
    app.include_router(playlists_router(ctx))
    app.include_router(preview_router(ctx))

    default_openapi = app.openapi

    def openapi() -> dict:
        if app.openapi_schema is None:
            mark_percentiles_nullable(default_openapi())  # builds and caches app.openapi_schema
        return app.openapi_schema

    app.openapi = openapi

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        """200 while the render thread is alive, 503 once it has exited.
        What systemd and a load balancer probe."""
        alive = ctx.runner.alive
        body = {"ok": alive, "runner": "alive" if alive else "stopped", "frame": ctx.runner.state.frame}
        return JSONResponse(body, status_code=200 if alive else 503)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", _RevalidatedStaticFiles(directory=STATIC_DIR), name="static")
    return app
