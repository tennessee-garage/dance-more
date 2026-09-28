"""External content on the floor: decoding a received frame, the built-in
External animation that shows it, and the source control that decides
when it is on.

    external = external_animation(receiver)       # an AnimationDef, id "external"
    registry.add_builtin(external)
    source = ExternalSource(runner, receiver, source="external", timeout_s=2.0)
    source.start()                                # watches the signal; takes over and lets go

Decoding. Channel values are frame bytes as they are: gamma-encoded like
everything an animation produces, which is what video content already is,
so a sender should leave its own output gamma at 1.0. `tile` and `grid`
are raster order from the TOP-LEFT of the canonical view - the corner
farthest from the Pi - because that is what a sender's grid fixture
produces; the flip to the floor's row order happens here. `grid` is
sampled at each LED's position, bilinearly, in linear light. `raw` is
chain order and needs no mapping.

The External animation renders the latest received frame and holds it
between arrivals, so it can sit in a playlist, run as a one-off or as the
layer like any other animation. It always produces a PixelFrame; the
encoder still sends a uniform tile as SET_COLOR.

The source control. `source` is the operator's choice:

    internal   Art-Net is ignored; the playlist plays
    external   External takes over (a one-off, `hold=None`) while there is signal
    mix        External runs as the layer in `mix` mode, `mix` its amount

When no frame has arrived for `timeout_s` the floor falls back to its own
show - the one-off ends and the playlist entry that was playing restarts,
or the layer is removed - and when signal returns the choice applies
again. A sender that crashes or sleeps never leaves the floor frozen or
dark. The controller only undoes what it did: an operator who starts a
different one-off or layer meanwhile is not overruled.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from df2_pi.animation.loader import AnimationDef
from df2_pi.animation.meta import AnimationMeta
from df2_pi.gamma import from_linear, to_linear
from df2_pi.geometry import FloorGeometry
from df2_pi.interfacing.packets import PIXELS_PER_UNIVERSE
from df2_pi.interfacing.receiver import ExternalFrame, Layout, Receiver
from df2_pi.pixels import PixelFrame, TileFrame

if TYPE_CHECKING:
    from df2_pi.engine.runner import Runner

log = logging.getLogger(__name__)

EXTERNAL_ID = "external"
SOURCES = ("internal", "external", "mix")
POLL_INTERVAL_S = 0.1


# ---- decoding ---------------------------------------------------------------------------


def pixels_of(frame: ExternalFrame) -> np.ndarray:
    """The frame's RGB pixels in order, (layout.pixels, 3) uint8."""
    per = PIXELS_PER_UNIVERSE * 3
    return frame.channels[:, :per].reshape(-1, 3)[: frame.layout.pixels]


def decode(frame: ExternalFrame, geometry: FloorGeometry) -> PixelFrame:
    """A received frame as floor pixels."""
    px = pixels_of(frame)
    layout = frame.layout
    if layout.mode == "raw":
        return PixelFrame(px.reshape(geometry.tiles, geometry.leds_per_tile, 3).copy(), geometry)
    if layout.mode == "tile":
        # Raster rows run top-down; canonical row 0 is at the bottom.
        tiles = px.reshape(geometry.tile_rows, geometry.tile_cols, 3)[::-1].copy()
        return TileFrame(tiles, geometry).to_pixels()
    image = px.reshape(layout.height, layout.width, 3)
    index, weights = _grid_sampling(geometry, layout.width, layout.height)
    lin = to_linear(image).reshape(-1, 3)[index]  # (leds, 4, 3): the four neighbours
    sampled = (lin * weights[..., None]).sum(axis=1)
    return PixelFrame(from_linear(sampled).reshape(geometry.tiles, geometry.leds_per_tile, 3), geometry)


@lru_cache(maxsize=8)
def _grid_sampling(geometry: FloorGeometry, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Per LED, the flat indices of the 4 image pixels around it and their
    bilinear weights - built once per grid size. The image's top row is the
    canonical view's top, the edge farthest from the Pi."""
    pos = geometry.led_positions.reshape(-1, 2).astype(np.float64)  # (y up, x) cell centres
    u = pos[:, 1] / geometry.width * width - 0.5
    v = (geometry.height - pos[:, 0]) / geometry.height * height - 0.5
    x0 = np.clip(np.floor(u), 0, width - 1).astype(np.intp)
    y0 = np.clip(np.floor(v), 0, height - 1).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = np.clip(u - x0, 0.0, 1.0)
    fy = np.clip(v - y0, 0.0, 1.0)
    index = np.stack([y0 * width + x0, y0 * width + x1, y1 * width + x0, y1 * width + x1], axis=1)
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1).astype(np.float32)
    return index, weights


# ---- the External animation -------------------------------------------------------------


def external_animation(receiver: Receiver) -> AnimationDef:
    """The built-in animation showing whatever `receiver` last completed."""
    cache: dict[str, object] = {"seq": None, "frame": None}

    def render(previous, ctx) -> PixelFrame:
        latest = receiver.latest()
        if latest is None:
            return PixelFrame.black(ctx.geometry)
        if cache["seq"] != latest.seq:
            cache["frame"] = decode(latest, ctx.geometry)
            cache["seq"] = latest.seq
        return cache["frame"].copy()  # never the frame handed back last time

    meta = AnimationMeta(
        name="External",
        description="Pixels from a media server over Art-Net or sACN (Resolume, TouchDesigner, ...).",
        author="df2",
        format="pixel",
        tags=("external",),
    )
    return AnimationDef(id=EXTERNAL_ID, path=Path("<built-in>"), meta=meta, render=render)


# ---- the source control -----------------------------------------------------------------


class ExternalSource:
    def __init__(
        self,
        runner: Runner,
        receiver: Receiver,
        *,
        source: str = "external",
        mix: float = 0.5,
        timeout_s: float = 2.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runner = runner
        self.receiver = receiver
        self._now = now
        self._lock = threading.Lock()
        self.source = _check_source(source)
        self.mix = _check_mix(mix)
        self.timeout_s = _check_timeout(timeout_s)
        self.applied = "internal"
        self._thread: threading.Thread | None = None
        self._running = False

    def live(self) -> bool:
        latest = self.receiver.latest()
        return latest is not None and self._now() - latest.received_at < self.timeout_s

    def set_source(self, source: str) -> None:
        self.source = _check_source(source)
        self.poll()

    def set_mix(self, amount: float) -> None:
        self.mix = _check_mix(amount)
        with self._lock:
            if self.applied == "mix":
                self.runner.set_layer_blend(amount=self.mix)

    def set_timeout(self, seconds: float) -> None:
        self.timeout_s = _check_timeout(seconds)

    def poll(self) -> None:
        """Bring the floor in line with the source and the signal."""
        with self._lock:
            wanted = self.source if self.live() else "internal"
            if wanted == self.applied:
                return
            log.info("external source: %s -> %s", self.applied, wanted)
            if self.applied == "external":
                self.runner.end_one_off(EXTERNAL_ID)
            elif self.applied == "mix":
                self.runner.clear_layer(EXTERNAL_ID)
            if wanted == "external":
                self.runner.play_animation(EXTERNAL_ID, hold=None)
            elif wanted == "mix":
                self.runner.set_layer(EXTERNAL_ID, mode="mix", amount=self.mix)
            self.applied = wanted

    def state(self) -> dict:
        return {
            "source": self.source,
            "mix": self.mix,
            "timeout_s": self.timeout_s,
            "live": self.live(),
            "applied": self.applied,
        }

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="external-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(2.0)

    def _run(self) -> None:
        while self._running:
            try:
                self.poll()
            except Exception:
                log.exception("external source poll failed")
            time.sleep(POLL_INTERVAL_S)


def _check_source(source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return source


def _check_mix(amount: float) -> float:
    amount = float(amount)
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"mix must be in 0..1, got {amount}")
    return amount


def _check_timeout(seconds: float) -> float:
    seconds = float(seconds)
    if not 0.2 <= seconds <= 60.0:
        raise ValueError(f"timeout must be 0.2..60 s, got {seconds}")
    return seconds


__all__ = [
    "EXTERNAL_ID",
    "SOURCES",
    "ExternalSource",
    "Layout",
    "decode",
    "external_animation",
]
