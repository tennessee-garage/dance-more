"""`ExternalInput`: the Art-Net / sACN receiver and the source control,
configured from the settings store and changed live.

    external = ExternalInput(runner, registry, store)   # registers the External built-in
    external.start()                                    # sockets open, the source control runs
    external.update(mode="grid", source="mix", mix=0.3) # validated, applied now, stored
    external.status()                                   # what the web UI shows
    external.stop()

It also owns the DMX control block (dmx_control.py): off until
`dmx_enabled`, then fed by the receiver from its own universe and start
address, released on the same timeout as the pixels.

Every setting lives in the `setting` table under the keys in `KEYS`, so a
restart comes back listening the way it was left. `update()` validates the
whole change before touching anything: all of it lands, or none.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from df2_pi.interfacing.dmx_control import WIDTH as DMX_WIDTH
from df2_pi.interfacing.dmx_control import DmxControl
from df2_pi.interfacing.external import ExternalSource, SOURCES, external_animation
from df2_pi.interfacing.receiver import MODES, Layout, Receiver

if TYPE_CHECKING:
    from df2_pi.animation import AnimationRegistry
    from df2_pi.engine.runner import Runner
    from df2_pi.playlists import PlaylistStore

log = logging.getLogger(__name__)

# update() field -> setting key
KEYS = {
    "artnet_enabled": "artnet_enabled",
    "sacn_enabled": "sacn_enabled",
    "mode": "external_mode",
    "grid_width": "external_grid_width",
    "grid_height": "external_grid_height",
    "artnet_universe": "artnet_universe",
    "sacn_universe": "sacn_universe",
    "source": "external_source",
    "mix": "external_mix",
    "timeout_s": "external_timeout_s",
    "dmx_enabled": "dmx_enabled",
    "dmx_artnet_universe": "dmx_artnet_universe",
    "dmx_sacn_universe": "dmx_sacn_universe",
    "dmx_address": "dmx_address",
}
DMX_KEYS = {"dmx_enabled", "dmx_artnet_universe", "dmx_sacn_universe", "dmx_address"}

ARTNET_UNIVERSES = 32768  # 15-bit port-addresses
SACN_UNIVERSE_MAX = 63999


class ExternalInput:
    def __init__(self, runner: Runner, registry: AnimationRegistry, store: PlaylistStore | None, **receiver_kwargs: Any) -> None:
        self.runner = runner
        self.store = store
        self._settings = self._load()
        s = self._settings
        self.receiver = Receiver(
            Layout(s["mode"], s["grid_width"], s["grid_height"]),
            artnet_universe=s["artnet_universe"],
            sacn_universe=s["sacn_universe"],
            artnet=s["artnet_enabled"],
            sacn=s["sacn_enabled"],
            **receiver_kwargs,
        )
        self.source = ExternalSource(runner, self.receiver, source=s["source"], mix=s["mix"], timeout_s=s["timeout_s"])
        registry.add_builtin(external_animation(self.receiver))
        self.dmx = DmxControl(runner, self.source, store, timeout_s=s["timeout_s"])
        self.receiver.on_control = self.dmx.handle
        self._configure_dmx(s)
        self._frame_times: deque[tuple[float, int]] = deque(maxlen=64)

    def start(self) -> None:
        self.receiver.start()
        self.source.start()
        self.dmx.start()

    def stop(self) -> None:
        self.dmx.stop()
        self.source.stop()
        self.receiver.stop()

    def _configure_dmx(self, s: dict[str, Any]) -> None:
        self.receiver.configure_control(
            artnet_universe=s["dmx_artnet_universe"],
            sacn_universe=s["dmx_sacn_universe"],
            address=s["dmx_address"],
            width=DMX_WIDTH,
            enabled=s["dmx_enabled"],
        )

    # ---- settings ---------------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        return dict(self._settings)

    def update(self, **changes: Any) -> dict[str, Any]:
        """Validate, apply now, and store. Raises ValueError, with nothing
        changed, if any value is bad."""
        unknown = set(changes) - set(KEYS)
        if unknown:
            raise ValueError(f"unknown external settings {sorted(unknown)}")
        merged = {**self._settings, **changes}
        _validate(merged)
        receiver_keys = {"artnet_enabled", "sacn_enabled", "mode", "grid_width", "grid_height", "artnet_universe", "sacn_universe"}
        if receiver_keys & set(changes):
            self.receiver.configure(
                layout=Layout(merged["mode"], merged["grid_width"], merged["grid_height"]),
                artnet_universe=merged["artnet_universe"],
                sacn_universe=merged["sacn_universe"],
                artnet=merged["artnet_enabled"],
                sacn=merged["sacn_enabled"],
            )
        if DMX_KEYS & set(changes):
            self._configure_dmx(merged)
        if "timeout_s" in changes:
            self.source.set_timeout(merged["timeout_s"])
            self.dmx.timeout_s = merged["timeout_s"]
        if "mix" in changes:
            self.source.set_mix(merged["mix"])
        if "source" in changes:
            self.source.set_source(merged["source"])
        self._settings = merged
        if self.store is not None:
            for field in changes:
                self.store.set_setting(KEYS[field], merged[field])
        return self.settings()

    def _load(self) -> dict[str, Any]:
        """The stored settings, falling back to the defaults for any value
        that no longer validates, so a bad row never stops the floor."""
        from df2_pi.playlists.store import DEFAULT_SETTINGS

        values = {field: DEFAULT_SETTINGS[key] for field, key in KEYS.items()}
        if self.store is None:
            return values
        readers = {bool: self.store.get_bool, int: self.store.get_int, float: self.store.get_float, str: self.store.get_str}
        for field, key in KEYS.items():
            stored = readers[type(DEFAULT_SETTINGS[key])](key)
            candidate = {**values, field: stored}
            try:
                _validate(candidate)
                values = candidate
            except (TypeError, ValueError) as exc:
                log.warning("setting %r: %s; using %r", key, exc, values[field])
        return values

    # ---- status -----------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        receiver = self.receiver.status()
        now = time.monotonic()
        frames = receiver["frames"]
        if not self._frame_times or self._frame_times[-1][1] != frames:
            self._frame_times.append((now, frames))
        window = [(t, n) for t, n in self._frame_times if now - t <= 2.0]
        fps = (window[-1][1] - window[0][1]) / (window[-1][0] - window[0][0]) if len(window) > 1 and window[-1][0] > window[0][0] else 0.0
        last = receiver.pop("last_frame_at")
        layout = self.receiver.layout
        return {
            **receiver,
            **self.source.state(),
            "age_s": None if last is None else round(now - last, 3),
            "fps": round(fps, 1),
            "universes": layout.universes,
            "ports": self.receiver.ports(),
            "dmx": self.dmx.status(),
        }


def _validate(s: dict[str, Any]) -> None:
    for key in ("artnet_enabled", "sacn_enabled", "dmx_enabled"):
        if not isinstance(s[key], bool):
            raise TypeError(f"{key} must be true or false")
    if s["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    layout = Layout(s["mode"], int(s["grid_width"]), int(s["grid_height"]))  # raises on a bad grid
    count = layout.universes
    if not 0 <= int(s["artnet_universe"]) <= ARTNET_UNIVERSES - count:
        raise ValueError(f"artnet_universe must be 0..{ARTNET_UNIVERSES - count} for {count} universe(s)")
    if not 1 <= int(s["sacn_universe"]) <= SACN_UNIVERSE_MAX - count + 1:
        raise ValueError(f"sacn_universe must be 1..{SACN_UNIVERSE_MAX - count + 1} for {count} universe(s)")
    if s["source"] not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if not 0.0 <= float(s["mix"]) <= 1.0:
        raise ValueError("mix must be in 0..1")
    if not 0.2 <= float(s["timeout_s"]) <= 60.0:
        raise ValueError("timeout_s must be 0.2..60")
    if not 0 <= int(s["dmx_artnet_universe"]) < ARTNET_UNIVERSES:
        raise ValueError(f"dmx_artnet_universe must be 0..{ARTNET_UNIVERSES - 1}")
    if not 1 <= int(s["dmx_sacn_universe"]) <= SACN_UNIVERSE_MAX:
        raise ValueError(f"dmx_sacn_universe must be 1..{SACN_UNIVERSE_MAX}")
    if not 1 <= int(s["dmx_address"]) <= 512 - DMX_WIDTH + 1:
        raise ValueError(f"dmx_address must be 1..{512 - DMX_WIDTH + 1} for the {DMX_WIDTH}-channel block")
