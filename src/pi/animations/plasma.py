"""Classic plasma: overlapping sine waves, rendered as a 136x136 image.

Demonstrates `PixelFrame.from_grid()`: draw any picture you like on the
full cell grid and the driver keeps the ~21% of cells that are real LEDs
and throws the rest away. It is the "just hand me an image" path, and it
is how you would show a video frame, a shader, or anything else that
thinks in pixels rather than tiles or edges.

The image is computed with numpy on a coordinate mesh - no Python loops
over pixels, which matters at 30 FPS on a Pi.
"""

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.pixels import PixelFrame


@animation(
    name="Plasma",
    description="Overlapping sine waves across the whole floor.",
    author="df2",
    format="pixel",
    tags=["ambient", "image"],
    params={
        "speed": Param(float, default=1.0, min=0.1, max=4.0, label="Speed"),
        "scale": Param(float, default=1.0, min=0.3, max=3.0, label="Scale", help="bigger is blobbier"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    t = ctx.t * ctx.params["speed"]
    k = 0.08 / ctx.params["scale"]  # cycles per cell
    y, x = np.mgrid[0 : geo.height, 0 : geo.width].astype(np.float32)

    v = (
        np.sin(x * k + t)
        + np.sin((y * k + t) * 0.8)
        + np.sin((x + y) * k * 0.5 + t * 1.3)
        + np.sin(np.hypot(x - geo.width / 2, y - geo.height / 2) * k + t)
    ) / 4.0  # -1..1

    # three channels, each a shifted wave, in 0..255
    img = np.stack(
        [
            (np.sin(v * np.pi) + 1) * 127.5,
            (np.sin(v * np.pi + 2.1) + 1) * 127.5,
            (np.sin(v * np.pi + 4.2) + 1) * 127.5,
        ],
        axis=-1,
    ).astype(np.uint8)

    return PixelFrame.from_grid(img, geometry=geo)  # samples the lit cells only
