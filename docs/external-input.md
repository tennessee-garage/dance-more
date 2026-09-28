# Driving the floor from a media server

The floor listens for **Art-Net** (UDP 6454) and **sACN / E1.31** (UDP 5568,
unicast or multicast), so Resolume Arena, TouchDesigner, a lighting desk or
anything else that pixel-maps over DMX can send it pixels. Everything below is
set in the web UI's **External** tab (`df2-pi serve`), and stored.

## Network

- **Wire the Pi into the lighting network.** Art-Net over Wi-Fi drops and
  stutters.
- Point the sender at the Pi's IP (unicast), or use broadcast. For sACN,
  multicast works too: the floor joins the group of every universe it
  listens to.
- The floor answers **ArtPoll**, so it appears in a sender's node list as
  "Dance Floor". The node report shows the mode and universe count.

## Choosing a mode

| Mode | What the sender sends | Universes | Use it when |
| --- | --- | --- | --- |
| **Tile** (default) | 64 RGB pixels: one colour per tile | 1 | You want it working in five minutes. An 8×8 fixture grid in any software |
| **Grid** | A W×H RGB image (default 34×34); the floor samples it at each LED | ⌈W·H / 170⌉ (34×34: 7) | Video content. Resolume's pixel map as a plain grid |
| **Raw** | Every LED, 3,840 RGB pixels in chain order | 23 | Exact per-LED control (TouchDesigner with the LED table) |

Pixels pack **170 to a universe** (510 channels) and never straddle one.
Universes run consecutively from the **first universe** you set: Art-Net
counts port-addresses from 0, sACN universes from 1.

**Orientation.** Tile and grid are raster order starting at the **top-left of
the floor as the preview shows it**. The preview puts the Pi's edge at the
bottom, so the first pixel is the far corner on the left. If the floor is
rotated (Settings → rotation), the picture turns with it, so a sender never
needs to know.

**Colour.** Values are used as sent: they are already gamma-encoded, as video
is. Leave the sender's output gamma or colour correction at 1.0 / off, or the
floor applies gamma twice.

## Source: what the floor shows

| Source | While there is signal | When the signal stops |
| --- | --- | --- |
| **External** (default) | The sender's picture takes over the floor | After the **timeout** (default 2 s), the playlist entry that was playing restarts |
| **Mix** | The sender's picture over the playlist, faded in by **Mix** | The picture is removed; the playlist carries on |
| **Internal** | Ignored | — |

A crashed or sleeping sender therefore never leaves the floor frozen or dark.
Brightness, blackout, the show controls (strobe, tint, …) and rotation all
apply to external content exactly as to the floor's own animations.

"External" is also an ordinary animation (id `external`), so you can put it in
a playlist or run it as a layer by hand. It shows the last frame received.

## Resolume Arena

Arena, not Avenue: only Arena has DMX output.

1. In Arena's advanced output, add a DMX / Lumiverse output to the floor's IP
   (Art-Net, or sACN).
2. For **tile** mode, add a fixture of 64 RGB pixels, laid out 8×8. For
   **grid** mode, lay out W×H pixels to match the External tab (34×34 by
   default). Pixel order runs left to right and top to bottom, from the top-left.
3. Set the fixture's universe to the External tab's first Art-Net universe
   (0 by default).
4. Place the fixture over the part of the composition the floor should show.

Content advice: only about 21% of the floor's cells are LEDs (each tile is a
ring of 60 with a dark centre). Large soft shapes and colour fields read well;
thin lines and text disappear between the rings.

## TouchDesigner

Use a **DMX Out CHOP** (Art-Net or sACN) at the floor's IP.

- **Tile / grid:** Resize the TOP to 8×8 (tile) or W×H (grid), then **TOP to
  CHOP** with RGB channels interleaved in raster order from the top-left. TOP
  to CHOP reads rows bottom-up by default, so flip vertically first.
- **Raw:** Download the LED table (External tab, or `df2-pi leds > leds.csv`).
  It lists every LED in wire order with `u,v` (0–1, origin bottom-left, which
  is TouchDesigner's texture convention) and its universe and channel. Sample
  your TOP at those `u,v` positions and send the result in `index` order.

## Checking it works

The External tab shows **Live** with the sender's address and frame rate,
frames and packets received, and polls answered. "Showing" says what the floor
is actually doing: `external`, `mix`, or `internal` when there is no signal.
Nothing arriving while the sender says it is sending usually means one of
these:
- a different universe;
- a firewall;
- the sender on the wrong network interface;
- (for sACN) multicast not reaching the Pi. Try unicast.
