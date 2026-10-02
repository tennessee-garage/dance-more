# Driving the floor from Resolume

This guide is for someone who uses Resolume and wants the dance floor in the
show: Resolume's visuals on the floor, the floor keeping time with Resolume's
tempo, and the floor's own show running underneath. You do not need to know
how the floor works inside.

> **Status:** the floor's side of everything here is tested, and Ableton Link
> has been run with Resolume (its tap tempo drives the floor). The DMX pixel
> setup and the OSC output follow Resolume's documentation and have not yet
> been run against the floor end to end - if a menu name doesn't match your
> version of Resolume, the idea will still be right; please fix this page.

## What you can do from Resolume

| What | How | Needs |
| --- | --- | --- |
| **Put Resolume's visuals on the floor**, replacing or mixing over the floor's own show | DMX output: a **Lumiverse** in Advanced Output, sending Art-Net | **Arena** - Avenue has no DMX output |
| **Share the tempo**, so the floor's beat-locked animations and animation changes follow Resolume's BPM and bars | **Ableton Link** | Arena or Avenue |
| **Run the floor's show controls** - switching animations, palette, tint, bump, brightness, mix | **OSC output**, with each clip or fader set to one of the floor's addresses - see [section 5](#5-show-controls-from-resolume-osc) | Arena or Avenue |

The **Resolume demo** has every feature, DMX output and Link included. Its
only limits are an occasional Resolume logo on the output and a voice
reminder - and since the floor shows part of Resolume's output, the logo can
appear on the floor too.

## Before you start

- **Find the floor.** Its web UI is at **http://dancefloor.local:8000**; its
  IP address (for Resolume's *Target IP*) is printed by `ping
  dancefloor.local` in a terminal. The examples below use `192.168.1.50` -
  use yours.
- **Use a wired network** for the visuals. Art-Net over Wi-Fi drops and
  stutters.
- **Keep the floor's External tab open** in the web UI while you set up. It
  shows **Live**, Resolume's address and the frame rate as soon as pixels
  arrive.
- **On a Mac, allow incoming connections** for Resolume if macOS asks, or
  Ableton Link won't find the floor.

Everything set on the floor's side is set in the web UI and remembered across
restarts.

## 1. Visuals on the floor: tile mode

The quickest way is **tile mode**: the floor shows an 8×8 picture, one colour
per tile, which Resolume sends as one fixture of 64 RGB pixels in one Art-Net
universe.

### On the floor

In the **External** tab: **Art-Net** on, **Mode** *Tile*, **first Art-Net
universe** 0, **Source** *External*.

### In Resolume Arena

1. **Make the fixture.** Open the **Fixture Editor** (the gear icon next to a
   Lumiverse fixture's *Fixture* dropdown), press **+**, and set it up:

   | Setting | Value |
   | --- | --- |
   | Name | *Dance Floor 8x8* |
   | Width × Height | **8 × 8** |
   | Colour space | **RGB** |
   | Distribution | the one that **starts top-left and runs left to right, row by row** (not a zigzag) |

2. **Add a Lumiverse.** Open the **Advanced Output** window (from the
   **Output** menu) and add a **DMX Lumiverse** with the **+** menu.
3. **Point it at the floor.** Right-click the Lumiverse: the floor appears in
   the list of detected Art-Net nodes as **Dance Floor**. Pick it. (Or choose
   the **IP Address** target, enter `192.168.1.50`, and set **Subnet** 0,
   **Universe** 0.)
4. **Add the fixture** to the Lumiverse with the big **+** menu, at **start
   channel 1**.
5. **Place it.** In the **Input Selection** view, move and scale the fixture
   over the part of your composition the floor should show. Keep it square -
   the floor is.

The External tab should now show **Live** with Resolume's address, and the
floor shows that square of your composition.

### Getting it to look right

- **Soft content works best.** Resolume reads the **centre pixel** of each of
  the 64 cells rather than averaging them, so fine detail in your content
  flickers on the floor as it moves past. Big shapes and colour fields read
  well; a **Blur** effect on the layer (or a softer clip) cures shimmer. On
  the floor itself only about a fifth of the area is LEDs - each tile is a
  ring with a dark middle - so thin lines and text vanish anyway.
- **Leave colour alone.** Keep the Lumiverse's brightness, contrast and colour
  at their defaults, and no gamma correction: the floor uses the values as
  sent and does its own.
- **Upside down or mirrored?** Check the fixture's Distribution, or use the
  fixture's flip buttons (the Pacman icons) in Advanced Output. To turn the
  whole picture by 90° steps, use the floor's **rotation** setting instead -
  the picture turns with it, and Resolume doesn't need to know.

## 2. More detail: grid mode

**Grid mode** sends a bigger picture - W×H pixels - which the floor samples
at every one of its LEDs. Pixels pack 170 to a universe and never straddle
one, so pick a grid that divides nicely:

| Grid | Universes | In Resolume |
| --- | --- | --- |
| **13 × 13** (169 pixels) | 1 | One 13×13 fixture, set up exactly as in section 1 |
| **34 × 34** (the default) | 7 | Seven **34 × 5** fixtures (a 34-pixel row × 5 rows is exactly 170), each in its own Lumiverse on universes 0-6, stacked top to bottom over the same square. The seventh holds the last 4 rows: make it 34 × 4 |

Set the same width and height in the External tab's **Grid** settings, and
the **first Art-Net universe** to the first fixture's universe. Universe 0
carries the **top** rows.

## 3. Mixing with the floor's own show

The floor's **Source** decides what happens to Resolume's picture:

| Source | While Resolume sends | When it stops |
| --- | --- | --- |
| **External** | Resolume's picture replaces the floor's show | After the timeout (2 s), the floor's show comes back |
| **Mix** | Resolume's picture over the floor's show, faded in by **Mix** | Resolume's picture is removed; the floor's show carries on |
| **Internal** | Ignored | - |

**Mix** suits a live set: the floor keeps playing its own animations - which
already move on the beat once tempo is shared - and Resolume's visuals sit on
top, as much or as little as the Mix slider says. If Resolume stops sending
(you close it, the Lumiverse is switched off), the floor carries on by itself
after the timeout rather than freezing or going dark.

The floor's show controls - brightness, blackout, strobe, tint, palette -
apply to Resolume's picture too.

## 4. Share the tempo: Ableton Link

This is the part that has been run with Resolume.

1. **In Resolume:** **View → Show Ableton Link** - a Link button appears in the
   toolbar near **BPM**, **TAP** and **RESYNC**. Click it to turn Link on.
   (Make sure Resolume is the active app when you open View: on a Mac the
   menu bar shows the active app's menus.)
2. **On the floor:** External tab → **Beat sync** → **Ableton Link**. It shows
   Resolume's BPM and **1 peer**.

Now:

- **Changing the tempo in Resolume** - the BPM field or **TAP** - changes the
  floor's tempo. (With Link on, Resolume disables its own Resync and Pause
  for the BPM; that's Resolume's design.)
- **Beat-locked animations** on the floor - Waves, Rainbow Sweep,
  Checkerboard - move with Resolume's beat.
- **Quantize launches** (Beat sync section), set to **Bar**, makes animation
  changes from the floor's web UI or a desk land on the next bar line. The
  transport bar shows *waiting for the bar* until they do.
- **Bar lines match.** Resolume's beat indicator - the four-quarter circle at
  the bottom left, next to play - starts its first quarter on the bar line;
  the floor's transport bar has a lamp per beat with **beat 1 in amber**. They
  light together.
- **Latency offset.** If the floor's changes land slightly after Resolume's
  beat, raise **Latency offset (ms)** in the Beat sync section until a kick
  and a flip on the floor line up by eye.
- **Beats per bar** in the floor's Beat sync section should match the bar
  length you use in Resolume (4 for most music).

Link finds peers by multicast on the local network, so Resolume and the floor
must be on the same one.

## 5. Show controls from Resolume: OSC

The floor takes OSC on **UDP port 7000**, and Resolume can send OSC from
anything in its interface. The trick is that Resolume lets you **choose the
address** each thing sends, so you point a clip or a fader at one of the
floor's addresses (the full list is in
[external-input.md](external-input.md#osc)).

### Point Resolume's OSC output at the floor

In Resolume's **Preferences → OSC**: turn **OSC Output** on, set the target to
the floor's IP (`192.168.1.50`) and the **outgoing port** to **7000**. Use a
custom output preset rather than *Output All OSC Messages* - "all" includes
every playhead position, which the floor would only ignore - and in it, set
the address for each thing you want the floor to follow.

### What to send

Clips send a value when they launch, and the floor's addresses with the
target in the path don't care what that value is:

| In Resolume | Set its outgoing address to | Floor does |
| --- | --- | --- |
| A clip, launched | `/floor/play/lightning` | Plays the Lightning animation |
| A clip, launched | `/floor/palette/fire` | Switches the floor palette to *fire* |
| A clip, launched | `/floor/goto/2` | Goes to entry 2 of the playlist (counting from 0) |
| A clip, launched | `/floor/bump` | A white flash |
| A clip, launched | `/floor/next` | The next playlist entry |
| A layer's opacity | `/floor/mix` | How much of Resolume's picture is over the floor's show (Source *Mix*) |
| The composition master, or a dashboard fader | `/floor/brightness` | The floor's master brightness |
| A dashboard knob | `/floor/macro/1` | The playing animation's main knob |
| A dashboard knob | `/floor/speed` | Show speed: 0.5 is normal |

Floats are 0..1, which is what Resolume sends for faders and knobs. With
**Quantize launches** set to *Bar* on the floor, animation and entry changes
from clips land on the next bar line.

### Check what arrives

The floor's **External** tab has an **OSC** section that lists the last 20
messages it received, from where, and marks any it didn't understand (a
misspelt address, an animation that doesn't exist). Launch a clip and watch
it appear there.

### Without OSC

A **MIDI controller** plugged into the Pi, such as an APC mini, runs the
floor directly ([MIDI](external-input.md#midi)), alongside whatever controls
Resolume.

The floor's **web UI** on a phone, tablet or laptop (**http://dancefloor.local:8000**)
runs everything too, and a **lighting desk or QLC+** can use the 18-channel
DMX control block ([Lighting desk control](external-input.md#lighting-desk-control)).


## An example set

- **Floor:** a playlist of a few animations, Beat sync on Ableton Link,
  Quantize launches on *Bar*, palette *ocean*, Source *Mix* at about 60%.
- **Resolume Arena:** a Lumiverse with the 8×8 fixture over a soft, colourful
  part of the composition; Link on; the BPM tapped in from the music.
- **Resolume clips** set to `/floor/palette/<name>` for colour changes and
  `/floor/play/<animation>` for the big moments; a layer's opacity on
  `/floor/mix`.
- **A tablet** with the floor's web UI open, just in case.

The floor's own animations ride the beat underneath, Resolume's visuals sit
on top, and changes land on the bar.

## Troubleshooting

| Symptom | Look at |
| --- | --- |
| **Dance Floor** isn't in the Lumiverse's node list | Both on the same wired network; the macOS firewall; or skip discovery and use the **IP Address** target |
| The External tab never shows **Live** | Art-Net is on in the External tab; the Lumiverse's Subnet / Universe match the tab's first universe (0 / 0 by default); the Lumiverse and fixture are enabled |
| The picture is upside down, mirrored or scrambled | The fixture's Distribution (top-left, left to right, row by row); the fixture's flip buttons; for grid mode, which universe holds the top rows |
| Detail shimmers or flickers | Resolume samples the centre of each cell: blur the layer, or use softer content |
| Colours look washed out or too dark | The Lumiverse's brightness / contrast / colour, or any gamma correction, back to default |
| The floor goes back to its own show after 2 s | Resolume has stopped sending - the Lumiverse switched off, or the composition stopped. The timeout is in the External tab |
| A Resolume logo appears on the floor | The demo's watermark: it is in the output the floor samples |
| Link shows no peers on the floor | Link switched on in Resolume (the toolbar button); both on the same network; Wi-Fi client isolation; the macOS firewall |
| A clip set to a `/floor/...` address does nothing | The External tab's OSC monitor: nothing listed means Resolume's OSC output isn't reaching the floor (target IP, port 7000, OSC Output on); listed with an error means the address or animation name is wrong |
| Floor changes land after Resolume's beat | Raise **Latency offset** in the floor's Beat sync section |

For the floor side in more depth - modes, universes, orientation, the timeout
and takeover - see [external-input.md](external-input.md).
