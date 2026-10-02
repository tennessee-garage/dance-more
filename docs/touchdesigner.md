# Driving the floor from TouchDesigner

This guide is for someone who has built a few TouchDesigner networks and wants
the dance floor in them: drawing pictures on it, switching what it plays,
changing its colours, flashing it on a kick, and keeping it in time with the
music. You do not need to know how the floor works inside.

> **Status:** the floor's side of everything here is tested. The TouchDesigner
> side follows Derivative's operator documentation and has not yet been run
> against the floor end to end - if a parameter name doesn't match your version
> of TouchDesigner, the idea will still be right; please fix this page.

## What you can control

The floor runs its own show - playlists of built-in animations - and
TouchDesigner can join in at four levels. Use whichever you need; they work
together.

| Level | What TouchDesigner does | Sent with | Section |
| --- | --- | --- | --- |
| **Pixels** | Draws its own picture on the floor, replacing or mixing over the floor's show | **DMX Out CHOP** (Art-Net or sACN) | [1](#1-send-a-picture-to-the-floor) |
| **Show controls** | Dimmer, strobe, speed, tint, bump, palette, hold, and switching animations | **DMX Out CHOP**: the floor's 18-channel *control block* | [2](#2-run-the-show-with-the-control-block) |
| **Everything else** | Play any animation by name, set its parameters, layers, triggers, hue, saturation - anything the web UI does | **Web Client DAT** (the floor's HTTP API) | [3](#3-anything-else-the-http-api) |
| **Tempo** | Share one beat, so the floor's beat-locked animations and launches follow your timeline | **Ableton Link CHOP** | [4](#4-share-the-tempo) |

OSC is planned ([#132](https://github.com/tennessee-garage/dance-more/issues/132))
and will be the most natural fit for TouchDesigner when it lands; until then,
the control block and the HTTP API cover everything.

## Before you start

- **Find the floor.** Its web UI is at **http://dancefloor.local:8000**. The
  DMX Out CHOP wants an IP address rather than a name: `ping dancefloor.local`
  in a terminal prints it. The examples below use `192.168.1.50` - use yours.
- **Use a wired network** for anything you send at frame rate. Art-Net over
  Wi-Fi drops and stutters.
- **Keep the External tab open** in the web UI while you set up. It shows
  **Live**, the sender's address and its frame rate as soon as pictures
  arrive, and the control block's last values as they change.
- **On a Mac, allow incoming connections** for TouchDesigner if macOS asks,
  or Ableton Link won't find the floor.

Everything below that is set on the floor's side is set in the web UI and
remembered across restarts.

## 1. Send a picture to the floor

The quickest way is **tile mode**: the floor takes an 8×8 picture, one colour
per tile. It is one universe of 192 DMX channels: 64 pixels, each red, green,
blue, in rows from the top-left.

### The network

```
moviefilein1 ─► resolution1 ─► flip1 ─► topto1 ─► select1 ─► shuffle1 ─► shuffle2 ─► dmxout1
   (TOP)          (TOP)        (TOP)    (CHOP)     (CHOP)     (CHOP)      (CHOP)      (CHOP)
```

| Operator | Set |
| --- | --- |
| **Movie File In TOP** (or any TOP: a Noise, a Ramp, your composition) | your content |
| **Resolution TOP** | Output Resolution **Custom**, **8 × 8** |
| **Flip TOP** | **Flip Y** on. TouchDesigner counts image rows from the bottom; the floor wants the top row first |
| **TOP to CHOP** | **RGBA Units** to **0 to 255**, **Output as Single Channel Set** on. This gives channels `r`, `g`, `b` and `a`, 64 samples each |
| **Select CHOP** | **Channel Names** `r g b` - drops alpha |
| **Shuffle CHOP** (shuffle1) | **Method** **Swap Channels and Samples** - now 64 channels, each `[r, g, b]` for one pixel |
| **Shuffle CHOP** (shuffle2) | **Method** **Sequence All Channels** - now one channel: `r0, g0, b0, r1, g1, b1, …` |
| **DMX Out CHOP** | **Interface** **Art-Net**, **Network Address** `192.168.1.50`, **Net** 0, **Subnet** 0, **Universe** 0, **Format** **Packet Per Channel** (each sample is a DMX channel), **Rate** 30 |

The two Shuffle CHOPs matter: the floor wants each pixel's red, green and blue
next to each other, and TOP to CHOP gives you all the reds, then all the
greens, then all the blues.

### On the floor

In the **External** tab: **Art-Net** on, **Mode** *Tile*, **first Art-Net
universe** 0, **Source** *External*. The tab should show **Live** with your
machine's address and about 30 fps, and the floor shows your picture.

### Choosing what the floor shows

| Source | While TouchDesigner sends | When it stops |
| --- | --- | --- |
| **External** | Your picture replaces the floor's show | After the timeout (2 s), the floor's show comes back |
| **Mix** | Your picture over the floor's show, faded in by **Mix** | Your picture is removed; the show carries on |
| **Internal** | Ignored | - |

**Mix** is the interesting one for a live show: the floor keeps playing its
own animations and you lay video over them, with the amount on a fader (see
channel 4 in [section 2](#2-run-the-show-with-the-control-block)).

### More detail than 8×8

- **Grid mode** samples a W×H picture at every LED. Set the size in the
  External tab and resize your TOP to match. A grid of **13 × 13** (169
  pixels) still fits in one universe, so the network above works unchanged
  apart from the Resolution TOP. Bigger grids need one universe per 170
  pixels: split the sequenced channel with a Shuffle CHOP (**Split N Samples**,
  N = 510) and send each piece with its own DMX Out CHOP on the next universe.
- **Raw mode** addresses every one of the 3,840 LEDs; see
  [external-input.md](external-input.md#touchdesigner) for the LED table.

Only about a fifth of the floor's area is LEDs - each tile is a ring with a
dark middle - so big soft shapes and colour fields read best; thin lines and
text vanish between the rings. Leave your output gamma at 1: the floor uses
values as you send them.

## 2. Run the show with the control block

The floor also behaves like a lighting fixture: 18 DMX channels that run the
show. TouchDesigner sends them with a second DMX Out CHOP.

### On the floor

In the External tab's **DMX control** section: **Listen for the control
block** on, **Art-Net universe** 1, **Start address** 1. That puts the 18
channels at the start of universe 1, away from your pixels in universe 0.

> **Keep it out of the pixel universe.** If the control block shared universe
> 0 with tile mode, a separate DMX Out CHOP sending pixels would send zeros on
> the control channels - and zero on channel 1 is the master dimmer at zero:
> a dark floor. The External tab warns when the two overlap.

### The channels

| Ch | Name | Values | Acts |
| --- | --- | --- | --- |
| 1 | Master dimmer | 0 dark … 255 full | always |
| 2 | Strobe | 0 off, 1…255 slow to fast (up to the strobe cap, 10 Hz by default) | always |
| 3 | Source | 0–84 internal, 85–169 external, 170–255 mix | always |
| 4 | Mix | how much of your picture, when the source is mix | always |
| 5 | Bank | playlist number, counting from 0 in name order (capitals sort first) | on change |
| 6 | Program | entry number in that playlist, from 0 | on change |
| 7 | Speed | 0 stopped, 128 normal, 255 four times | always |
| 8–11 | Macro 1–4 | the playing animation's main knobs | on change |
| 12–14 | Tint red, green, blue | the tint colour | always |
| 15 | Tint amount | 0 off … 255 fully the tint colour | always |
| 16 | Bump | a white flash of value/255 **each time it rises** | on a rise |
| 17 | Hold | 0–127 the playlist moves on as usual; 128–255 the current animation plays until released | crossing 128 |
| 18 | Palette | palette number, as listed in the **Palettes** tab | on change |

*Always* channels follow your values from the first packet. *On change*
channels do nothing until a value changes, so connecting with them at 0
doesn't reload a playlist or switch palette. If TouchDesigner stops sending
for the timeout, the dimmer, strobe, speed, tint and source go back to the
floor's own settings, and a hold the control block set is released.

### The network

The simplest source is a **Constant CHOP** with 18 channels, in order,
values 0–255:

```
constant1 (CHOP, 18 channels) ─► dmxout2 (CHOP)
```

| Operator | Set |
| --- | --- |
| **Constant CHOP** | 18 channels named `dimmer strobe source mix bank program speed macro1 macro2 macro3 macro4 tintr tintg tintb tint bump hold palette`. Start with dimmer 255, speed 128, the rest 0 |
| **DMX Out CHOP** | **Interface** **Art-Net**, **Network Address** `192.168.1.50`, **Universe** 1, **Format** **Packet Per Sample** (each channel is a DMX channel), **Rate** 30 |

Then drive the values however you like: export sliders and buttons from a
control panel onto the Constant CHOP's values, or merge in CHOPs from
elsewhere in your network. A few recipes:

- **Switch animations.** Make a playlist in the web UI - say *TD* - with
  the animations you want as its entries. Set **Bank** to its number
  (playlists count from 0 in name order, capitals first) and **Program** to
  the entry. With *Quantize launches* on (see [section 4](#4-share-the-tempo))
  the change lands on the next beat or bar.
- **Change colour scheme.** Set **Palette** to a number from the Palettes
  tab's list (0 rainbow, 1 fire, 2 ice, 3 ocean, …). Animations that follow
  the floor palette recolour on the next frame.
- **Tint.** Set **Tint red/green/blue** to a colour and raise **Tint amount**.
- **Flash on the kick.** Feed a pulse that goes up on each hit into **Bump**:
  for example an **Audio Device In CHOP** into an **Analyze CHOP** and a
  **Trigger CHOP**, scaled to 0–255 with a **Math CHOP**. Each rise is one
  flash.
- **Fade the floor's show under your video.** Source 255 (mix), then ride
  **Mix**.

## 3. Anything else: the HTTP API

Everything the web UI does goes through a small web API, so TouchDesigner can
do all of it - play any animation by name with your own parameter values,
nudge a parameter live, run a layer, fire triggers, shift hue. The **Web
Client DAT** sends web requests without stalling TouchDesigner.

### Setup

1. Add a **Web Client DAT** named `webclient1`.
2. Add a **Text DAT** named `floor` holding:

```python
import json

FLOOR = 'http://192.168.1.50:8000/api/'


def post(path, body=None):
    """Send one command to the floor, e.g. post('transport/bump', {'level': 1.0})."""
    op('webclient1').request(
        FLOOR + path, 'POST',
        header={'Content-Type': 'application/json'},
        data=json.dumps(body or {}),
    )
```

3. Call it from anywhere - a CHOP Execute DAT watching a button, a panel
   script, a Timer CHOP's callbacks - as `mod.floor.post(...)`:

```python
# in a CHOP Execute DAT watching your buttons, with Off to On enabled
def onOffToOn(channel, sampleIndex, val, prev):
    if channel.name == 'drop':
        mod.floor.post('transport/animation', {'id': 'lightning', 'params': {'forks': 4}})
    elif channel.name == 'calm':
        mod.floor.post('transport/animation', {'id': 'stardust'})
    elif channel.name == 'red':
        mod.floor.post('palettes/active', {'name': 'fire'})
```

If the Web Client DAT's call signature differs in your version, Python's own
`urllib.request` does the same job; it waits for the reply, which on a wired
network takes a few milliseconds.

### What you can send

All are `POST` to `http://<floor>:8000/api/…` with a JSON body.

| Path | Body | Does |
| --- | --- | --- |
| `transport/animation` | `{"id": "waves", "params": {"map": "radial"}, "hold": 30}` | Play one animation, with these parameter values, for `hold` seconds (leave it out to play until *next*), then back to the playlist |
| `transport/params` | `{"speed": 2.0}` | Change the playing animation's parameters live |
| `transport/next`, `transport/previous` | - | Move through the playlist |
| `transport/goto` | `{"index": 3}` | Jump to a playlist entry |
| `transport/load` | `{"playlist": "TD"}` | Switch playlist, by name or id |
| `transport/hold` | `{"on": true}` | Keep the current animation playing until released |
| `transport/bump` | `{"level": 1.0, "decay_s": 0.25}` | A flash toward white |
| `transport/tint` | `{"r": 255, "g": 40, "b": 0, "amount": 0.5}` | Tint toward a colour |
| `transport/strobe` | `{"rate_hz": 6}` | Strobe; 0 is off |
| `transport/speed` | `{"value": 0.5}` | Run everything slower or faster (0–4) |
| `transport/hue_shift` | `{"value": 0.25}` | Turn every hue, in turns |
| `transport/saturation` | `{"value": 0.0}` | 0 grey … 1 normal … 2 vivid |
| `transport/freeze` | `{"on": true}` | Hold the picture while time runs on underneath |
| `transport/brightness` | `{"value": 180}` | Master brightness, 0–255 |
| `transport/blackout` | `{"on": true}` | Black out the floor |
| `transport/trigger` | `{"slot": 3, "velocity": 1.0}` | A hit for animations that react to triggers (Ripple drops a ring) |
| `transport/layer` | `{"id": "stardust", "mode": "add", "amount": 0.6}` | Run a second animation over whatever plays |
| `transport/clear_layer` | - | Remove it |
| `transport/reset_show` | - | Strobe, tint, hue, saturation, speed and freeze back to normal |
| `palettes/active` | `{"name": "ocean"}` | Change the floor palette |
| `beat/tap` | - | One tap of tap tempo |

To see what there is to play and what each animation's parameters are
called, open **http://dancefloor.local:8000/api/animations** in a browser;
**/api/state** is what the floor is doing right now; and
**http://dancefloor.local:8000/docs** is the complete, browsable reference
with every option.

## 4. Share the tempo

The floor can follow an Ableton Link session, and so can TouchDesigner, so
both run on the same beat - whichever app sets the tempo.

1. **On the floor:** External tab → **Beat sync** → **Ableton Link**. It shows
   the tempo and how many peers it sees.
2. **In TouchDesigner:** add an **Ableton Link CHOP** and turn **Enable** on.
   Its `beat`, `bar` and `phase` channels are the shared beat: drive your
   content from them and it stays locked to the floor.
3. Set the same bar length on both: **Beats per bar** in the floor's Beat sync
   section, the **Signature** parameter in TouchDesigner.

With the beat shared, the floor's beat-locked animations (Waves, Rainbow
Sweep, Checkerboard) move with the music, and **Quantize launches** set to
*Bar* makes animation changes - from the control block's bank/program, the
HTTP API or the web UI - land on the next bar line. Link finds peers by
multicast on the local network, so both machines must be on the same one.

## An example show

One way to put it together:

- **Floor:** a playlist *TD* of a few animations, Beat sync on Link, Quantize
  launches *Bar*, the floor palette set to *ocean*.
- **Pixels, universe 0:** your visuals into tile mode, with the floor's source
  on **mix** so its own animations play underneath.
- **Control block, universe 1:** a fader on **Mix**, the master **Dimmer** on
  another, **Bump** from the kick drum, and three buttons on **Program** for
  the verse, chorus and drop entries of the *TD* playlist.
- **HTTP API:** a button that plays Lightning with extra forks for the big
  moment, and palette buttons for the colour changes.
- **Tempo:** an Ableton Link CHOP in your network and the floor on the same
  session.

## Troubleshooting

| Symptom | Look at |
| --- | --- |
| External tab never shows **Live** | The DMX Out CHOP's Network Address is the floor's IP; Art-Net is on in the External tab; your universe matches the tab's first universe (Art-Net counts from 0, sACN from 1); a firewall on your machine |
| The picture is upside down, mirrored or the wrong colours | The Flip TOP (Flip Y); the Select CHOP keeps `r g b` in that order; both Shuffle CHOPs are in place |
| Colours look washed out or too dark | Output gamma or colour correction left on: the floor applies its own |
| The floor goes **dark** when the control block is on | Channel 1 is the master dimmer: start it at 255. Or the control block shares the pixel universe |
| Bank, program, palette or a macro does nothing | They act only when the value **changes**; bank counts playlists in name order from 0; palette counts the Palettes tab list from 0 |
| Bump flashes once and stops | It flashes on each **rise**: send it back to 0 between hits |
| HTTP commands do nothing | Open `http://<floor>:8000/api/state` in a browser from the same machine; check the Web Client DAT for errors (a `404` is a wrong path or animation id, a `422` a value out of range) |
| Ableton Link shows no peers | The machines are on different networks, or Wi-Fi client isolation; allow TouchDesigner through the macOS firewall |

For the floor side in more depth - modes, universes, orientation, the timeout
and takeover - see [external-input.md](external-input.md).
