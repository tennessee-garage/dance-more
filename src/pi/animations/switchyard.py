"""Switchyard: Comet Train, until the corners start throwing switches.

It opens as Comet Train (comet_train.py): a comet on every edge running
one way, nose to tail, pulsing a few LEDs on each beat and now and then all
turning a corner together. After a while (Interval) the turns stop being
all at once. Instead one corner, on one line, throws a switch: from then
on every comet reaching that corner on that line turns there.

The rest of the floor re-routes around it:

- Lines the turned stream never reaches carry on as they were.
- Where the turned stream reaches the next corner and another stream is
  crossing it, they swap: the crossing comets turn onto the new heading
  and the turned stream takes their place. So a turn north jogs every line
  above it up by one at that corner, all the way to the floor's edge. The
  swap spreads at comet speed, one corner per tile side, as the first
  turned comet gets there.
- The edges after the switch have lost their supply, so they get a feed
  made from nothing: comets born at the corner behind the switch, which
  run in and turn to fill them.

Every Interval another switch is thrown, up to Switches of them, building a
map of currents. Then they come out again in reverse order, one per
Interval, until the floor is uniform and it is Comet Train once more.

How it works: the floor is a lattice of tile corners (junctions) joined by
links, each link one tile side of both lanes of a grid line (adjacent
tiles don't share LEDs, so every grid line has two lanes, as in Comet
Train). Each link has a flow, +1, -1 or none, and each junction a route
from the ways in to the ways out, straight unless a switch or a swap says
otherwise. Comets move a whole link per segment, from one corner to the
next, as Comet Train does; at each corner every comet takes its junction's
route, the inside lane taking the inside lane on a turn. A comet whose
route has nowhere to go ends there; a link flowing out of a junction that
nothing routes into gets new comets, born at that corner - which is also
how comets come in from the sides of the floor. The routes are a
one-to-one map at every junction, so no two comets ever land on one edge.

Throwing a switch saves the map first, so taking it out is putting the
saved map back; comets on links the old map doesn't use end at their next
corner.
"""

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.edges import Axis
from df2_pi.palette import choice, palette_param
from df2_pi.pixels import PixelFrame

WHITE = 0.8  # how far the head is pushed from its colour towards white
EASE = 3  # a step eases out: 1 - (1 - x)^EASE

OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
SIDEWAYS = {"N": "EW", "S": "EW", "E": "NS", "W": "NS"}
OUT = {"E": 1, "W": -1, "N": 1, "S": -1}  # the flow of a link leaving a junction this way
AXIS = {"E": Axis.X, "W": Axis.X, "N": Axis.Y, "S": Axis.Y}


@animation(
    name="Switchyard",
    description="Comet Train until the corners throw switches: one by one the flow turns for good at a corner and the floor re-routes round it, then the switches come out again.",
    author="df2",
    format="pixel",
    tags=["edges", "rhythm", "palette", "experimental"],
    sync="beat",
    params={
        "interval": Param(float, default=20.0, min=10.0, max=120.0, label="Interval (s)", help="How long it runs as Comet Train, and then between switches going in or coming out", macro=1),
        "switches": Param(int, default=4, min=1, max=8, label="Switches", help="How many are thrown before they start coming out", role="density"),
        "tail": Param(int, default=14, min=1, max=14, label="Tail length (LEDs)", help="Behind the head: 14 makes each comet a whole tile side, 15 LEDs", role="scale"),
        "step": Param(int, default=3, choices=[1, 3, 5, 15], label="LEDs per pulse", help="Divides a tile side, so the heads keep landing on the corners"),
        "beats": Param(float, default=1.0, choices=[0.5, 1.0, 2.0, 4.0], label="Beats per pulse"),
        "move": Param(float, default=0.3, min=0.05, max=1.0, label="Step time", help="The part of each pulse spent stepping: short slams into place, 1 never stops"),
        "turns": Param(float, default=0.3, min=0.0, max=1.0, label="Turn chance", help="While it runs as Comet Train: the chance, each time the heads reach the corners, that they all turn", role="variation"),
        "turn_time": Param(float, default=0.9, min=0.2, max=1.0, label="Turn time", help="The part of a pulse an all-together turn takes"),
        "palette": palette_param(),
        "drift": Param(float, default=0.05, min=0.0, max=0.5, label="Colour drift", help="How far round the palette each new line of comets moves on"),
        "variety": Param(float, default=0.15, min=0.0, max=1.0, label="Colour variety", help="How much new comets' colours differ from each other"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    p = ctx.params
    state = ctx.state
    if not state:
        _start(ctx)
    lattice = state["lattice"]
    n = lattice.n
    state["clock"] += ctx.dt

    position = ctx.t_beats / p["beats"]
    pulse = int(np.floor(position))
    if pulse != state["pulse"]:
        if state["pulse"] is not None:
            state["offset"] = int(round(state["to"])) - (n - 1)
            if state["offset"] >= n:
                _commit(state)
        state["pulse"] = pulse
        if state["offset"] == 0:
            _at_corner(ctx)
        state["from"] = n - 1 + state["offset"]
        state["to"] = 2 * n - 1 if state["segment"]["turn"] else min(state["from"] + p["step"], 2 * n - 1)

    x = position - pulse
    if state["segment"]["turn"]:
        progress = min(x / p["turn_time"], 1.0)  # an even run round the corner
    else:
        progress = 1.0 - (1.0 - min(x / p["move"], 1.0)) ** EASE
    head = state["from"] + (state["to"] - state["from"]) * progress
    return _draw(ctx, state["segment"], head)


# ---- the lattice ------------------------------------------------------------------------------


class Lattice:
    """Junctions are tile corners (row line, col line), 0..rows by 0..cols.
    A link is one tile side of both lanes of a grid line: ("h", row line,
    col) runs east from junction (row line, col), ("v", col line, row) runs
    north from junction (row, col line). Lane side +1 is above (or east of)
    the line, -1 below (or west of) it, as in comet_train.py."""

    def __init__(self, geo) -> None:
        self.rows, self.cols, self.n = geo.tile_rows, geo.tile_cols, geo.leds_per_side
        self.rails = {
            axis: np.stack([geo.rails(axis, i).reshape(-1, self.n) for i in range(2 * (self.rows if axis is Axis.X else self.cols))])
            for axis in Axis
        }

    def links(self, axis: Axis) -> list:
        if axis is Axis.X:
            return [("h", r, c) for r in range(self.rows + 1) for c in range(self.cols)]
        return [("v", c, r) for c in range(self.cols + 1) for r in range(self.rows)]

    def link(self, j, way: str):
        """The link leaving junction `j` going `way`, or None off the floor."""
        r, c = j
        if way == "E":
            return ("h", r, c) if c < self.cols else None
        if way == "W":
            return ("h", r, c - 1) if c >= 1 else None
        if way == "N":
            return ("v", c, r) if r < self.rows else None
        return ("v", c, r - 1) if r >= 1 else None

    def ahead(self, link, flow: int):
        """The junction a link's comets are heading for, and the way they come into it."""
        kind, line, k = link
        if kind == "h":
            return ((line, k + 1), "W") if flow > 0 else ((line, k), "E")
        return ((k + 1, line), "S") if flow > 0 else ((k, line), "N")

    def leds(self, link, side: int, flow: int):
        """One lane of a link, in the order its comets travel, or None off the floor."""
        kind, line, k = link
        rails = self.rails[Axis.X if kind == "h" else Axis.Y]
        lane = 2 * line if side > 0 else 2 * line - 1
        if not 0 <= lane < len(rails):
            return None
        return rails[lane, k] if flow > 0 else rails[lane, k][::-1]


def _lane(come: str, go: str, side: int) -> int:
    """The lane side a comet takes out of a junction: straight on it keeps
    its lane; on a turn the inside lane takes the inside lane."""
    if go == OPP[come]:
        return side
    d, turn = -OUT[come], OUT[go]
    return -d if side == turn else d


def _way(axis: Axis, sign: int) -> str:
    return ("E" if sign > 0 else "W") if axis is Axis.X else ("N" if sign > 0 else "S")


# ---- the run ----------------------------------------------------------------------------------


def _start(ctx) -> None:
    state = ctx.state
    p = ctx.params
    lattice = state["lattice"] = Lattice(ctx.geometry)
    state["axis"] = Axis.X if ctx.rng.random() < 0.5 else Axis.Y
    state["dir"] = 1 if ctx.rng.random() < 0.5 else -1
    state["flows"] = {link: state["dir"] for link in lattice.links(state["axis"])}  # the map
    state["routes"] = {}  # (junction, way in) -> way out, or None to end there; straight when absent
    state["on"] = dict(state["flows"])  # the flows the comets are on now
    state["base"] = ctx.rng.random()
    state["u"] = {}
    along = lattice.cols if state["axis"] is Axis.X else lattice.rows
    for link in state["on"]:
        k = link[2]  # how far along the floor, in the flow's axis
        age = k if state["dir"] > 0 else along - 1 - k
        for side in (1, -1):  # as if it had been running: each comet a drift older than the one behind it
            state["u"][(link, side)] = state["base"] - p["drift"] * age + _jitter(ctx)
    state["phase"] = "train"  # train -> build -> unwind -> train
    state["clock"] = 0.0
    state["next"] = p["interval"]
    state["saved"] = []  # the map before each switch
    state["fronts"] = []  # (junction, way in) where a turned stream arrives next
    state["switched"] = []  # (junction, way in, way out) of each switch, for the curious
    state["offset"] = 0
    state["pulse"] = None
    state["segment"] = None


def _at_corner(ctx) -> None:
    """The heads are on the corners: change the map if it's time, then plan
    the next tile side."""
    state = ctx.state
    p = ctx.params
    lattice = state["lattice"]
    fronts, state["fronts"] = state["fronts"], []
    for j, come in fronts:  # the first turned comets arrive now: the swap happens as they do
        state["fronts"] += _arrive(state, lattice, j, come, ctx.rng)
    due = state["clock"] >= state["next"]

    if state["phase"] == "train":
        if not due:
            if ctx.rng.random() < p["turns"]:
                turn = 1 if ctx.rng.random() < 0.5 else -1
                axis = Axis.Y if state["axis"] is Axis.X else Axis.X
                go = _way(axis, turn)
                state["axis"], state["dir"] = axis, turn
                state["flows"] = {link: turn for link in lattice.links(axis)}
                state["segment"] = _segment(ctx, lambda j, come: go, turn=True)
                return
        else:
            state["phase"] = "build"
    if state["phase"] == "build" and due and not state["fronts"]:
        if len(state["saved"]) < p["switches"] and _throw(ctx):
            state["next"] = state["clock"] + p["interval"]
        else:
            state["phase"] = "unwind"
    if state["phase"] == "unwind" and due:
        if state["saved"]:
            state["flows"], state["routes"] = state["saved"].pop()
            state["switched"].pop()
            state["fronts"] = []
        state["next"] = state["clock"] + p["interval"]
        if not state["saved"]:
            state["phase"] = "train"

    routes = state["routes"]
    state["segment"] = _segment(ctx, lambda j, come: routes.get((j, come), OPP[come]))


def _route(state, j, come):
    return state["routes"].get((j, come), OPP[come])


def _throw(ctx) -> bool:
    """Throw a switch: one corner on one line where the comets turn from now
    on. False if there is nowhere left to throw one."""
    state = ctx.state
    lattice = state["lattice"]
    flows = state["flows"]
    candidates = []
    for link, flow in flows.items():
        j, come = lattice.ahead(link, flow)
        if _route(state, j, come) != OPP[come] or lattice.link(j, OPP[come]) is None:
            continue  # only where comets go straight on, and onto the floor
        for go in SIDEWAYS[come]:
            out = lattice.link(j, go)
            if out is not None and out not in flows:
                interior = lattice.link(j, OPP[go]) is not None  # room behind for the feed
                candidates.append((not interior, j, come, go))
    if not candidates:
        return False
    best = [c for c in candidates if c[0] == min(c[0] for c in candidates)]
    _, j, come, go = best[ctx.rng.randrange(len(best))]
    state["saved"].append((dict(flows), dict(state["routes"])))
    state["switched"].append((j, come, go))
    switch(state, lattice, j, come, go)
    return True


def switch(state, lattice, j, come, go) -> None:
    """Comets coming into junction `j` from `come` turn to `go`, for good.
    The edges they used to carry on to are fed from the far side: comets
    born at the next corner back, turning in."""
    flows, routes = state["flows"], state["routes"]
    straight, behind = OPP[come], OPP[go]
    out = lattice.link(j, go)
    routes[(j, come)] = go
    flows[out] = OUT[go]
    feed = lattice.link(j, behind)
    if feed is not None and flows.get(feed, -OUT[behind]) == -OUT[behind]:
        flows[feed] = -OUT[behind]  # into j: where nothing else feeds it, comets are born at its far end
        routes[(j, behind)] = straight
    state["fronts"].append(lattice.ahead(out, OUT[go]))


def _arrive(state, lattice, j, come, rng) -> list:
    """A turned stream reaches junction `j` from `come`. If another stream
    crosses here, they swap: it turns to carry on the turned stream's way,
    and the turned stream takes its place. Returns where to look next."""
    flows, routes = state["flows"], state["routes"]
    ahead = OPP[come]
    for cross in SIDEWAYS[come]:
        into, onto = lattice.link(j, cross), lattice.link(j, OPP[cross])
        if into is not None and onto is not None and flows.get(into) == -OUT[cross] and _route(state, j, cross) == OPP[cross]:
            routes[(j, come)] = OPP[cross]
            return _onward(state, lattice, j, cross, ahead, rng)
    return _onward(state, lattice, j, come, ahead, rng)


def _onward(state, lattice, j, come, go, rng) -> list:
    """Send the stream coming into `j` from `come` out `go`: off the floor,
    onto an empty link (and on to the next corner), or onto a link only
    births were feeding. Blocked, it turns aside onto an empty link if
    there is one, or ends at this corner."""
    flows, routes = state["flows"], state["routes"]
    out = lattice.link(j, go)
    if out is None:
        routes[(j, come)] = go
        return []
    if out not in flows:
        routes[(j, come)] = go
        flows[out] = OUT[go]
        return [lattice.ahead(out, OUT[go])]
    if flows[out] == OUT[go] and not _fed(state, lattice, j, go):
        routes[(j, come)] = go
        return []
    aside = [w for w in SIDEWAYS[come] if lattice.link(j, w) is not None and lattice.link(j, w) not in flows]
    if aside:
        w = aside[rng.randrange(len(aside))]
        routes[(j, come)] = w
        flows[lattice.link(j, w)] = OUT[w]
        return [lattice.ahead(lattice.link(j, w), OUT[w])]
    routes[(j, come)] = None
    return []


def _fed(state, lattice, j, go) -> bool:
    """Whether a stream into `j` is routed out `go`."""
    for come in OPP:
        link = lattice.link(j, come)
        if come != go and link is not None and state["flows"].get(link) == -OUT[come] and _route(state, j, come) == go:
            return True
    return False


def _segment(ctx, route, turn: bool = False) -> dict:
    """One tile side of travel: every comet from the link it is on to the
    one its next corner routes it to (or to nothing), and new comets for
    links nothing is routed into. Paths are 2n flat LEDs, -1 off the floor."""
    state = ctx.state
    lattice = state["lattice"]
    n = lattice.n
    new = state["flows"]
    none = np.full(n, -1)
    paths, u, keys = [], [], []
    fed = set()
    for link, flow in state["on"].items():
        j, come = lattice.ahead(link, flow)
        go = route(j, come)
        out = lattice.link(j, go) if go is not None else None
        if out is not None and (new.get(out) != OUT[go] or out in fed):
            out = None  # nowhere to go: it ends at the corner
        if out is not None:
            fed.add(out)
        for side in (1, -1):
            before = lattice.leds(link, side, flow)
            colour = state["u"].get((link, side), state["base"])
            if out is None:
                key, after = None, None
            else:
                key = (out, _lane(come, go, side))
                after = lattice.leds(out, key[1], OUT[go])
            if before is not None or after is not None:
                paths.append(np.concatenate([none if before is None else before, none if after is None else after]))
                u.append(colour)
            keys.append((key, colour))
    state["base"] += ctx.params["drift"]
    for link, flow in new.items():
        if link in fed:
            continue
        for side in (1, -1):
            colour = state["base"] + _jitter(ctx)
            after = lattice.leds(link, side, flow)
            if after is not None:
                paths.append(np.concatenate([none, after]))
                u.append(colour)
            keys.append(((link, side), colour))
    return {
        "paths": np.array(paths).reshape(-1, 2 * n),
        "u": np.array(u, dtype=np.float64),
        "keys": keys,
        "turn": turn,
    }


def _commit(state) -> None:
    """The segment is over: every comet is on its new link."""
    state["u"] = {key: colour for key, colour in state["segment"]["keys"] if key is not None}
    state["on"] = dict(state["flows"])
    state["offset"] = 0


def _jitter(ctx) -> float:
    return ctx.params["variety"] * (ctx.rng.random() - 0.5)


def _draw(ctx, segment: dict, head: float) -> PixelFrame:
    tail = ctx.params["tail"]
    frame = PixelFrame.black(ctx.geometry)
    if not len(segment["paths"]):
        return frame
    colour = choice(ctx, ctx.params["palette"]).at(segment["u"]).astype(np.float32)  # (comets, 3)
    hot = colour + (255.0 - colour) * WHITE
    behind = head - np.arange(segment["paths"].shape[1])  # how far each LED of a path is behind the head
    head_w = np.clip(1.0 - np.abs(behind), 0.0, 1.0)
    tail_w = np.interp(behind, [0.0, 1.0, tail + 1.0], [0.0, 1.0, 0.0], left=0.0, right=0.0)
    lit = head_w[None, :, None] * hot[:, None, :] + tail_w[None, :, None] * colour[:, None, :]

    on = segment["paths"] >= 0
    light = np.zeros((ctx.geometry.tiles * ctx.geometry.leds_per_tile, 3), dtype=np.float32)
    np.maximum.at(light, segment["paths"][on], lit[on])
    frame.flat[:] = np.clip(light, 0, 255).astype(np.uint8)
    return frame
