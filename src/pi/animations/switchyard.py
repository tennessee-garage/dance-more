"""Switchyard: Comet Train, until the corners start throwing switches.

It opens as Comet Train (comet_train.py): a comet on every edge running
one way, nose to tail, pulsing a few LEDs on each beat and now and then all
turning a corner together. After a while (Interval) the turns stop being
all at once. Instead one corner throws a switch: the stream through it
turns there, for good, and the floor re-routes round it so that no two
streams ever cross. Say everything runs right and the switch at the
middle corner turns up:

- Below the switch, nothing changes.
- Above and to the left, each line turns up one corner before the line
  below it, so as not to cross that line's upward stream: a diagonal of
  turns running up and left from the switch.
- Above and to the right, the lines have lost their supply. Each is fed
  from the top: a stream coming down a column and turning right one corner
  before the line below it - a diagonal running up and right. The edge just
  after the switch is left empty, between the stream going up and the
  first one coming down.

Every Interval another switch is thrown, up to Switches of them, each
re-routing the map it lands in, so it builds into a map of currents. Then
they come out again in reverse order, one per Interval, until the floor is
uniform and it is Comet Train once more.

The map. The floor is a lattice of tile corners (junctions) joined by
links, each link one tile side of both lanes of a grid line (adjacent tiles
don't share LEDs, so each grid line has two lanes, as in Comet Train). The
map gives every junction one way in and one way out, and the streams are
paths through it that never share a junction, entering and leaving at the
floor's edges. Comets move a whole link per segment, from one corner to
the next. At each corner a comet goes out by that corner's way out, the
inside lane taking the inside lane on a turn. That is true even as the
map changes: whatever arrives at a corner leaves by its new way out, so a
switch, or one coming out, re-routes every comet at once without making or
losing any mid-floor. New comets only come in at the floor's edges.

Throwing a switch (`switch`): the stream turns and runs straight to the
floor's edge. Any stream it would cross turns the same way one corner
earlier, which is itself a switch, so this recurses into the diagonal.
Then the pieces of streams that were cut off are fed (`_feed`), nearest
the switch first: each from the side, by a stream laid from the floor's
edge through junctions nothing live is using. A switch that leaves the
map inconsistent - a stream running into another head on, nowhere to feed
from - is not thrown, nor is one that would send comets back the way they
came, going in or coming out; another corner is tried.
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


class Blocked(Exception):
    """A switch that cannot be laid without streams meeting."""


@animation(
    name="Switchyard",
    description="Comet Train until the corners throw switches: one by one a stream turns for good and the floor re-routes round it into a map of currents, then the switches come out again.",
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
    n = state["lattice"].n
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
        # An even run round the corner, landing on the next beat (or the next pulse, if that comes sooner).
        progress = min(x * max(p["beats"], 1.0), 1.0)
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

    def junctions(self):
        return [(r, c) for r in range(self.rows + 1) for c in range(self.cols + 1)]

    def next(self, j, way: str):
        """The junction one link from `j` going `way`, or None off the floor."""
        r, c = j
        r, c = {"E": (r, c + 1), "W": (r, c - 1), "N": (r + 1, c), "S": (r - 1, c)}[way]
        return (r, c) if 0 <= r <= self.rows and 0 <= c <= self.cols else None

    def link(self, j, way: str):
        """The link leaving junction `j` going `way`, or None off the floor."""
        if self.next(j, way) is None:
            return None
        r, c = j
        return {"E": ("h", r, c), "W": ("h", r, c - 1), "N": ("v", c, r), "S": ("v", c, r - 1)}[way]

    def leds(self, link, side: int, flow: int):
        """One lane of a link, in the order its comets travel, or None off the floor."""
        kind, line, k = link
        rails = self.rails[Axis.X if kind == "h" else Axis.Y]
        lane = 2 * line if side > 0 else 2 * line - 1
        if not 0 <= lane < len(rails):
            return None
        return rails[lane, k] if flow > 0 else rails[lane, k][::-1]

    def uniform(self, axis: Axis, sign: int) -> dict:
        """The map with every line running one way: junction -> (way in, way out)."""
        go = _way(axis, sign)
        return {j: (OPP[go], go) for j in self.junctions()}


def _way(axis: Axis, sign: int) -> str:
    return ("E" if sign > 0 else "W") if axis is Axis.X else ("N" if sign > 0 else "S")


def _lane(come: str, go: str, side: int) -> int:
    """The lane side a comet takes out of a junction: straight on it keeps
    its lane; on a turn the inside lane takes the inside lane."""
    if go == OPP[come]:
        return side
    d, turn = -OUT[come], OUT[go]
    return -d if side == turn else d


# ---- the map ------------------------------------------------------------------------------------


def live(lattice, paths: dict, j) -> bool:
    """Whether the stream through `j` traces back to the floor's edge."""
    seen = set()
    while j not in seen:
        seen.add(j)
        come = paths[j][0]
        back = lattice.next(j, come)
        if back is None:
            return True
        if back not in paths or paths[back][1] != OPP[come]:
            return False  # cut off
        j = back
    return False  # a loop


def switch(lattice, paths: dict, j, go: str, _depth: int = 0) -> None:
    """The stream through junction `j` turns `go` there and runs straight
    to the floor's edge. Any live stream it would cross turns the same way
    a junction earlier, recursively. Leaves cut-off pieces for `_feed`."""
    if _depth > lattice.rows + lattice.cols or paths[j][0] == go:
        raise Blocked
    k = j
    while (k := lattice.next(k, go)) is not None:
        if k not in paths or not live(lattice, paths, k):
            continue
        come = paths[k][0]
        if come == go:
            raise Blocked  # a stream coming the other way, head on
        if come in SIDEWAYS[go]:
            back = lattice.next(k, come)
            if back is not None and back in paths and paths[back][1] == OPP[come]:
                switch(lattice, paths, back, go, _depth + 1)
            # else it comes in from the floor's edge right here: the run cuts it off
    paths[j] = (paths[j][0], go)
    k = j
    while (k := lattice.next(k, go)) is not None:
        paths[k] = (OPP[go], go)


def _feed(lattice, paths: dict, origin, prefer: str) -> None:
    """Re-supply every cut-off piece of stream, nearest `origin` first: from
    the side, preferably `prefer`, by a stream laid in from the floor's edge
    through junctions no live stream uses. Raises Blocked if one can't be."""
    for _ in range(len(paths) + 1):
        starts = []
        for j, (come, _) in paths.items():
            back = lattice.next(j, come)
            if back is not None and (back not in paths or paths[back][1] != OPP[come]) and not live(lattice, paths, j):
                starts.append(j)
        if not starts:
            return
        f = min(starts, key=lambda j: (abs(j[0] - origin[0]) + abs(j[1] - origin[1]), j))
        go = paths[f][1]
        options = []
        for side in SIDEWAYS[go]:
            walk, g = [], f
            while (g := lattice.next(g, side)) is not None:
                if g in paths and live(lattice, paths, g):
                    break
                walk.append(g)
            else:
                options.append((side != prefer, side, walk))
        if not options:
            raise Blocked
        _, side, walk = min(options)
        paths[f] = (side, go)
        for g in walk:
            paths[g] = (side, OPP[side])
    raise Blocked


def consistent(lattice, paths: dict) -> bool:
    """Every stream runs from the floor's edge to the floor's edge, never
    stopping mid-floor, never doubling back."""
    for j, (come, go) in paths.items():
        if come == go or not live(lattice, paths, j):
            return False
        ahead = lattice.next(j, go)
        if ahead is not None and (ahead not in paths or paths[ahead][0] != OPP[go]):
            return False
    return True


def throw(lattice, paths: dict, j, go: str) -> dict:
    """The map with a switch at `j` turning `go`, re-routed round it.
    Raises Blocked if it can't be done cleanly."""
    paths = dict(paths)
    switch(lattice, paths, j, go)
    _feed(lattice, paths, j, prefer=go)
    if not consistent(lattice, paths):
        raise Blocked
    return paths


def reverses(before: dict, after: dict) -> bool:
    """Whether changing map sends any comet back the way it came: arriving
    at a corner from the way that is now its way out."""
    return any(j in after and after[j][1] == come for j, (come, _) in before.items())


# ---- the run ----------------------------------------------------------------------------------


def _start(ctx) -> None:
    state = ctx.state
    p = ctx.params
    lattice = state["lattice"] = Lattice(ctx.geometry)
    state["axis"] = Axis.X if ctx.rng.random() < 0.5 else Axis.Y
    state["dir"] = 1 if ctx.rng.random() < 0.5 else -1
    state["map"] = lattice.uniform(state["axis"], state["dir"])  # the map for the next segment
    state["on"] = state["map"]  # the map the comets are on now
    state["base"] = ctx.rng.random()
    state["u"] = {}
    along = lattice.cols if state["axis"] is Axis.X else lattice.rows
    for link in _flows(lattice, state["on"]):
        k = link[2]  # how far along the floor, in the flow's axis
        age = k if state["dir"] > 0 else along - 1 - k
        for side in (1, -1):  # as if it had been running: each comet a drift older than the one behind it
            state["u"][(link, side)] = state["base"] - p["drift"] * age + _jitter(ctx)
    state["phase"] = "train"  # train -> build -> unwind -> train
    state["clock"] = 0.0
    state["next"] = p["interval"]
    state["saved"] = []  # the map before each switch
    state["switched"] = []  # (junction, way) of each switch in
    state["offset"] = 0
    state["pulse"] = None
    state["segment"] = None


def _flows(lattice, paths: dict) -> dict:
    """link -> +1 / -1 for every link a stream runs along."""
    flows = {}
    for j, (_, go) in paths.items():
        link = lattice.link(j, go)
        if link is not None:
            flows[link] = OUT[go]
    return flows


def _at_corner(ctx) -> None:
    """The heads are on the corners: change the map if it's time, then plan
    the next tile side."""
    state = ctx.state
    p = ctx.params
    lattice = state["lattice"]
    due = state["clock"] >= state["next"]
    changed = False

    if state["phase"] == "train":
        if not due:
            if ctx.rng.random() < p["turns"]:
                state["axis"] = Axis.Y if state["axis"] is Axis.X else Axis.X
                state["dir"] = 1 if ctx.rng.random() < 0.5 else -1
                state["map"] = lattice.uniform(state["axis"], state["dir"])
                changed = True
        else:
            state["phase"] = "build"
    if state["phase"] == "build" and due:
        if len(state["saved"]) < p["switches"] and _throw(ctx):
            state["next"] = state["clock"] + p["interval"]
            changed = True
        else:
            state["phase"] = "unwind"
    if state["phase"] == "unwind" and due:
        if state["saved"]:
            state["map"] = state["saved"].pop()
            state["switched"].pop()
            changed = True
        state["next"] = state["clock"] + p["interval"]
        if not state["saved"]:
            state["phase"] = "train"
    state["segment"] = _segment(ctx, turn=changed)


def _throw(ctx) -> bool:
    """Throw a switch at a corner where a stream runs straight, re-routing
    the map round it. False if no corner will take one."""
    state = ctx.state
    lattice = state["lattice"]
    paths = state["map"]
    candidates = [
        (j, go)
        for j, (come, out) in paths.items()
        if come == OPP[out] and 0 < j[0] < lattice.rows and 0 < j[1] < lattice.cols  # straight, and not on the floor's edge
        for go in SIDEWAYS[out]
    ]
    ctx.rng.shuffle(candidates)
    for j, go in candidates:
        try:
            new = throw(lattice, paths, j, go)
        except Blocked:
            continue
        if reverses(paths, new) or reverses(new, paths):
            continue  # going in or coming out, comets would have to turn back on themselves
        state["saved"].append(paths)
        state["switched"].append((j, go))
        state["map"] = new
        return True
    return False


def _segment(ctx, turn: bool = False) -> dict:
    """One tile side of travel. Every comet arrives at the corner ahead and
    leaves by its way out in the new map (or ends, off the floor's edge);
    links nothing arrives for get new comets. Paths are 2n flat LEDs, -1
    off the floor."""
    state = ctx.state
    lattice = state["lattice"]
    n = lattice.n
    new = state["map"]
    flows = _flows(lattice, new)
    none = np.full(n, -1)
    paths, u, keys = [], [], []
    fed = set()
    for j, (_, go_was) in state["on"].items():
        link = lattice.link(j, go_was)
        if link is None:
            continue
        ahead = lattice.next(j, go_was)
        come = OPP[go_was]
        go = new[ahead][1] if ahead in new else None
        out = lattice.link(ahead, go) if go is not None and go != come else None
        if out is not None:
            fed.add(out)
        for side in (1, -1):
            before = lattice.leds(link, side, OUT[go_was])
            colour = state["u"].get((link, side), state["base"])
            key = after = None
            if out is not None:
                key = (out, _lane(come, go, side))
                after = lattice.leds(out, key[1], OUT[go])
            if before is not None or after is not None:
                paths.append(np.concatenate([none if before is None else before, none if after is None else after]))
                u.append(colour)
            keys.append((key, colour))
    state["base"] += ctx.params["drift"]
    for link, flow in flows.items():
        if link in fed:
            continue
        for side in (1, -1):  # new comets, coming in at the floor's edge (or where the map changed under nothing)
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
        "flows": flows,
        "born": [link for link in flows if link not in fed],
        "turn": turn,
    }


def _commit(state) -> None:
    """The segment is over: every comet is on its new link."""
    state["u"] = {key: colour for key, colour in state["segment"]["keys"] if key is not None}
    state["on"] = state["map"]
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
