import dataclasses
import json
import math
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from df2_pi.animation import AnimationRegistry
from df2_pi.engine import FrameClock, Runner, RunnerState
from df2_pi.output import FanOut, NullSink, PreviewSink
from df2_pi.playlists import PlaylistStore
from df2_pi.web import AppContext, create_app

SOLID = '''
from df2_pi.animation import animation, Param
from df2_pi.pixels import TileFrame

@animation(name="Solid", params={
    "level": Param(int, default=10, min=0, max=255),
    "mode": Param(str, default="up", choices=["up", "down"]),
})
def render(previous, ctx):
    frame = TileFrame.black(ctx.geometry)
    frame.data[:] = (ctx.params["level"], min(255, ctx.frame), 0)
    return frame
'''


class FakeTime:
    def __init__(self) -> None:
        self.t = 100.0

    def now(self) -> float:
        self.t += 1e-6
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class Raises:
    """A sink that always raises, so the fan-out detaches it with a reason."""

    name = "flaky"

    def submit(self, frame, info, effects=None):
        raise RuntimeError("cable pulled")

    def close(self):
        pass


class Probe:
    """Calls `on_frame(n)` from inside each submit and stops the runner
    after `stop_after` frames; records the state each frame saw."""

    name = "probe"

    def __init__(self, runner_ref: list, stop_after: int, on_frame=None) -> None:
        self.runner_ref = runner_ref
        self.stop_after = stop_after
        self.on_frame = on_frame
        self.states: list[RunnerState] = []

    def submit(self, frame, info, effects=None):
        runner = self.runner_ref[0]
        self.states.append(runner.state)
        if self.on_frame is not None:
            self.on_frame(info.n)
        if len(self.states) >= self.stop_after:
            runner.stop()

    def latch(self):
        pass

    def close(self):
        pass


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    d = tmp_path / "animations"
    d.mkdir()
    (d / "solid.py").write_text(textwrap.dedent(SOLID))
    return AnimationRegistry.discover(d)


@pytest.fixture
def store(registry) -> PlaylistStore:
    s = PlaylistStore(":memory:", registry=registry)
    pl = s.create_playlist("Party")
    s.add_entry(pl, "solid", duration_s=10.0)
    s.add_entry(pl, "solid", duration_s=10.0)
    yield s
    s.close()


def fake_clock() -> FrameClock:
    fake = FakeTime()
    return FrameClock(fps=10.0, spin_margin=0.0, now=fake.now, sleep=fake.sleep)


def real_state(registry) -> RunnerState:
    """A snapshot from a runner that has actually run, so nothing in it is a
    placeholder."""
    return Runner(registry, FanOut([NullSink()]), clock=fake_clock()).state


@pytest.fixture
def mocked(registry, store):
    """An app whose runner is a mock; the lifespan never runs (no `with`)."""
    runner = MagicMock(spec=Runner)
    runner.state = real_state(registry)
    preview = PreviewSink()
    ctx = AppContext(registry, store, FanOut([NullSink(), preview]), runner, preview)
    return TestClient(create_app(ctx)), runner


# ---- routes -> runner calls -------------------------------------------------------------------


@pytest.mark.parametrize(
    "route, body, method, args",
    [
        ("play", None, "play", ()),
        ("pause", None, "pause", ()),
        ("resume", None, "resume", ()),
        ("next", None, "next", ()),
        ("previous", None, "previous", ()),
        ("restart", None, "restart", ()),
        ("goto", {"index": 3}, "goto", (3,)),
        ("brightness", {"value": 200}, "set_brightness", (200,)),
        ("blackout", {"on": True}, "blackout", ()),
        ("blackout", {"on": False}, "unblackout", ()),
    ],
)
def test_route_maps_to_runner_call(mocked, route, body, method, args):
    client, runner = mocked
    response = client.post(f"/api/transport/{route}", json=body)
    assert response.status_code == 200, response.text
    getattr(runner, method).assert_called_once_with(*args)
    assert response.json()["queued"] is True
    assert response.json()["state"]["frame"] == runner.state.frame


@pytest.mark.parametrize("key", ["Party", 1])
def test_load_by_name_or_id_passes_the_resolved_playlist(mocked, key):
    client, runner = mocked
    response = client.post("/api/transport/load", json={"playlist": key})
    assert response.status_code == 200, response.text
    (resolved,), _ = runner.load_playlist.call_args
    assert resolved.playlist.name == "Party" and len(resolved.entries) == 2


def test_animation_coerces_params_through_the_specs(mocked):
    client, runner = mocked
    response = client.post("/api/transport/animation", json={"id": "solid", "params": {"level": 7.0}, "hold": 30})
    assert response.status_code == 200, response.text
    runner.play_animation.assert_called_once_with("solid", {"level": 7, "mode": "up"}, 30.0)
    (_, params, _), _ = runner.play_animation.call_args
    assert type(params["level"]) is int


def test_animation_hold_defaults_to_until_next(mocked):
    client, runner = mocked
    assert client.post("/api/transport/animation", json={"id": "solid"}).status_code == 200
    runner.play_animation.assert_called_once_with("solid", {"level": 10, "mode": "up"}, None)


# ---- validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route, body, status, fragment",
    [
        ("goto", {"index": -1}, 422, "greater than or equal to 0"),
        ("goto", {}, 422, "Field required"),
        ("brightness", {"value": 256}, 422, "less than or equal to 255"),
        ("brightness", {"value": -1}, 422, "greater than or equal to 0"),
        ("blackout", {"on": "sometimes"}, 422, "valid boolean"),
        ("load", {"playlist": "Nope"}, 404, "Nope"),
        ("load", {"playlist": 99}, 404, "99"),
        ("animation", {"id": "nope"}, 404, "no animation 'nope'"),
        ("animation", {"id": "solid", "params": {"level": 300}}, 422, "300 is above the maximum 255"),
        ("animation", {"id": "solid", "params": {"mode": "sideways"}}, 422, "is not one of ['up', 'down']"),
        ("animation", {"id": "solid", "params": {"speed": 2}}, 422, "unknown parameter 'speed'"),
        ("animation", {"id": "solid", "hold": 0}, 422, "greater than 0"),
    ],
)
def test_invalid_input_is_rejected_and_never_reaches_the_runner(mocked, route, body, status, fragment):
    client, runner = mocked
    response = client.post(f"/api/transport/{route}", json=body)
    assert response.status_code == status, response.text
    assert fragment in response.text
    assert not [c for c in runner.method_calls if c[0] != "state"], runner.method_calls


def test_load_without_a_store_is_404(registry):
    runner = MagicMock(spec=Runner)
    runner.state = real_state(registry)
    preview = PreviewSink()
    client = TestClient(create_app(AppContext(registry, None, FanOut([preview]), runner, preview)))
    assert client.post("/api/transport/load", json={"playlist": "Party"}).status_code == 404
    runner.load_playlist.assert_not_called()


# ---- state --------------------------------------------------------------------------------


def test_state_round_trips_a_real_runner_state(registry, store):
    """Telemetry percentiles and phases, sink health including a detached
    sink's reason, and the (id, name) pairs all survive JSON."""
    ref: list = []
    probe = Probe(ref, stop_after=12)
    preview = PreviewSink()
    fanout = FanOut([probe, NullSink(), preview, Raises()], max_failures=3)
    runner = Runner(registry, fanout, store=store, clock=fake_clock())
    ref.append(runner)
    runner.load_playlist("Party")
    runner.run()
    state = probe.states[-1]
    assert state.timing.frames > 0 and state.timing.phases, "the runner never produced telemetry"
    assert state.sinks["flaky"]["attached"] is False

    stub = MagicMock(spec=Runner)
    stub.state = state
    client = TestClient(create_app(AppContext(registry, store, fanout, stub, preview)))
    body = client.get("/api/state").json()

    assert body["playlist"] == [state.playlist[0], "Party"]
    assert body["animation"] == ["solid", "Solid"]
    assert body["timing"]["jitter_ms"]["p95"] == pytest.approx(state.timing.jitter_ms.p95)
    assert set(body["timing"]["phases"]) == set(state.timing.phases)
    assert body["sinks"]["flaky"] == {"attached": False, "reason": "RuntimeError: cable pulled"}
    assert body["sinks"]["preview"]["attached"] is True
    assert "pixels" not in body and "frame_data" not in body
    assert body == json.loads(TypeAdapter(RunnerState).dump_json(state))


def test_state_serialises_empty_percentiles_as_null(registry, mocked):
    """A clock with no frames yet has NaN percentiles; JSON has no NaN."""
    client, runner = mocked
    assert math.isnan(runner.state.timing.jitter_ms.p95)
    response = client.get("/api/state")
    assert response.status_code == 200
    assert response.json()["timing"]["jitter_ms"]["p95"] is None


def test_a_queued_command_shows_in_the_state_on_the_next_tick(registry, store):
    """The POST's own response predates the command; the next frame's
    snapshot has it."""
    ref: list = []
    responses: dict[int, dict] = {}
    client: list[TestClient] = []

    def on_frame(n: int) -> None:
        if n == 3:
            responses[n] = client[0].post("/api/transport/pause").json()

    probe = Probe(ref, stop_after=6, on_frame=on_frame)
    preview = PreviewSink()
    fanout = FanOut([probe, preview])
    runner = Runner(registry, fanout, store=store, clock=fake_clock())
    ref.append(runner)
    client.append(TestClient(create_app(AppContext(registry, store, fanout, runner, preview))))
    runner.load_playlist("Party")
    runner.run()

    assert responses[3]["queued"] is True
    assert responses[3]["state"]["paused"] is False  # queued, not yet applied
    paused = [s.paused for s in probe.states]
    assert paused[:4] == [False] * 4 and all(paused[4:])


# ---- brightness is persisted -----------------------------------------------------------------


def test_the_brightness_command_is_also_stored(mocked, store):
    client, runner = mocked
    assert client.post("/api/transport/brightness", json={"value": 77}).status_code == 200
    runner.set_brightness.assert_called_once_with(77)
    assert store.get_int("brightness") == 77
    assert client.get("/api/settings").json()["brightness"] == 77


def test_a_refused_brightness_is_not_stored(mocked, store):
    client, _ = mocked
    assert client.post("/api/transport/brightness", json={"value": 300}).status_code == 422
    assert store.get_setting("brightness") is None


# ---- live params ----------------------------------------------------------------------------


def playing_solid(runner) -> None:
    """Make the mocked runner report `solid` as what is playing."""
    runner.state = dataclasses.replace(runner.state, animation=("solid", "Solid"))


def test_params_reach_set_params_coerced(mocked):
    client, runner = mocked
    playing_solid(runner)
    response = client.post("/api/transport/params", json={"level": 7.0, "mode": "down"})
    assert response.status_code == 200, response.text
    runner.set_params.assert_called_once_with(level=7, mode="down")
    assert type(runner.set_params.call_args.kwargs["level"]) is int


@pytest.mark.parametrize(
    "body, param, fragment",
    [
        ({"speed": 2}, "speed", "solid has no parameter 'speed'"),
        ({"level": 300}, "level", "300 is above the maximum 255"),
        ({"level": -1}, "level", "-1 is below the minimum 0"),
        ({"mode": "sideways"}, "mode", "is not one of ['up', 'down']"),
        ({"level": "loud"}, "level", "is not a valid int"),
        ({"level": 5, "mode": "sideways"}, "mode", "is not one of"),  # one bad value rejects the lot
    ],
)
def test_a_bad_param_is_422_naming_it_and_never_reaches_the_runner(mocked, body, param, fragment):
    client, runner = mocked
    playing_solid(runner)
    response = client.post("/api/transport/params", json=body)
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["param"] == param and fragment in detail["message"]
    runner.set_params.assert_not_called()


def test_params_while_idle_is_409(mocked):
    client, runner = mocked
    runner.state = dataclasses.replace(runner.state, animation=("_idle", "Idle"))
    assert client.post("/api/transport/params", json={"level": 7}).status_code == 409
    runner.set_params.assert_not_called()


def test_a_live_edit_shows_in_state_params_on_the_next_tick(registry, store):
    ref: list = []
    responses: dict[int, dict] = {}
    client: list[TestClient] = []

    def on_frame(n: int) -> None:
        if n == 3:
            responses[n] = client[0].post("/api/transport/params", json={"level": 99}).json()

    probe = Probe(ref, stop_after=6, on_frame=on_frame)
    preview = PreviewSink()
    fanout = FanOut([probe, preview])
    runner = Runner(registry, fanout, store=store, clock=fake_clock())
    ref.append(runner)
    client.append(TestClient(create_app(AppContext(registry, store, fanout, runner, preview))))
    runner.load_playlist("Party")
    runner.run()

    assert responses[3]["state"]["params"]["level"] == 10  # queued, not yet applied
    levels = [s.params["level"] for s in probe.states]
    assert levels[:4] == [10] * 4 and levels[4:] == [99] * (len(levels) - 4)


# ---- OpenAPI ------------------------------------------------------------------------------


def test_openapi_documents_that_commands_are_asynchronous(mocked):
    client, _ = mocked
    schema = client.get("/openapi.json").json()
    assert "next frame boundary" in schema["info"]["description"]
    commands = {path: ops["post"] for path, ops in schema["paths"].items() if path.startswith("/api/transport/")}
    assert len(commands) == 12
    for path, op in commands.items():
        assert "next frame boundary" in op["description"], path
    assert {"RunnerState", "TelemetrySnapshot", "Percentiles"} <= set(schema["components"]["schemas"])


def test_openapi_says_percentiles_can_be_null(mocked):
    """What an empty window actually sends; a generated client (#71) must
    not reject it."""
    client, _ = mocked
    percentiles = client.get("/openapi.json").json()["components"]["schemas"]["Percentiles"]["properties"]
    for name in ("p50", "p95", "max"):
        assert {"type": "null"} in percentiles[name]["anyOf"], name
    assert percentiles["count"]["type"] == "integer"
