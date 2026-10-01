from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from df2_pi.animation import AnimationRegistry
from df2_pi.engine import Runner
from df2_pi.interfacing.beat import LinkSource, LinkUnavailable
from df2_pi.interfacing.beat_service import BeatService
from df2_pi.output import FanOut, NullSink, PreviewSink
from df2_pi.playlists import PlaylistStore
from df2_pi.web import AppContext, create_app


class Now:
    def __init__(self) -> None:
        self.t = 500.0

    def __call__(self) -> float:
        return self.t


class FakeLink:
    beat, tempo, num_peers, quantum = 8.0, 126.0, 2, 4


@pytest.fixture
def registry(tmp_path: Path) -> AnimationRegistry:
    (tmp_path / "animations").mkdir()
    return AnimationRegistry.discover(tmp_path / "animations")


@pytest.fixture
def store(registry):
    s = PlaylistStore(":memory:", registry=registry)
    yield s
    s.close()


def fake_link(quantum, now):
    return LinkSource(quantum=quantum, now=now, link=FakeLink())


def no_link(quantum, now):
    raise LinkUnavailable("aalink is not installed: pip install -e '.[beat]'")


def build(registry, store, link_factory=fake_link, with_beat=True):
    runner = MagicMock(spec=Runner)
    runner.state = Runner(registry, FanOut([NullSink()])).state
    now = Now()
    beat = BeatService(runner, store, now=now, link_factory=link_factory) if with_beat else None
    preview = PreviewSink()
    ctx = AppContext(registry, store, FanOut([NullSink(), preview]), runner, preview, None, beat)
    return TestClient(create_app(ctx)), beat, runner, now


def test_defaults_attach_the_clock_and_leave_launches_alone(registry, store):
    client, beat, runner, _ = build(registry, store)
    body = client.get("/api/beat").json()
    assert body["settings"] == {
        "source": "off", "beats_per_bar": 4, "multiplier": 1.0, "offset_ms": 0.0, "launch_quantum": "off", "fallback_bpm": 120.0,
    }
    assert body["status"]["active"] is False
    runner.attach_beat.assert_called_once_with(beat.clock)
    runner.set_launch_quantum.assert_called_once_with("off")
    runner.set_fallback_bpm.assert_called_once_with(120.0)


def test_the_fallback_tempo_is_applied_and_stored(registry, store):
    client, _, runner, _ = build(registry, store)
    assert client.patch("/api/beat", json={"fallback_bpm": 128}).status_code == 200
    runner.set_fallback_bpm.assert_called_with(128.0)
    assert store.get_float("fallback_bpm") == 128.0
    assert client.patch("/api/beat", json={"fallback_bpm": 10}).status_code == 422


def test_a_change_is_applied_now_and_stored(registry, store):
    client, beat, runner, _ = build(registry, store)
    body = client.patch("/api/beat", json={"source": "link", "beats_per_bar": 3, "multiplier": 2.0, "offset_ms": 40, "launch_quantum": "bar"}).json()
    assert body["settings"]["source"] == "link" and body["status"]["peers"] == 2
    assert (beat.clock.beats_per_bar, beat.clock.multiplier, beat.clock.offset_ms) == (3, 2.0, 40.0)
    assert beat.clock.source.name == "link"
    runner.set_launch_quantum.assert_called_with("bar")
    assert (store.get_str("beat_source"), store.get_int("beats_per_bar"), store.get_float("beat_multiplier")) == ("link", 3, 2.0)
    assert (store.get_float("beat_offset_ms"), store.get_str("launch_quantum")) == (40.0, "bar")


def test_bad_values_change_nothing(registry, store):
    client, beat, _, _ = build(registry, store)
    for body in ({"beats_per_bar": 0}, {"multiplier": 3}, {"offset_ms": 900}, {"source": "midi"}, {"launch_quantum": "phrase"}, {"tempo": 120}):
        assert client.patch("/api/beat", json=body).status_code == 422, body
    assert beat.settings()["beats_per_bar"] == 4 and store.get_int("beats_per_bar") == 4


def test_settings_survive_a_restart(registry, store):
    build(registry, store)[0].patch("/api/beat", json={"source": "tap", "launch_quantum": "beat"})
    _, beat, runner, _ = build(registry, store)
    assert beat.settings()["source"] == "tap"
    runner.set_launch_quantum.assert_called_once_with("beat")


def test_tap_tempo_through_the_api(registry, store):
    client, _, _, now = build(registry, store)
    client.patch("/api/beat", json={"source": "tap"})
    for t in (500.0, 500.5, 501.0, 501.5):
        now.t = t
        body = client.post("/api/beat/tap").json()
    assert body["status"]["taps"] == 4
    client.app.state.ctx.beat.clock.info(now.t)
    assert client.get("/api/beat").json()["status"]["tempo"] == pytest.approx(120.0)


def test_resync_and_nudge_reach_the_clock(registry, store):
    client, beat, _, _ = build(registry, store)
    beat.clock.resync = MagicMock()
    beat.clock.nudge = MagicMock()
    assert client.post("/api/beat/resync").status_code == 200
    assert client.post("/api/beat/nudge", json={"ms": -15}).status_code == 200
    beat.clock.resync.assert_called_once_with()
    beat.clock.nudge.assert_called_once_with(-15.0)


def test_link_unavailable_is_reported_not_raised(registry, store):
    client, beat, _, _ = build(registry, store, link_factory=no_link)
    body = client.patch("/api/beat", json={"source": "link"}).json()
    assert body["settings"]["source"] == "link"
    assert "aalink" in body["status"]["error"] and body["status"]["active"] is False
    assert beat.clock.source is None


def test_without_beat_sync_the_routes_answer_503(registry, store):
    client = build(registry, store, with_beat=False)[0]
    assert client.get("/api/beat").status_code == 503
    assert client.post("/api/beat/tap").status_code == 503
