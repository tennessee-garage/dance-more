import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from df2_pi.animation import AnimationRegistry
from df2_pi.engine import Runner
from df2_pi.output import FanOut, NullSink, PreviewSink
from df2_pi.web import AppContext, create_app

EVERY_PARAM = '''
from df2_pi.animation import animation, Param
from df2_pi.pixels import TileFrame

@animation(
    name="Every Param",
    description="One of each kind.",
    author="test",
    format="tile",
    tags=["test", "flat"],
    params={
        "level": Param(int, default=10, min=0, max=255, label="Level", help="How bright", role="intensity", macro=2),
        "speed": Param(float, default=1.5, min=0.1, max=5.0, curve="log"),
        "mode": Param(str, default="up", choices=["up", "down"]),
        "wobble": Param(bool, default=False),
    },
    period=2.0,
    sync="beat",
    triggers=True,
    preview_hint="loop",
    energy="low",
    weights=[1, 2, 3],
    odd={1, 2},
)
def render(previous, ctx):
    return TileFrame.black(ctx.geometry)
'''

PIXEL = '''
from df2_pi.animation import animation
from df2_pi.pixels import PixelFrame

@animation(name="Pixel", format="pixel")
def render(previous, ctx):
    return PixelFrame.black(ctx.geometry)
'''

BROKEN = "def render(:\n"


def write(directory: Path, name: str, source: str, mtime: float | None = None) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(source))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def anim_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animations"
    d.mkdir()
    write(d, "every.py", EVERY_PARAM)
    write(d, "pixel.py", PIXEL)
    write(d, "broken.py", BROKEN)
    return d


def client_for(registry: AnimationRegistry) -> TestClient:
    runner = MagicMock(spec=Runner)
    preview = PreviewSink()
    return TestClient(create_app(AppContext(registry, None, FanOut([NullSink(), preview]), runner, preview)))


@pytest.fixture
def registry(anim_dir) -> AnimationRegistry:
    return AnimationRegistry.discover(anim_dir)


@pytest.fixture
def client(registry) -> TestClient:
    return client_for(registry)


# ---- serialisation --------------------------------------------------------------------------


def test_every_field_of_an_animation_is_served(client):
    body = client.get("/api/animations").json()
    every = next(a for a in body["animations"] if a["id"] == "every")
    keys = ("name", "description", "author", "format", "tags", "period", "sync", "triggers", "path", "error")
    assert {k: every[k] for k in keys} == {
        "name": "Every Param",
        "description": "One of each kind.",
        "author": "test",
        "format": "tile",
        "tags": ["test", "flat"],
        "period": 2.0,
        "sync": "beat",
        "triggers": True,
        "path": "every.py",
        "error": None,
    }


def test_every_param_field_and_type_name_is_serialised(client):
    params = client.get("/api/animations/every").json()["params"]
    assert params == {
        "level": {
            "type": "int", "default": 10, "min": 0, "max": 255, "choices": None, "label": "Level", "help": "How bright",
            "role": "intensity", "macro": 2, "curve": "linear",
        },
        "speed": {  # a role by its name
            "type": "float", "default": 1.5, "min": 0.1, "max": 5.0, "choices": None, "label": None, "help": None,
            "role": "speed", "macro": None, "curve": "log",
        },
        "mode": {
            "type": "str", "default": "up", "min": None, "max": None, "choices": ["up", "down"], "label": None, "help": None,
            "role": None, "macro": None, "curve": "linear",
        },
        "wobble": {
            "type": "bool", "default": False, "min": None, "max": None, "choices": None, "label": None, "help": None,
            "role": None, "macro": None, "curve": "linear",
        },
    }


def test_extra_is_served_verbatim_and_never_breaks_the_response(client):
    extra = client.get("/api/animations/every").json()["extra"]
    assert extra["preview_hint"] == "loop" and extra["energy"] == "low"
    assert extra["weights"] == [1, 2, 3]
    assert extra["odd"] == repr({1, 2})  # JSON has no set: its repr, not a 500


def test_animations_are_listed_by_name(client):
    names = [a["name"] for a in client.get("/api/animations").json()["animations"]]
    assert names == ["Every Param", "Pixel"]


def test_no_absolute_path_is_ever_served(anim_dir, tmp_path):
    """Paths are relative to their directory; a duplicate id's message,
    which names the first file by its absolute path, is scrubbed too."""
    other = tmp_path / "more"
    other.mkdir()
    write(other, "pixel.py", PIXEL)  # a duplicate of anim_dir/pixel.py
    client = client_for(AnimationRegistry.discover([anim_dir, other]))
    text = client.get("/api/animations").text
    assert str(tmp_path) not in text and str(tmp_path.resolve()) not in text
    body = client.get("/api/animations").json()
    assert body["errors"]["pixel"]["path"] == "pixel.py"
    assert "duplicate" in body["errors"]["pixel"]["message"]


# ---- errors ---------------------------------------------------------------------------------


def test_load_errors_carry_stage_message_and_path(client):
    errors = client.get("/api/animations").json()["errors"]
    assert set(errors) == {"broken"}
    assert errors["broken"]["stage"] == "syntax"
    assert "SyntaxError" in errors["broken"]["message"]
    assert errors["broken"]["path"] == "broken.py"


def test_a_failed_id_is_404_with_the_error_in_the_body(client):
    response = client.get("/api/animations/broken")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["stage"] == "syntax" and "SyntaxError" in body["error"]["message"]
    assert "broken" in body["detail"]


def test_an_unknown_id_is_404_without_an_error(client):
    response = client.get("/api/animations/nope")
    assert response.status_code == 404
    assert "error" not in response.json()


# ---- reload ---------------------------------------------------------------------------------


def test_reload_reports_nothing_when_nothing_changed(client):
    assert client.post("/api/animations/reload").json() == {
        "changed": [],
        "errors": client.get("/api/animations").json()["errors"],
    }


def test_a_fixed_file_leaves_errors_after_reload(anim_dir, client):
    write(anim_dir, "broken.py", PIXEL.replace('"Pixel"', '"Mended"'), mtime=2e9)
    body = client.post("/api/animations/reload").json()
    assert body["changed"] == ["broken"]
    assert body["errors"] == {}
    listed = client.get("/api/animations").json()
    assert "broken" not in listed["errors"]
    assert "Mended" in [a["name"] for a in listed["animations"]]


def test_a_failed_reload_keeps_the_last_good_version_with_its_error(anim_dir, client):
    write(anim_dir, "pixel.py", BROKEN, mtime=2e9)
    body = client.post("/api/animations/reload").json()
    assert body["changed"] == ["pixel"] and body["errors"]["pixel"]["stage"] == "syntax"
    pixel = client.get("/api/animations/pixel")
    assert pixel.status_code == 200  # the last good version is still live
    assert pixel.json()["name"] == "Pixel"
    assert pixel.json()["error"]["stage"] == "syntax"


def test_a_removed_file_leaves_the_list(anim_dir, client):
    (anim_dir / "pixel.py").unlink()
    assert client.post("/api/animations/reload").json()["changed"] == ["pixel"]
    assert [a["id"] for a in client.get("/api/animations").json()["animations"]] == ["every"]


# ---- OpenAPI --------------------------------------------------------------------------------


def test_openapi_describes_the_responses(client):
    schema = client.get("/openapi.json").json()
    assert {"AnimationList", "AnimationInfo", "ParamSpec", "LoadErrorInfo", "Reloaded"} <= set(schema["components"]["schemas"])
    paths = schema["paths"]
    assert paths["/api/animations"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AnimationList"
    }
    assert "404" in paths["/api/animations/{animation_id}"]["get"]["responses"]
    assert "post" in paths["/api/animations/reload"]
