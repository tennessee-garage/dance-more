import os
import time
from pathlib import Path

import numpy as np
import pytest

iio = pytest.importorskip("imageio.v3", reason="video import needs the [preview] extra")

from df2_pi import cli  # noqa: E402
from df2_pi.animation import AnimationRegistry, default_animations_dir  # noqa: E402
from df2_pi.gamma import from_linear  # noqa: E402
from df2_pi.geometry import FloorGeometry  # noqa: E402
from df2_pi.video import clip_path, import_clip, list_clips, load_clip, to_leds  # noqa: E402


@pytest.fixture(scope="module")
def geo() -> FloorGeometry:
    return FloorGeometry()


def gif(path: Path, frames: list[np.ndarray], ms: int = 100) -> Path:
    iio.imwrite(path, np.stack(frames), duration=ms, loop=0)
    return path


def band_frames(count: int, h: int = 64, w: int = 64) -> list[np.ndarray]:
    """A white band moving down a blue frame, one row of travel per frame."""
    frames = []
    for i in range(count):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 2] = 120
        img[(i * 3) % h : (i * 3) % h + 6] = 255
        frames.append(img)
    return frames


# ---- one frame to LEDs ----------------------------------------------------------------------------


def test_the_top_of_the_video_is_the_top_of_the_floor(geo):
    img = np.zeros((90, 90, 3), dtype=np.uint8)
    img[:45] = 255  # top half white
    leds = to_leds(img, geo).max(axis=-1).reshape(geo.tile_rows, geo.tile_cols, -1)
    assert leds[geo.tile_rows - 1].min() == 255  # tile row 7: the far side from the Pi, drawn at the top
    assert leds[0].max() == 0


def test_crop_takes_the_square_it_is_asked_for(geo):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :100, 0] = 255  # left half red
    img[:, 100:, 2] = 255  # right half blue
    left = to_leds(img, geo, crop=(0.25, 0.5, 1.0))
    assert (left[..., 0] == 255).all() and (left[..., 2] == 0).all()
    middle = to_leds(img, geo)  # the default: the middle square, half red half blue
    assert (middle[..., 0] == 255).any() and (middle[..., 2] == 255).any()
    with pytest.raises(ValueError):
        to_leds(img, geo, crop=(0.5, 0.5, 0.0))


def test_frames_are_averaged_in_light(geo):
    checker = (np.indices((272, 272)).sum(axis=0) % 2 * 255).astype(np.uint8)  # 2x2 per cell: half the light
    img = np.repeat(checker[..., None], 3, axis=-1)
    expected = int(from_linear(np.array([0.5]))[0])
    assert expected > 160
    assert np.abs(to_leds(img, geo).astype(int) - expected).max() <= 1  # not 128's darker average of bytes


# ---- importing and loading ------------------------------------------------------------------------


def test_import_samples_at_the_asked_rate_from_start_for_duration(tmp_path):
    source = gif(tmp_path / "band.gif", band_frames(20), ms=100)  # 10 fps, 2 s
    path = import_clip(source, "band", media_dir=tmp_path, fps=5.0)
    clip = load_clip("band", tmp_path)
    assert path == clip_path("band", tmp_path) and clip.fps == 5.0 and len(clip.frames) == 10
    assert clip.frames.shape[1:] == (64, 60, 3) and clip.frames.dtype == np.uint8 and clip.source == "band.gif"
    import_clip(source, "part", media_dir=tmp_path, fps=10.0, start=0.5, duration=1.0)
    assert len(load_clip("part", tmp_path).frames) == 10
    assert list_clips(tmp_path) == ["band", "part"]


def test_a_reimported_clip_is_picked_up(tmp_path):
    source = gif(tmp_path / "band.gif", band_frames(10), ms=100)
    import_clip(source, "band", media_dir=tmp_path, fps=10.0)
    assert len(load_clip("band", tmp_path).frames) == 10
    time.sleep(0.01)
    import_clip(source, "band", media_dir=tmp_path, fps=5.0)
    os.utime(clip_path("band", tmp_path), ns=(time.time_ns(), time.time_ns() + 10**9))
    assert len(load_clip("band", tmp_path).frames) == 5


def test_missing_and_too_short(tmp_path):
    assert load_clip("nope", tmp_path) is None and list_clips(tmp_path / "absent") == []
    with pytest.raises(ValueError):
        import_clip(gif(tmp_path / "one.gif", band_frames(2)), "one", media_dir=tmp_path, fps=1.0)


# ---- the Video animation --------------------------------------------------------------------------


def video_registry(monkeypatch, media: Path) -> AnimationRegistry:
    monkeypatch.setenv("DF2_MEDIA", str(media))
    return AnimationRegistry.discover(default_animations_dir())  # re-runs video.py, which lists the clips


def test_video_lists_the_clips_and_plays_one(tmp_path, monkeypatch):
    import_clip(gif(tmp_path / "band.gif", band_frames(24)), "band", media_dir=tmp_path, fps=10.0)
    registry = video_registry(monkeypatch, tmp_path)
    spec = registry["video"].meta.params["clip"]
    assert list(spec.choices) == ["band"] and spec.default == "band"
    frames = [registry["video"].start(params={"black": 0.0}).render().frame]
    assert frames[0].data.max() > 240  # the band is there


def test_the_loop_has_no_seam(tmp_path, monkeypatch):
    import_clip(gif(tmp_path / "band.gif", band_frames(30)), "band", media_dir=tmp_path, fps=10.0)
    clip = load_clip("band", tmp_path)
    within = max(np.abs(clip.frames[i + 1].astype(int) - clip.frames[i].astype(int)).max() for i in range(len(clip.frames) - 1))
    registry = video_registry(monkeypatch, tmp_path)
    # 3 clip frames per rendered frame at 30 fps: through the loop point several times
    run = registry["video"].start(params={"speed": 1.0, "loop_fade": 1.0, "black": 0.0})
    rendered = [run.render().frame.data.astype(int) for _ in range(40)]
    jumps = [np.abs(b - a).max() for a, b in zip(rendered, rendered[1:])]
    assert max(jumps) <= within + 2  # crossing the loop is no bigger a step than the clip's own


def test_rotation_turns_the_picture(tmp_path, monkeypatch):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32] = 255
    other = img.copy()
    other[0, 0] = 254  # a GIF writer merges identical frames; keep two (the change is in the white half)
    import_clip(gif(tmp_path / "top.gif", [img, other]), "top", media_dir=tmp_path, fps=10.0)
    registry = video_registry(monkeypatch, tmp_path)
    upright = registry["video"].start(params={"black": 0.0}).render().frame.data.max(axis=-1).reshape(8, 8, 60)
    turned = registry["video"].start(params={"black": 0.0, "rotation": "90"}).render().frame.data.max(axis=-1).reshape(8, 8, 60)
    assert upright[7].min() == 255 and upright[0].max() == 0
    assert turned[:, 7].min() == 255 and turned[:, 0].max() == 0  # a quarter clockwise: the top goes to the east


def test_with_no_clips_it_shows_a_dim_wash(tmp_path, monkeypatch):
    registry = video_registry(monkeypatch, tmp_path / "empty")
    assert list(registry["video"].meta.params["clip"].choices) == ["(none)"]
    data = registry["video"].start().render().frame.data
    assert 0 < data.max() < 80


# ---- the CLI --------------------------------------------------------------------------------------


def test_cli_imports_and_lists(tmp_path, capsys):
    source = gif(tmp_path / "band.gif", band_frames(20))
    assert cli.main(["video", "--media", str(tmp_path), "import", str(source), "--name", "surf", "--fps", "5", "--crop", "0.5,0.5,0.8"]) == 0
    assert "10 frames at 5 fps" in capsys.readouterr().out
    assert cli.main(["video", "--media", str(tmp_path), "list"]) == 0
    assert "surf" in capsys.readouterr().out
    assert cli.main(["video", "--media", str(tmp_path), "import", str(tmp_path / "missing.mp4")]) == 1
