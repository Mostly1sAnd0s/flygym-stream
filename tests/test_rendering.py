"""Tests for flygym_stream.rendering (PLAN.md Phase 4).

``encode_jpeg`` is pure and runs everywhere. OffscreenRenderer tests need
a real GL context + the fly scene, so they are marked ``gpu`` and skip
unless ``MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`` is exported before pytest
starts (mujoco reads the env at import time).
"""

from __future__ import annotations

import io
import threading

import mujoco as mj
import numpy as np
import pytest
from PIL import Image

from flygym_stream.config import Config
from flygym_stream.rendering import (
    OffscreenRenderer,
    apply_depth_overlay,
    encode_jpeg,
)
from flygym_stream.scene import build_default_scene
from tests.test_scene import make_cam_tiny_scene


@pytest.mark.gpu
def test_front_view_renders_the_flys_front() -> None:
    """End-to-end (GL): the app-convention front view (az 0, el 0) must put
    the camera on the +x side — the fly's FRONT side. There the head (0.2
    ahead of the thorax, r 0.1 at 0.6 m: 18 deg across) completely OCCLUDES
    the thorax (r 0.05 at 0.8 m: 7 deg) which sits collinearly behind it. On
    the -x side (the pre-fix MuJoCo placement) it is the thorax that sits in
    front and the head renders as a ring around it — both blobs visible.
    (Area alone can't discriminate: the head's 2x radius wins either side.)"""
    from mujoco.renderer import Renderer

    s = make_cam_tiny_scene()
    s.set_orbit_camera(azimuth=0.0, elevation=0.0, distance_mm=800.0)
    cam = mj.MjvCamera()
    s.update_mjv_camera(cam)
    ren = Renderer(s.mj_model, height=240, width=320)
    ren.update_scene(s.mj_data, camera=cam)
    img = ren.render().astype(int)
    ren.close()

    def blob_area(mask_fn) -> int:
        return int(mask_fn(img).sum())

    head = blob_area(lambda i: (i[:, :, 1] > 40) & (i[:, :, 0] < 80) & (i[:, :, 2] < 80))
    thorax = blob_area(lambda i: (i[:, :, 0] > 40) & (i[:, :, 1] < 60) & (i[:, :, 2] < 60))
    assert head > 500, f"head not visible (head={head} px)"
    # camera at +x: the head (18 deg) hides the collinear thorax (7 deg)
    assert thorax < 50, (
        f"thorax {thorax} px visible around the head: the camera is on the "
        f"-x side (pre-fix MuJoCo free-camera placement)")


def _test_frame() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (60, 80, 3)).astype(np.uint8)


def test_encode_jpeg_round_trip() -> None:
    blob = encode_jpeg(_test_frame(), quality=80)
    assert blob[:2] == b"\xff\xd8"  # JPEG magic
    with Image.open(io.BytesIO(blob)) as img:
        assert img.size == (80, 60)  # (width, height)
        assert img.mode == "RGB"


def test_encode_jpeg_quality_affects_size() -> None:
    frame = _test_frame()
    assert len(encode_jpeg(frame, quality=95)) > len(encode_jpeg(frame, quality=10))


def test_encode_jpeg_rejects_bad_quality() -> None:
    with pytest.raises(ValueError):
        encode_jpeg(_test_frame(), quality=0)
    with pytest.raises(ValueError):
        encode_jpeg(_test_frame(), quality=101)


def test_apply_depth_overlay_shape_dtype_and_content() -> None:
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, (48, 64, 3)).astype(np.uint8)
    depth = rng.uniform(0.1, 10.0, (48, 64)).astype(np.float32)
    out = apply_depth_overlay(rgb, depth)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, rgb)  # the overlay tints depth pixels


def test_apply_depth_overlay_background_untouched() -> None:
    """Zero depth = no geometry: those pixels must be left as-is."""
    rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
    depth = np.zeros((8, 8), dtype=np.float32)
    out = apply_depth_overlay(rgb, depth)
    assert np.array_equal(out, rgb)


def test_apply_depth_overlay_rejects_bad_inputs() -> None:
    rgb = _test_frame()
    with pytest.raises(ValueError):
        apply_depth_overlay(rgb, np.zeros((59, 79), dtype=np.float32))
    with pytest.raises(ValueError):
        apply_depth_overlay(rgb, _test_frame())  # wrong dtype


@pytest.fixture(scope="module")
def gpu_scene_and_renderer():
    """Fly scene + OffscreenRenderer at 320x240 (GL required)."""
    scene = build_default_scene()
    renderer = OffscreenRenderer(scene, Config(width=320, height=240, output_fps=30))
    yield scene, renderer
    renderer.close()


@pytest.mark.gpu
def test_offscreen_renderer_renders_fly(gpu_scene_and_renderer) -> None:
    scene, renderer = gpu_scene_and_renderer
    for _ in range(30):  # advance sim time past the first render tick
        scene.step()
    assert renderer.render_as_needed() is True
    frame = renderer.render_frame()
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert float(frame.std()) > 1.0  # non-trivial content
    blob = renderer.encode_jpeg(frame)
    with Image.open(io.BytesIO(blob)) as img:
        assert img.size == (320, 240)


@pytest.mark.gpu
def test_offscreen_renderer_orbit_changes_view(gpu_scene_and_renderer) -> None:
    scene, renderer = gpu_scene_and_renderer
    scene.step()
    first = renderer.render_frame()
    scene.set_orbit_camera(azimuth=200.0, elevation=10.0, distance_mm=25.0)
    second = renderer.render_frame()
    assert abs(float(first.mean()) - float(second.mean())) > 1.0


@pytest.mark.gpu
def test_offscreen_renderer_reset_makes_frame_due(gpu_scene_and_renderer) -> None:
    scene, renderer = gpu_scene_and_renderer
    for _ in range(300):
        scene.step()
        renderer.render_as_needed()  # consume due frames
    renderer.reset()
    assert renderer.render_as_needed() is True


@pytest.mark.gpu
def test_offscreen_renderer_depth_overlay_changes_pixels() -> None:
    """--overlay depth: the same view renders different pixels (GL)."""
    scene = build_default_scene()
    plain = OffscreenRenderer(
        scene, Config(width=320, height=240, output_fps=30))
    overlaid = OffscreenRenderer(
        scene, Config(width=320, height=240, output_fps=30, overlay="depth"))
    try:
        for _ in range(30):
            scene.step()
        assert plain.render_as_needed() is True
        base = plain.render_frame()
        assert overlaid.render_as_needed() is True
        with_depth = overlaid.render_frame()
        diff = float(np.abs(base.astype(int) - with_depth.astype(int)).mean())
        assert diff > 1.0, diff
        print(f"\ndepth overlay: mean pixel diff {diff:.1f}")
    finally:
        plain.close()
        overlaid.close()


@pytest.mark.gpu
def test_offscreen_renderer_cross_thread_like_serve() -> None:
    """The serve() pattern: built on the main thread, rendered on the sim thread.

    EGL contexts bind to the creating thread; OffscreenRenderer must create
    the GL context lazily on the first render so eglMakeCurrent never runs
    on a foreign thread (EGL_BAD_ACCESS).
    """
    scene = build_default_scene()
    renderer = OffscreenRenderer(
        scene, Config(width=320, height=240, output_fps=30))
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(30):
                scene.step()
            assert renderer.render_as_needed() is True
            frame = renderer.render_frame()
            assert frame.shape == (240, 320, 3)
        except BaseException as err:  # noqa: BLE001 - surfaced below
            errors.append(err)

    worker = threading.Thread(target=work, name="render-test")
    worker.start()
    worker.join(timeout=120)
    renderer.close()
    assert not worker.is_alive()
    assert not errors, errors
