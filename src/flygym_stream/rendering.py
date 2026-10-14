"""Headless frame pipeline (PLAN.md Phase 4): render -> encode.

``OffscreenRenderer`` reuses ``flygym.rendering.Renderer`` for scene setup
and sim-time pacing (DRY, PLAN.md §2.2); the orbit camera is the
persistent ``MjvCamera`` filled by ``Scene.update_mjv_camera`` (see
scene.py's docstring for why). The GL device is selected by the
``MUJOCO_GL``/``PYOPENGL_PLATFORM`` env vars — the mujoco 3.x Python
bindings take no device kwarg — and the CLI asserts they are set
(PLAN.md §6), so ``serve``/``render-preview`` refuse to start otherwise.
"""

from __future__ import annotations

import io

import mujoco as mj
import numpy as np
from PIL import Image

import flygym.rendering as flygym_rendering

from flygym_stream.config import Config
from flygym_stream.scene import Scene


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    """Encode an RGB uint8 frame as JPEG; returns the JPEG bytes.

    Args:
        frame: ``(height, width, 3)`` uint8 RGB array.
        quality: JPEG quality, 1-100.

    Raises:
        ValueError: If quality is out of range or frame shape/dtype wrong.
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be 1..100, got {quality}")
    if frame.shape[2:] != (3,) or frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB frame, got {frame.shape}/{frame.dtype}")
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class OffscreenRenderer:
    """Fly scene -> RGB frame -> JPEG bytes, paced by sim time.

    Args:
        scene: Scene with a ``camera_name`` (the model camera used for
            flygym Renderer's pacing pass) and orbit state for the view
            actually streamed.
        config: Resolution / fps / jpeg quality (PLAN.md §3 Config).
    """

    def __init__(self, scene: Scene, config: Config) -> None:
        if scene.camera_name is None:
            raise ValueError("Scene.camera_name is required for OffscreenRenderer")
        self._scene = scene
        self._config = config
        # The flygym Renderer (and its EGL context) is created lazily on the
        # first render call: EGL contexts bind to the creating thread, so
        # creating it where rendering happens (the sim thread in serve) is
        # what keeps eglMakeCurrent from failing cross-thread (EGL_BAD_ACCESS).
        self._renderer: flygym_rendering.Renderer | None = None
        self._mjv_cam = mj.MjvCamera()

    def _ensure_renderer(self) -> flygym_rendering.Renderer:
        if self._renderer is None:
            self._renderer = flygym_rendering.Renderer(
                self._scene.mj_model,
                self._scene.camera_name,
                camera_res=self._config.camera_res,
                playback_speed=self._config.playback_speed,
                output_fps=self._config.output_fps,
                buffer_frames=False,
                render_rgb=True,
            )
        return self._renderer

    def render_as_needed(self) -> bool:
        """Sim-time pacing (flygym Renderer); True when a frame is due."""
        return self._ensure_renderer().render_as_needed(self._scene.mj_data)

    def render_frame(self) -> np.ndarray:
        """Render the current orbit-camera view (call after pacing tick)."""
        renderer = self._ensure_renderer()
        self._scene.update_mjv_camera(self._mjv_cam)
        renderer.mj_renderer.update_scene(
            self._scene.mj_data, self._mjv_cam, renderer.scene_option
        )
        return renderer.mj_renderer.render()

    def encode_jpeg(self, frame: np.ndarray) -> bytes:
        """Encode at the configured JPEG quality."""
        return encode_jpeg(frame, self._config.jpeg_quality)

    def reset(self) -> None:
        """Reset the render timer (the next frame becomes immediately due)."""
        self._ensure_renderer().reset()

    def close(self) -> None:
        """Release the underlying MuJoCo GL resources (if ever created)."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "OffscreenRenderer":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.close()
        return False  # don't suppress exceptions
