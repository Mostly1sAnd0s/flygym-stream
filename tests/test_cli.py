"""Tests for flygym_stream.cli (PLAN.md Phases 4-5)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request

import pytest

from flygym_stream.cli import require_gl_env

GL_ENV = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}


def test_require_gl_env_refuses_without_egl(monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        require_gl_env()
    assert "MUJOCO_GL" in str(exc_info.value)


def test_require_gl_env_accepts_egl(monkeypatch) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    require_gl_env()  # must not raise


@pytest.mark.gpu
def test_render_preview_cli_end_to_end(tmp_path) -> None:
    """The Phase 4 DoD: headless MP4 of the fly, produced by the CLI."""
    out = tmp_path / "preview.mp4"
    proc = subprocess.run(
        [sys.executable, "-m", "flygym_stream.cli", "render-preview",
         "--duration", "0.5", "--width", "320", "--height", "240",
         "--out", str(out)],
        capture_output=True, text=True, timeout=600,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert out.exists()
    assert out.stat().st_size > 5000  # a real, non-empty video
    print(f"\npreview: {out} ({out.stat().st_size} bytes)")


def test_fetch_assets_cli() -> None:
    """fetch-assets warms the one-time S3 cache (idempotent when warm)."""
    proc = subprocess.run(
        [sys.executable, "-m", "flygym_stream.cli", "fetch-assets"],
        capture_output=True, text=True, timeout=600, env=GL_ENV,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "flygym_assets" in proc.stdout


def test_serve_refuses_insecure_bind_without_flag() -> None:
    """0.0.0.0 requires --insecure (PLAN.md §6) — fails before scene build."""
    proc = subprocess.run(
        [sys.executable, "-m", "flygym_stream.cli", "serve",
         "--bind", "0.0.0.0", "--port", "1"],
        capture_output=True, text=True, timeout=60, env=GL_ENV,
    )
    assert proc.returncode != 0
    assert "--insecure" in proc.stderr


@pytest.mark.gpu
def test_serve_cli_smoke() -> None:
    """Phase 5 DoD: serve boots, /healthz OK, WS streams a real JPEG.

    Spawns ``python -m flygym_stream.cli serve`` as a subprocess and
    talks to it with a real WebSocket client (uvicorn[standard]
    ships the ``websockets`` package).
    """
    import websockets

    port = 8917
    proc = subprocess.Popen(
        [sys.executable, "-m", "flygym_stream.cli", "serve",
         "--bind", "127.0.0.1", "--port", str(port),
         "--width", "320", "--height", "240"],
        env=GL_ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 180
        while True:
            if time.monotonic() > deadline:
                raise AssertionError(
                    "server never became healthy:\n" + (proc.stdout or "").read()[-2000:])
            try:
                if urllib.request.urlopen(base + "/healthz", timeout=1).status == 200:
                    break
            except OSError:
                time.sleep(0.5)

        async def ws_check() -> tuple[dict, int]:
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
                await ws.send(json.dumps({"type": "hello", "proto": 1}))
                status = json.loads(await asyncio.wait_for(ws.recv(), 30))
                while True:
                    msg = await asyncio.wait_for(ws.recv(), 60)
                    if isinstance(msg, (bytes, bytearray)) \
                            and bytes(msg[:2]) == b"\xff\xd8":
                        return status, len(msg)

        status, frame_len = asyncio.run(ws_check())
        assert status["type"] == "status"
        assert status["n_actuators"] == 42
        assert status["model"] == "fly"
        assert frame_len > 1000  # a real 320x240 JPEG
        print(f"\nserve smoke: status={status} frame={frame_len} bytes")
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_scene_config_from_args_defaults() -> None:
    """serve flag defaults reproduce the historical default scene config
    (PLAN.md Phase 7, M7.1)."""
    import argparse
    from flygym.anatomy import ActuatedDOFPreset
    from flygym.compose import ActuatorType, MeshType
    from flygym_stream.cli import scene_config_from_args

    args = argparse.Namespace(
        dof="legs-active", mesh="fullsize", actuator="position", world="flat",
        spawn=None, vision=False, adhesion=False, musculoskeletal=False,
    )
    cfg = scene_config_from_args(args)
    assert cfg.actuated_preset is ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    assert cfg.mesh_type is MeshType.FULLSIZE
    assert cfg.actuator_type is ActuatorType.POSITION
    assert cfg.world == "flat" and cfg.spawn is None  # keyframe root rest
    assert not (cfg.vision or cfg.adhesion or cfg.musculoskeletal)


def test_scene_config_from_args_flags() -> None:
    import argparse
    from flygym.anatomy import ActuatedDOFPreset
    from flygym.compose import ActuatorType, MeshType
    from flygym_stream.cli import scene_config_from_args

    args = argparse.Namespace(
        dof="all", mesh="simplified", actuator="velocity", world="tethered",
        spawn="1,2,3.5", vision=True, adhesion=True, musculoskeletal=False,
    )
    cfg = scene_config_from_args(args)
    assert cfg.actuated_preset is ActuatedDOFPreset.ALL
    assert cfg.mesh_type is MeshType.SIMPLIFIED_MAX2000FACES
    assert cfg.actuator_type is ActuatorType.VELOCITY
    assert cfg.world == "tethered"
    assert cfg.spawn == (1.0, 2.0, 3.5)
    assert cfg.vision and cfg.adhesion
