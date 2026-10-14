"""Tests for flygym_stream.cli (PLAN.md Phase 4: render-preview DoD)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from flygym_stream.cli import require_gl_env


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
