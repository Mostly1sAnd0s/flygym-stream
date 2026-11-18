"""Windows preview is opt-in; the Linux deployment still requires EGL."""
import os
from unittest.mock import patch

import pytest

from flygym_stream.cli import require_gl_env


def test_windows_opt_in_sets_context_before_import():
    with patch('sys.platform', 'win32'), patch.dict(os.environ, {'PYOPENGL_PLATFORM': 'egl'}):
        require_gl_env(local_windows=True)
        assert os.environ['MUJOCO_GL'] == 'glfw'
        assert 'PYOPENGL_PLATFORM' not in os.environ


def test_linux_rejects_windows_option():
    with patch('sys.platform', 'linux'), pytest.raises(SystemExit, match='Windows testing'):
        require_gl_env(local_windows=True)


def test_default_still_requires_egl():
    with patch.dict(os.environ, {}, clear=True), pytest.raises(SystemExit, match='egl'):
        require_gl_env()
