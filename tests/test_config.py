"""Tests for flygym_stream.config.Config (PLAN.md §3 Configuration)."""

from __future__ import annotations

import dataclasses

import pytest

from flygym_stream.config import Config


def test_defaults_match_plan() -> None:
    cfg = Config()
    assert cfg.width == 1280
    assert cfg.height == 720
    assert cfg.output_fps == 30
    assert cfg.jpeg_quality == 80
    assert cfg.playback_speed == 1.0
    assert cfg.bind == "127.0.0.1"
    assert cfg.port == 8765
    assert cfg.backend == "cpu"
    assert cfg.scene == "fly_flat_ground"


def test_camera_res_is_height_width() -> None:
    # MuJoCo / flygym convention: (rows, cols) == (height, width).
    assert Config(width=640, height=360).camera_res == (360, 640)


def test_config_is_immutable() -> None:
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.width = 10  # type: ignore[misc]


@pytest.mark.parametrize("kwargs", [
    {"width": 0},
    {"height": -1},
    {"output_fps": 0},
    {"jpeg_quality": 0},
    {"jpeg_quality": 101},
    {"playback_speed": 0.0},
    {"playback_speed": -2.0},
    {"port": 0},
    {"port": 70000},
    {"backend": "tpu"},
    {"scene": "unknown_scene"},
])
def test_invalid_values_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_valid_alternatives_accepted() -> None:
    for backend in ("cpu", "gpu", "auto"):
        cfg = Config(backend=backend, width=320, height=240, output_fps=10,
                     jpeg_quality=1, playback_speed=0.5, port=1)
        assert cfg.backend == backend
