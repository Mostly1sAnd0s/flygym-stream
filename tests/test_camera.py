"""Tests for camera views (PLAN.md Phase 7, M7.7): the ``camera`` control
message routing. Scene-level camera behavior (viewcube snaps, body-frame
1st/3rd-person geometry) lives in tests/test_scene.py; the wire protocol
in tests/test_protocol.py. Here: Camera -> Backend -> Scene dispatch and
the SimThread routing (camera is a view concern, not a physics input)."""

from __future__ import annotations

import pytest

from flygym_stream.protocol import Camera, Orbit
from flygym_stream.simulation import MuJoCoBackend
from flygym_stream.stream import InputQueue
from test_scene import make_cam_tiny_scene
from test_simulation import make_sim


def test_camera_mode_message_routes_to_scene() -> None:
    scene = make_cam_tiny_scene()
    b = MuJoCoBackend(scene)
    b.apply_input(Camera(mode="third"))
    assert scene.camera_mode == "third"
    b.apply_input(Camera(mode="orbit"))
    assert scene.camera_mode == "orbit"


def test_camera_view_message_snaps_orbit() -> None:
    scene = make_cam_tiny_scene()
    b = MuJoCoBackend(scene)
    b.apply_input(Camera(view="back"))
    assert scene.camera_mode == "orbit"
    assert scene.orbit[0] == pytest.approx(180.0)  # heading 0 at rest


def test_orbit_message_returns_to_orbit_mode() -> None:
    """A manual orbit always means orbit mode: the server captures the
    effective body-view camera, then applies the client's new orbit."""
    scene = make_cam_tiny_scene()
    b = MuJoCoBackend(scene)
    b.apply_input(Camera(mode="first"))
    b.apply_input(Orbit(azimuth=12.0, elevation=-3.0, distance=100.0))
    assert scene.camera_mode == "orbit"
    assert scene.orbit[0] == pytest.approx(12.0)


def test_camera_message_reaches_backend_via_simthread(
        fake_backend, fake_renderer, fake_clock) -> None:
    inputs = InputQueue()
    sim = make_sim(fake_backend, fake_renderer, fake_clock, inputs)
    inputs.put(Camera(mode="first"))
    sim.tick()
    assert any(type(m).__name__ == "Camera" for m in fake_backend.inputs)


def test_backend_exposes_camera_mode(fake_backend) -> None:
    assert fake_backend.camera_mode == "orbit"
