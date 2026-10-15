"""Tests for telemetry gathering + JSONL logging (M7.4, no GL)."""

from __future__ import annotations

import json

import pytest

from flygym_stream.telemetry import (
    JsonlLogger,
    TelemetryGatherer,
    TelemetryQueue,
    gather_model_info,
)


def test_telemetry_queue_latest_wins() -> None:
    q = TelemetryQueue()
    assert q.get_nowait() is None
    q.put({"t": 1.0})
    q.put({"t": 2.0})
    assert q.get_nowait() == {"t": 2.0}
    assert q.get_nowait() is None  # consumed


def test_model_info_tiny_model(tiny_mj_model) -> None:
    m, d = tiny_mj_model
    info = gather_model_info(m, fly_name="tiny", world="flat")
    assert info["units"] == "mm"
    assert info["nq"] == 1 and info["nv"] == 1 and info["nu"] == 1
    joints = {j["name"]: j for j in info["joints"]}
    assert "hinge" in joints and joints["hinge"]["type"] == "hinge"
    acts = {a["name"]: a for a in info["actuators"]}
    assert any(a["gaintype"] == 0 and a["biastype"] == 1 for a in acts.values())
    assert any(s["name"] == "target" for s in info["sites"])
    assert info["world"] == "flat"


def test_gatherer_reads_mjdata(tiny_mj_model) -> None:
    m, d = tiny_mj_model
    d.qpos[0] = 0.5
    d.qvel[0] = -1.5
    d.ctrl[0] = 0.25
    d.time = 0.42
    g = TelemetryGatherer(m, d, model_info=gather_model_info(m,
                                                              fly_name="tiny",
                                                              world="flat"))
    tel = g.gather(paused=False, mode="walk")
    assert tel["t"] == pytest.approx(0.42)
    assert tel["qpos"] == [0.5]
    assert tel["qvel"] == [-1.5]
    assert tel["ctrl"] == [0.25]
    assert tel["paused"] is False and tel["mode"] == "walk"
    # tiny model's root body is fixed -> root comes from the body frame
    assert set(tel["root"]) == {"pos", "quat"}
    assert len(tel["root"]["pos"]) == 3 and len(tel["root"]["quat"]) == 4
    assert tel["contact"] == {}  # no contact sensors in the tiny model


def test_gatherer_contact_sensors_default_scene():
    """The default fly scene has the 6 built-in leg contact sensors."""
    pytest.importorskip("numpy")
    from flygym_stream.scenes import SceneConfig, build_scene
    scene = build_scene(SceneConfig())
    g = TelemetryGatherer(scene.mj_model, scene.mj_data,
                          model_info={})
    tel = g.gather(paused=True, mode="manual")
    assert set(tel["contact"]) == {"lf", "lm", "lh", "rf", "rm", "rh"}
    # root is free in the flat world: pos == qpos[0:3]
    assert tel["root"]["pos"] == list(scene.mj_data.qpos[0:3])


def test_jsonl_logger(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    logger = JsonlLogger(path)
    try:
        logger.write({"a": 1})
        logger.write({"b": [1, 2]})
    finally:
        logger.close()
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines == [{"a": 1}, {"b": [1, 2]}]
