"""Telemetry gathering, model-info, and JSONL logging (Phase 7, M7.4).

- ``TelemetryQueue``: latest-wins single-slot queue (FrameQueue semantics).
- ``gather_model_info``: static model description for arbitrary external
  clients (sent after hello and on ``info``).
- ``TelemetryGatherer``: rate-limited MjData snapshot (the MjData is the
  single source of truth; the 6 built-in leg contact sensors are read
  from ``sensordata``).
- ``JsonlLogger``: append-only, thread-safe JSON Lines logging.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np

#: Built-in leg ground-contact sensor names (flygym), per leg.
CONTACT_SENSOR_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")


class TelemetryQueue:
    """Latest-wins single-slot queue for telemetry payloads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: dict | None = None

    def put(self, item: dict) -> None:
        with self._lock:
            self._item = item  # drop anything older (latest wins)

    def get_nowait(self) -> dict | None:
        with self._lock:
            item, self._item = self._item, None
            return item


def _enum_name(enum_cls, value: int) -> str:
    """A readable lowercase name for an enum value (e.g. 'hinge')."""
    try:
        name = enum_cls(int(value)).name
    except (ValueError, TypeError):
        return str(value)
    for prefix in ("mjJNT_", "mjSENS_"):
        if name.startswith(prefix):
            return name[len(prefix):].lower()
    return name[2:].lower() if name.startswith("mj") else name.lower()


def gather_model_info(model: mj.MjModel, *, fly_name: str,
                      world: str) -> dict:
    """Build the static model-info payload (M7.4).

    Units are model units: this project runs the full-size NMf in mm
    (gravity 9810 mm/s^2). Enables arbitrary external clients.
    """
    joints = [{
        "id": i,
        "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) or "",
        "type": _enum_name(mj.mjtJoint, int(model.jnt_type[i])),
        "qadr": int(model.jnt_qposadr[i]),
    } for i in range(model.njnt)]
    def _actuator_entry(i: int) -> dict:
        # No single "type" field in MuJoCo: an actuator's behavior is the
        # (gaintype, biastype, dyntype) triple. gaintype 0 = joint target,
        # 1 = tendon, 2 = force; the target id is in trnid[:, 0].
        gaintype = int(model.actuator_gaintype[i])
        target = int(model.actuator_trnid[i, 0])
        return {
            "id": i,
            "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) or "",
            "gaintype": gaintype,
            "biastype": int(model.actuator_biastype[i]),
            "dyntype": int(model.actuator_dyntype[i]),
            "joint": target if gaintype == 0 else None,
            "tendon": target if gaintype == 1 else None,
            "range": [float(v) for v in model.actuator_ctrlrange[i]],
        }

    actuators = [_actuator_entry(i) for i in range(model.nu)]
    sites = [{"id": i,
              "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, i) or ""}
             for i in range(model.nsite)]
    cameras = [{"id": i,
                "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) or ""}
               for i in range(model.ncam)]
    keyframes = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_KEY, i) or ""
                 for i in range(model.nkey)]
    sensors = [{
        "id": i,
        "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i) or "",
        "type": _enum_name(mj.mjtSensor, int(model.sensor_type[i])),
    } for i in range(model.nsensor)]
    return {
        "units": "mm",
        "fly_name": fly_name,
        "world": world,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "timestep": float(model.opt.timestep),
        "gravity": [float(v) for v in model.opt.gravity],
        "joints": joints,
        "actuators": actuators,
        "sites": sites,
        "cameras": cameras,
        "keyframes": keyframes,
        "sensors": sensors,
    }


class TelemetryGatherer:
    """Snapshots MjData into the wire telemetry dict (M7.4)."""

    def __init__(self, model: mj.MjModel, data, *, model_info: dict) -> None:
        self._model = model
        self._data = data
        self._model_info = model_info
        # Root: free root joint (qpos[0:7]) or the thorax body frame
        # (tethered/fixed-root worlds pin the root joint).
        self._root_free = (
            model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE)
        # Root body fallback when the root is pinned: the composable
        # scenes call it `nmf/c_thorax`; the FlyMimic MSK model (M7.6)
        # calls it `Thorax`.
        self._thorax_body = -1
        for name in (f"{model_info.get('fly_name', 'nmf')}/c_thorax",
                     "Thorax"):
            self._thorax_body = mj.mj_name2id(
                model, mj.mjtObj.mjOBJ_BODY, name)
            if self._thorax_body >= 0:
                break
        # Leg ground-contact sensor data addresses (may be absent).
        self._contact: dict[str, tuple[int, int]] = {}
        for leg in CONTACT_SENSOR_NAMES:
            sid = mj.mj_name2id(
                model, mj.mjtObj.mjOBJ_SENSOR, f"ground_contact_{leg}_leg")
            if sid >= 0:
                self._contact[leg] = (int(model.sensor_adr[sid]),
                                      int(model.sensor_dim[sid]))

    def gather(self, *, paused: bool, mode: str) -> dict:
        """One telemetry snapshot (JSON-serializable)."""
        m, d = self._model, self._data
        if self._root_free:
            root = {"pos": [float(v) for v in d.qpos[0:3]],
                    "quat": [float(v) for v in d.qpos[3:7]]}
        elif self._thorax_body >= 0:
            root = {"pos": [float(v) for v in d.xpos[self._thorax_body]],
                    "quat": [float(v) for v in d.xquat[self._thorax_body]]}
        else:  # pragma: no cover - defensive
            root = {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
        contact = {
            leg: [float(v) for v in d.sensordata[adr:adr + dim]]
            for leg, (adr, dim) in self._contact.items()
        }
        return {
            "t": float(d.time),
            "qpos": [float(v) for v in d.qpos],
            "qvel": [float(v) for v in d.qvel],
            "ctrl": [float(v) for v in d.ctrl],
            "root": root,
            "contact": contact,
            "paused": bool(paused),
            "mode": mode,
        }


class JsonlLogger:
    """Append-only thread-safe JSON Lines file (one object per line)."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, obj: dict) -> None:
        line = json.dumps(obj)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()


@dataclass
class TelemetrySink:
    """What the sim thread needs to rate-limit and publish telemetry:
    the gatherer, the latest-wins queue, and the rate (Hz)."""

    gatherer: TelemetryGatherer
    queue: TelemetryQueue
    hz: float
