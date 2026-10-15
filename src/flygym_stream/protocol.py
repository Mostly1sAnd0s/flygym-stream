"""Wire protocol v1 (PLAN.md §3) — single source of truth for messages.

Client -> server: WebSocket text JSON control messages decoded by
``decode_control`` into frozen dataclasses. Server -> client: binary JPEG
frames plus ``status`` text JSON built by ``encode_status``; web/app.js
mirrors these dataclasses from one place.
"""

import json
import logging
import math
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
CLOSE_POLICY_ERROR, CLOSE_PROTOCOL_VERSION = 1008, 1002  # policy / version errors


class ProtocolError(Exception):
    """A control message that cannot or must not be accepted."""

    def __init__(self, message: str, close_code: int = CLOSE_POLICY_ERROR) -> None:
        super().__init__(message)
        self.close_code = close_code


@dataclass(frozen=True)
class Hello:  # first client message; carries the protocol version
    proto: int

@dataclass(frozen=True)
class Orbit:  # free camera; azimuth/elevation deg, distance mm
    azimuth: float
    elevation: float
    distance: float
@dataclass(frozen=True)
class Actuator:  # set one actuator ctrl by index
    index: int
    value: float
@dataclass(frozen=True)
class ActuatorSet:  # set all actuator ctrls at once
    values: tuple[float, ...]

def _make_msg(name: str, doc: str) -> type:
    """Build a frozen, field-less message dataclass (one line per type)."""
    return dataclass(frozen=True)(type(name, (), {"__doc__": doc}))
Neutral = _make_msg("Neutral", "Reset actuators to the neutral pose.")
Pause = _make_msg("Pause", "Freeze the sim loop.")
Resume = _make_msg("Resume", "Unfreeze the sim loop.")
Reset = _make_msg("Reset", "Reset the model to its initial (neutral) keyframe.")

@dataclass(frozen=True)
class Step:  # while paused: advance exactly n substeps, then render
    n: int
@dataclass(frozen=True)
class Speed:  # playback speed multiplier (sim time per wall second)
    value: float

# --- Phase 7 (M7.2): research-sandbox state control --------------------------
# Whitelist of model options settable via the `param` message: name ->
# expected value length (0 = scalar). Anything else is rejected (1008).
PARAM_WHITELIST = {"gravity": 3, "timestep": 0, "margin": 0, "solref": 2,
                   "solimp": 5, "friction": 3}
ADHESION_LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")  # = add_leg_adhesion order
MODES = ("neutral", "walk", "flap", "manual")  # M7.3 locomotion modes

@dataclass(frozen=True)
class SetState:  # state write; any of qpos/qvel/ctrl (full-length arrays)
    qpos: tuple[float, ...] | None
    qvel: tuple[float, ...] | None
    ctrl: tuple[float, ...] | None
@dataclass(frozen=True)
class Spawn:  # free-root pose (tethered/fixed-root scenes: raises at apply)
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]  # wxyz
@dataclass(frozen=True)
class Force:  # qfrc_applied injection (body by name or id)
    body: str | int
    force: tuple[float, float, float]
    torque: tuple[float, float, float] | None = None
    duration: float | None = None  # seconds held; None/0 = single step
@dataclass(frozen=True)
class Param:  # whitelist-only model option (PARAM_WHITELIST)
    name: str
    value: float | tuple[float, ...]
@dataclass(frozen=True)
class Keyframe:  # apply a named keyframe
    name: str
@dataclass(frozen=True)
class Adhesion:  # stick/release one leg (adhesion builds only)
    leg: str
    state: int  # 0 or 1
@dataclass(frozen=True)
class Mode:  # locomotion pattern mode (M7.3)
    value: str  # one of MODES
@dataclass(frozen=True)
class Loco:  # live locomotion tuning; None fields = unchanged (M7.3)
    freq: float | None = None  # Hz, (0, 50]
    amplitude: float | None = None  # rad, [0, pi]
    phase: float | None = None  # cycles, [0, 1)

# --- Phase 7 (M7.7): camera views (viewcube + 1st/3rd person) -----------------
CAMERA_MODES = ("orbit", "first", "third")
#: Viewcube faces, fly-relative: front/back/left/right are azimuth snaps
#: around the fly's live heading; top/bottom are elevation snaps.
CAMERA_VIEWS = ("front", "back", "left", "right", "top", "bottom")

@dataclass(frozen=True)
class Camera:  # camera mode and/or viewcube view (at least one, M7.7)
    mode: str | None = None  # one of CAMERA_MODES (None = keep)
    view: str | None = None  # one of CAMERA_VIEWS (viewcube face)
@dataclass(frozen=True)
class Log:  # start/stop server-side JSONL logging (M7.4)
    on: int  # 0 or 1
@dataclass(frozen=True)
class Info:  # request a model-info resend (M7.4)
    pass

@dataclass(frozen=True)
class Status:  # server -> client periodic status (about 1 Hz)
    sim_time: float
    fps: float
    paused: bool
    n_actuators: int
    backend: str
    model: str
    orbit: Orbit | None = None  # live orbit state (None = unknown)
    mode: str | None = None  # locomotion mode (M7.3; None = unknown)
    n_dof: int | None = None  # model nq (M7.4; None = unknown)
    cam_mode: str | None = None  # camera mode (M7.7; None = unknown)


InputMessage = (Hello | Orbit | Actuator | ActuatorSet | Neutral | Pause | Resume
                | Step | Speed | Reset | SetState | Spawn | Force | Param
                | Keyframe | Adhesion | Mode | Loco | Camera | Log | Info)

_FIELDLESS = {"neutral": Neutral, "pause": Pause, "resume": Resume, "reset": Reset}

#: Wire type name for each control message class (M7.4 JSONL logging).
WIRE_TYPES = {
    Hello: "hello", Orbit: "orbit", Actuator: "actuator",
    ActuatorSet: "actuators", Neutral: "neutral", Pause: "pause",
    Resume: "resume", Step: "step", Speed: "speed", Reset: "reset",
    SetState: "set_state", Spawn: "spawn", Force: "force", Param: "param",
    Keyframe: "keyframe", Adhesion: "adhesion", Mode: "mode", Loco: "loco",
    Camera: "camera", Log: "log", Info: "info",
}


def message_type(obj: object) -> str:
    """The wire `type` name of a decoded control message (M7.4)."""
    return WIRE_TYPES[type(obj)]

def _num(value: object, what: str) -> float:
    # Validate a finite JSON number (not bool) and return it as float.
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ProtocolError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _int(value: object, what: str) -> int:
    # Validate a JSON integer (not bool) and return it.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{what} must be an integer, got {value!r}")
    return value


def _field(payload: dict, key: str) -> object:
    # Return payload[key] or raise ProtocolError naming the field.
    if key not in payload:
        raise ProtocolError(f"missing field {key!r}")
    return payload[key]


def _num_list(payload: dict, key: str, what: str,
              length: int | None = None) -> tuple[float, ...]:
    """Validate a JSON list of finite numbers (optionally fixed length)."""
    raw = _field(payload, key)
    if not isinstance(raw, list) or not raw or any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) for v in raw):
        raise ProtocolError(f"{what} must be a non-empty list of numbers")
    if length is not None and len(raw) != length:
        raise ProtocolError(f"{what} must have {length} values, "
                            f"got {len(raw)}")
    return tuple(float(v) for v in raw)


def _opt_num_list(payload: dict, key: str, what: str,
                  length: int) -> tuple[float, ...] | None:
    if key not in payload:
        return None
    return _num_list(payload, key, what, length=length)


def decode_control(payload: bytes | str) -> InputMessage | None:
    """Decode a WebSocket text frame into an InputMessage (None = unknown
    type, tolerated). Raises ProtocolError for malformed JSON/fields."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
        raise ProtocolError("payload must be a JSON object with a string 'type'")
    mtype = obj["type"]
    if factory := _FIELDLESS.get(mtype):
        return factory()
    if mtype == "hello":
        proto = _int(_field(obj, "proto"), "proto")
        if proto != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported proto {proto}; server speaks "
                                f"{PROTOCOL_VERSION}", CLOSE_PROTOCOL_VERSION)
        return Hello(proto)
    if mtype == "orbit":
        orbit = Orbit(*(_num(_field(obj, k), k)
                        for k in ("azimuth", "elevation", "distance")))
        if orbit.distance <= 0:
            raise ProtocolError(f"distance must be > 0, got {orbit.distance}")
        return orbit
    if mtype == "actuator":
        index = _int(_field(obj, "index"), "index")
        if index < 0:
            raise ProtocolError(f"index must be >= 0, got {index}")
        return Actuator(index, _num(_field(obj, "value"), "value"))
    if mtype == "actuators":
        raw = _field(obj, "values")
        if not isinstance(raw, list) or not raw:
            raise ProtocolError("'values' must be a non-empty list of numbers")
        return ActuatorSet(tuple(_num(v, "values[i]") for v in raw))
    if mtype == "step":
        n = _int(_field(obj, "n"), "n")
        if n < 1:
            raise ProtocolError(f"n must be >= 1, got {n}")
        return Step(n)
    if mtype == "speed":
        value = _num(_field(obj, "value"), "value")
        if not 0 < value <= 100.0:
            raise ProtocolError(f"value must be in (0, 100], got {value}")
        return Speed(value)
    # --- Phase 7 (M7.2): research-sandbox state control ---
    if mtype == "set_state":
        qpos = _opt_num_list(obj, "qpos", "qpos", length=None)
        qvel = _opt_num_list(obj, "qvel", "qvel", length=None)
        ctrl = _opt_num_list(obj, "ctrl", "ctrl", length=None)
        if qpos is None and qvel is None and ctrl is None:
            raise ProtocolError(
                "set_state needs at least one of qpos/qvel/ctrl")
        return SetState(qpos=qpos, qvel=qvel, ctrl=ctrl)
    if mtype == "spawn":
        return Spawn(_num_list(obj, "pos", "pos", length=3),
                     _num_list(obj, "quat", "quat", length=4))
    if mtype == "force":
        body = _field(obj, "body")
        if not (isinstance(body, str) or isinstance(body, int) and not
                isinstance(body, bool)):
            raise ProtocolError(f"body must be a name or id, got {body!r}")
        duration = obj.get("duration")
        if duration is not None:
            duration = _num(duration, "duration")
            if duration <= 0:
                raise ProtocolError(f"duration must be > 0, got {duration}")
        return Force(body, _num_list(obj, "force", "force", length=3),
                     _opt_num_list(obj, "torque", "torque", length=3),
                     duration)
    if mtype == "param":
        name = _field(obj, "name")
        if not isinstance(name, str) or name not in PARAM_WHITELIST:
            raise ProtocolError(
                f"param {name!r} not in whitelist {sorted(PARAM_WHITELIST)}")
        length = PARAM_WHITELIST[name]
        value = _field(obj, "value")
        if length == 0:
            value = _num(value, "value")
        else:
            value = _num_list(obj, "value", f"value for {name}", length=length)
        return Param(name, value)
    if mtype == "keyframe":
        name = _field(obj, "name")
        if not isinstance(name, str) or not name:
            raise ProtocolError(f"keyframe name must be a non-empty string")
        return Keyframe(name)
    if mtype == "adhesion":
        leg = _field(obj, "leg")
        state = _field(obj, "state")
        if leg not in ADHESION_LEGS:
            raise ProtocolError(f"leg {leg!r} not in {ADHESION_LEGS}")
        if state not in (0, 1) or isinstance(state, bool):
            raise ProtocolError(f"state must be 0 or 1, got {state!r}")
        return Adhesion(leg, state)
    if mtype == "mode":
        value = _field(obj, "value")
        if not isinstance(value, str) or value not in MODES:
            raise ProtocolError(f"mode must be one of {MODES}, got {value!r}")
        return Mode(value)
    if mtype == "loco":
        freq = _loco_num(obj, "freq", lo=0.0, hi=50.0, lo_open=True)
        amplitude = _loco_num(obj, "amplitude", lo=0.0, hi=math.pi)
        phase = _loco_num(obj, "phase", lo=0.0, hi=1.0, hi_open=True)
        if freq is None and amplitude is None and phase is None:
            raise ProtocolError(
                "loco needs at least one of freq/amplitude/phase")
        return Loco(freq=freq, amplitude=amplitude, phase=phase)
    if mtype == "log":
        on = _field(obj, "on")
        if on not in (0, 1) or isinstance(on, bool):
            raise ProtocolError(f"log on must be 0 or 1, got {on!r}")
        return Log(on)
    if mtype == "camera":
        mode, view = obj.get("mode"), obj.get("view")
        if mode is not None and (not isinstance(mode, str)
                                 or mode not in CAMERA_MODES):
            raise ProtocolError(
                f"camera mode must be one of {CAMERA_MODES}, got {mode!r}")
        if view is not None and (not isinstance(view, str)
                                 or view not in CAMERA_VIEWS):
            raise ProtocolError(
                f"camera view must be one of {CAMERA_VIEWS}, got {view!r}")
        if mode is None and view is None:
            raise ProtocolError("camera needs at least one of mode/view")
        return Camera(mode=mode, view=view)
    if mtype == "info":
        return Info()
    logger.info("ignoring unknown control message type %r", mtype)
    return None


def _loco_num(payload: dict, key: str, lo: float, hi: float,
              lo_open: bool = False, hi_open: bool = False) -> float | None:
    """Optional bounded scalar (M7.3 loco tuning)."""
    if key not in payload:
        return None
    value = _num(payload[key], key)
    if (value <= lo) if lo_open else (value < lo):
        raise ProtocolError(f"{key} must be {'>' if lo_open else '>='} {lo}, "
                            f"got {value}")
    if (value >= hi) if hi_open else (value > hi):
        raise ProtocolError(f"{key} must be {'<' if hi_open else '<='} {hi}, "
                            f"got {value}")
    return value


def encode_status(status: Status) -> str:
    """Encode a Status for the server -> client channel (text JSON)."""
    obj = asdict(status)
    obj["type"] = "status"
    return json.dumps(obj)


def encode_telemetry(obj: dict) -> str:
    """Encode a telemetry payload (M7.4) for the server -> client channel."""
    out = dict(obj)
    out["type"] = "telemetry"
    return json.dumps(out)


def encode_model_info(info: dict) -> str:
    """Encode a model-info payload (M7.4) for the server -> client channel."""
    out = dict(info)
    out["type"] = "model-info"
    return json.dumps(out)
