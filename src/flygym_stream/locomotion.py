"""Locomotion pattern engine (PLAN.md Phase 7, M7.3).

Pure, deterministic functions of sim time: a built-in gait is applied on
the sim thread before each substep while a pattern mode is active. The
pattern writes *offsets from the neutral ctrl* so every DOF it does not
drive keeps its neutral (or slider-set, in ``manual``) value.

Modes (protocol.MODES):
- ``neutral``: hold the neutral keyframe pose (same as the neutral ctrl).
- ``walk``: six-leg tripod gait (lf/rm/rh vs rf/lm/lh half a cycle apart)
  on the sagittal leg DOFs: femur swing, knee lagging a quarter cycle,
  small double-frequency body bob. Sagittal-plane only (no coxa yaw) so
  the fly advances straight along +x — a mirrored-yaw oscillation would
  weave sideways, since tripod-mate L/R legs are anti-phase.
- ``flap``: both wing-pitch DOFs in phase (Drosophila wings couple; the
  plan's "antiphase" wording was corrected to the biological fact).
  Requires wing actuators (``--dof all``).
- ``manual``: pattern engine idle — client slider authority over ctrl.
"""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np

from flygym_stream.protocol import MODES

__all__ = [
    "LEGS", "LocoParams", "LocController", "flap_wing_values",
    "walk_leg_values",
]

#: Leg names in the order flygym's leg APIs use them (also the adhesion
#: actuator order, protocol.ADHESION_LEGS).
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")

#: Tripod A (phase 0); the other three legs are Tripod B (phase 0.5).
TRIPOD_A = frozenset({"lf", "rm", "rh"})

#: Default gait parameters (live-tunable via the `loco` message).
DEFAULT_FREQ_HZ = 2.0
DEFAULT_AMPLITUDE_RAD = 0.35
DEFAULT_PHASE = 0.0


class LocoParams:
    """Live locomotion tuning (freq Hz, amplitude rad, phase cycles)."""

    __slots__ = ("freq", "amplitude", "phase")

    def __init__(self, freq: float = DEFAULT_FREQ_HZ,
                 amplitude: float = DEFAULT_AMPLITUDE_RAD,
                 phase: float = DEFAULT_PHASE) -> None:
        self.freq = float(freq)
        self.amplitude = float(amplitude)
        self.phase = float(phase)

    def copy(self) -> "LocoParams":
        return LocoParams(self.freq, self.amplitude, self.phase)


def walk_leg_values(t: float, params: LocoParams, leg: str) -> dict[str, float]:
    """Offset-from-neutral ctrl values for one leg at sim time ``t``.

    Returns a mapping over the seven leg DOFs (values in rad; unused
    DOFs are 0.0). Deterministic; depends only on (t, params, leg).
    """
    if leg not in LEGS:
        raise ValueError(f"leg {leg!r} not in {LEGS}")
    offset = params.phase + (0.0 if leg in TRIPOD_A else 0.5)
    phi = 2.0 * math.pi * (params.freq * t + offset)
    a = params.amplitude
    return {
        "femur_pitch": a * math.sin(phi),
        "tibia_pitch": a * math.sin(phi - 0.5 * math.pi),
        "coxa_yaw": 0.0,
        "coxa_pitch": 0.2 * a * math.sin(2.0 * phi),
        "coxa_roll": 0.0,
        "femur_roll": 0.0,
        "tarsus_pitch": 0.0,
    }


def flap_wing_values(t: float, params: LocoParams) -> tuple[float, float]:
    """(left, right) wing-pitch offsets at sim time ``t`` — in phase."""
    v = params.amplitude * math.sin(
        2.0 * math.pi * (params.freq * t + params.phase))
    return (v, v)


#: Per-DOF actuator-name suffixes, relative to the fly body name.
_LEG_ACTUATOR_NAMES = {
    "coxa_yaw": "c_thorax-{leg}_coxa-yaw-position",
    "coxa_pitch": "c_thorax-{leg}_coxa-pitch-position",
    "coxa_roll": "c_thorax-{leg}_coxa-roll-position",
    "femur_pitch": "{leg}_coxa-{leg}_trochanterfemur-pitch-position",
    "femur_roll": "{leg}_coxa-{leg}_trochanterfemur-roll-position",
    "tibia_pitch": "{leg}_trochanterfemur-{leg}_tibia-pitch-position",
    "tarsus_pitch": "{leg}_tibia-{leg}_tarsus1-pitch-position",
}
_WING_PITCH_NAMES = ("c_thorax-l_wing-pitch-position",
                     "c_thorax-r_wing-pitch-position")


class LocController:
    """Applies a locomotion mode to ``mj_data.ctrl`` (sim thread).

    Actuator indices are resolved by name once at construction, so the
    same controller works for any actuated-DOF preset (42 or 126).
    Missing actuators (e.g. wings without ``--dof all``) disable the
    DOF that needs them; ``flap`` without wing actuators is rejected.
    """

    def __init__(self, mj_model: mj.MjModel, neutral_ctrl: object,
                 fly_name: str = "nmf") -> None:
        self._neutral = np.asarray(neutral_ctrl, dtype=float)
        self._leg_ids: dict[str, dict[str, int]] = {}
        for leg in LEGS:
            ids: dict[str, int] = {}
            for dof, suffix in _LEG_ACTUATOR_NAMES.items():
                name = f"{fly_name}/{suffix.format(leg=leg)}"
                ids[dof] = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_ACTUATOR,
                                         name)
            self._leg_ids[leg] = ids
        self._wing_ids = tuple(
            mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_ACTUATOR,
                          f"{fly_name}/{suffix}")
            for suffix in _WING_PITCH_NAMES)
        self._mode = "neutral"
        self._params = LocoParams()
        self._baseline: np.ndarray | None = None
        self._baseline_mode: str | None = None
        if not any(any(v >= 0 for v in ids.values())
                   for ids in self._leg_ids.values()):
            raise ValueError("no leg actuators found in model "
                             f"(fly name {fly_name!r}?)")

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def params(self) -> LocoParams:
        """A copy: mutation here does not affect the controller."""
        return self._params.copy()

    def set_mode(self, mode: str) -> None:
        """Switch locomotion mode.

        Raises:
            ValueError: Unknown mode, or ``flap`` without wing actuators.
        """
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if mode == "flap" and any(i < 0 for i in self._wing_ids):
            raise ValueError(
                "flap requires wing actuators (serve with --dof all)")
        self._mode = mode

    def set_params(self, freq: float | None = None,
                   amplitude: float | None = None,
                   phase: float | None = None) -> None:
        """Update a subset of the gait parameters (same ranges as the
        protocol's ``loco`` message)."""
        p = self._params
        if freq is not None:
            if not 0.0 < freq <= 50.0:
                raise ValueError(f"freq must be in (0, 50], got {freq}")
            p.freq = float(freq)
        if amplitude is not None:
            if not 0.0 <= amplitude <= math.pi:
                raise ValueError(
                    f"amplitude must be in [0, pi], got {amplitude}")
            p.amplitude = float(amplitude)
        if phase is not None:
            if not 0.0 <= phase < 1.0:
                raise ValueError(f"phase must be in [0, 1), got {phase}")
            p.phase = float(phase)

    def apply(self, data) -> None:
        """Write the active pattern into ``data.ctrl`` (no-op in
        ``manual``). Called on the sim thread before each substep.

        The pattern adds its offsets to a *baseline captured on mode
        entry*: walking from a standing set_state'd pose walks; walking
        from the (prone) neutral keyframe righting-kicks instead. In
        ``neutral`` mode the neutral keyframe ctrl is restored instead.
        """
        if self._mode == "manual":
            return
        if self._mode == "neutral":
            data.ctrl[:] = self._neutral
            self._baseline_mode = None
            return
        if self._baseline_mode != self._mode:
            self._baseline = np.array(data.ctrl, dtype=float)
            self._baseline_mode = self._mode
        base = self._baseline
        t = float(data.time)
        if self._mode == "walk":
            for leg in LEGS:
                values = walk_leg_values(t, self._params, leg)
                for dof, index in self._leg_ids[leg].items():
                    if index >= 0:
                        data.ctrl[index] = base[index] + values[dof]
        elif self._mode == "flap":
            left, right = flap_wing_values(t, self._params)
            for index, value in zip(self._wing_ids, (left, right)):
                if index >= 0:
                    data.ctrl[index] = base[index] + value
