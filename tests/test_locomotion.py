"""Pure-math tests for the locomotion pattern engine (M7.3, no GL/model)."""

from __future__ import annotations

import numpy as np
import pytest

from flygym_stream.locomotion import (
    LEGS,
    LocoParams,
    flap_wing_values,
    walk_leg_values,
)


def _t(freq: float, n: int = 2001) -> np.ndarray:
    """One full cycle of the pattern, sampled."""
    return np.linspace(0.0, 1.0 / freq, n, endpoint=False)


def test_walk_values_bounded_and_oscillatory() -> None:
    p = LocoParams(freq=2.0, amplitude=0.4, phase=0.0)
    t = _t(2.0)
    vals = [walk_leg_values(float(x), p, "lf")["femur_pitch"] for x in t]
    arr = np.asarray(vals)
    assert arr.max() == pytest.approx(0.4, abs=1e-3)
    assert arr.min() == pytest.approx(-0.4, abs=1e-3)
    # one interior zero crossing per cycle (the second sits exactly on the
    # cycle boundary, excluded by endpoint=False)
    zeros = int(np.count_nonzero(np.diff(np.sign(arr[1:])) != 0))
    assert zeros == 1
    for dof, v in walk_leg_values(0.1, p, "lf").items():
        assert abs(v) <= p.amplitude + 1e-9, dof


def test_walk_tripod_phasing() -> None:
    """Tripod A (lf/rm/rh) and B (rf/lm/lh) are half a cycle apart."""
    p = LocoParams(freq=1.0, amplitude=0.5, phase=0.0)
    # at t = 0.25 s (quarter cycle) lf femur is at +peak; rf at -peak
    a = walk_leg_values(0.25, p, "lf")["femur_pitch"]
    b = walk_leg_values(0.25, p, "rf")["femur_pitch"]
    assert a == pytest.approx(0.5, abs=1e-6)
    assert b == pytest.approx(-0.5, abs=1e-6)
    # all tripod-mates agree
    for mate in ("rm", "rh"):
        assert walk_leg_values(0.25, p, mate)["femur_pitch"] == pytest.approx(a)
    for mate in ("lm", "lh"):
        assert walk_leg_values(0.25, p, mate)["femur_pitch"] == pytest.approx(b)


def test_walk_knee_lags_femur_quarter_cycle() -> None:
    p = LocoParams(freq=1.0, amplitude=0.5, phase=0.0)
    # knee peaks a quarter cycle after femur: femur peak at t=0.25 ->
    # knee peak at t=0.5
    knee = walk_leg_values(0.5, p, "lf")["tibia_pitch"]
    femur = walk_leg_values(0.5, p, "lf")["femur_pitch"]
    assert knee == pytest.approx(0.5, abs=1e-6)
    assert femur == pytest.approx(0.0, abs=1e-6)


def test_walk_yaw_zero_in_baseline() -> None:
    """Baseline walk is a sagittal-plane gait (femur swing + knee + bob);
    coxa yaw stays at neutral (0). Tripod gaits are NOT instantaneous
    left/right mirrors (lf/rf are half a cycle apart), so a mirrored yaw
    oscillation would weave the body sideways. Yaw steering can be added
    later via loco tuning / set_state without breaking forward walking."""
    p = LocoParams(freq=1.0, amplitude=0.5, phase=0.0)
    t = np.linspace(0.0, 1.0, 501, endpoint=False)
    for leg in LEGS:
        for x in t:
            assert walk_leg_values(float(x), p, leg)["coxa_yaw"] == 0.0
            assert walk_leg_values(float(x), p, leg)["femur_roll"] == 0.0
            assert walk_leg_values(float(x), p, leg)["coxa_roll"] == 0.0
            assert walk_leg_values(float(x), p, leg)["tarsus_pitch"] == 0.0


def test_walk_phase_param_shifts_gait() -> None:
    base = LocoParams(freq=1.0, amplitude=0.5, phase=0.0)
    shifted = LocoParams(freq=1.0, amplitude=0.5, phase=0.25)
    # base +peak is at t=0.25; shifting phase by +0.25 cycles moves it
    # to t=0.0 (sin(2 pi (0 + 0.25)) = +1)
    assert walk_leg_values(0.25, base, "lf")["femur_pitch"] == pytest.approx(0.5, abs=1e-6)
    assert walk_leg_values(0.0, shifted, "lf")["femur_pitch"] == pytest.approx(0.5, abs=1e-6)
    assert walk_leg_values(0.0, base, "lf")["femur_pitch"] == pytest.approx(0.0, abs=1e-6)


def test_walk_deterministic() -> None:
    p = LocoParams(freq=3.0, amplitude=0.3, phase=0.1)
    for leg in LEGS:
        assert walk_leg_values(1.234, p, leg) == walk_leg_values(1.234, p, leg)


def test_flap_in_phase_both_wings() -> None:
    """Drosophila wings couple and beat in phase (plan said antiphase;
    corrected to the biological fact)."""
    p = LocoParams(freq=5.0, amplitude=0.6, phase=0.0)
    l, r = flap_wing_values(0.1, p)
    assert l == pytest.approx(r, abs=1e-12)
    t = np.linspace(0.0, 1.0, 5001, endpoint=False)  # 1 s = 5 cycles
    vals = np.asarray([flap_wing_values(float(x), p)[0] for x in t])
    assert vals.max() == pytest.approx(0.6, abs=1e-3)
    assert vals.min() == pytest.approx(-0.6, abs=1e-3)
    # 5 Hz -> 5 peaks in 1 s
    peaks = int(np.sum((vals > 0.55) & (np.roll(vals, 1) <= 0.55)))
    assert peaks == 5
