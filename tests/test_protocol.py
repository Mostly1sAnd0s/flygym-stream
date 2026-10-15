"""Tests for flygym_stream.protocol (PLAN.md §3 wire protocol v1).

Covered behaviors: round-trips for every control message and status,
rejected payloads -> ProtocolError (close 1008), unknown type tolerated
(None), hello proto mismatch -> version error (close 1002).
"""

from __future__ import annotations

import json

import pytest

import flygym_stream.protocol as proto
from flygym_stream.protocol import (
    Actuator,
    ActuatorSet,
    Hello,
    Orbit,
    Pause,
    ProtocolError,
    Reset,
    Resume,
    Speed,
    Step,
    Status,
    decode_control,
    encode_status,
)


def test_hello_round_trip() -> None:
    assert decode_control('{"type": "hello", "proto": 1}') == Hello(proto=1)


def test_orbit_round_trip() -> None:
    msg = decode_control(
        '{"type": "orbit", "azimuth": 30, "elevation": -12.5, "distance": 500}'
    )
    assert msg == Orbit(azimuth=30.0, elevation=-12.5, distance=500.0)


def test_actuator_round_trip() -> None:
    assert decode_control('{"type": "actuator", "index": 2, "value": -0.5}') == (
        Actuator(index=2, value=-0.5)
    )


def test_actuator_set_round_trip() -> None:
    msg = decode_control('{"type": "actuators", "values": [0.1, 0.2, 0.3]}')
    assert msg == ActuatorSet(values=(0.1, 0.2, 0.3))


def test_fieldless_messages_round_trip() -> None:
    assert decode_control('{"type": "neutral"}') == proto.Neutral()
    assert decode_control('{"type": "pause"}') == Pause()
    assert decode_control('{"type": "resume"}') == Resume()
    assert decode_control('{"type": "reset"}') == Reset()


def test_step_and_speed_round_trip() -> None:
    assert decode_control('{"type": "step", "n": 17}') == Step(n=17)
    assert decode_control('{"type": "speed", "value": 0.25}') == Speed(value=0.25)


def test_accepts_bytes_payload() -> None:
    assert decode_control(b'{"type": "pause"}') == Pause()


@pytest.mark.parametrize("payload", [
    "not json",
    "[1, 2]",
    "null",
    '{"type": 5}',
    "{}",
    '{"type": "orbit", "azimuth": "x", "elevation": 0, "distance": 1}',
    '{"type": "orbit", "azimuth": 0, "distance": 1}',  # missing field
    '{"type": "orbit", "azimuth": 0, "elevation": 0, "distance": -1}',
    '{"type": "actuator", "index": -1, "value": 0}',
    '{"type": "actuator", "index": 0.5, "value": 0}',
    '{"type": "actuator", "index": 0, "value": "hi"}',
    '{"type": "actuators", "values": []}',
    '{"type": "actuators", "values": [0.1, "x"]}',
    '{"type": "actuators", "values": 1}',
    '{"type": "step", "n": 0}',
    '{"type": "step", "n": 1.5}',
    '{"type": "speed", "value": 0}',
    '{"type": "speed", "value": 1000}',
    '{"type": "speed", "value": true}',
    '{"type": "hello"}',  # missing proto
    '{"type": "hello", "proto": true}',
])
def test_rejected_payloads_raise_protocol_error(payload: str) -> None:
    with pytest.raises(ProtocolError):
        decode_control(payload)


def test_unknown_type_returns_none() -> None:
    # Tolerated: caller logs + ignores, server must not crash.
    assert decode_control('{"type": "teleport", "x": 1}') is None


def test_hello_version_mismatch_raises_with_close_code() -> None:
    with pytest.raises(ProtocolError) as exc_info:
        decode_control('{"type": "hello", "proto": 2}')
    err = exc_info.value
    assert err.close_code == 1002
    assert str(proto.PROTOCOL_VERSION) in str(err)
    assert "2" in str(err)


def test_default_close_code_is_1008() -> None:
    with pytest.raises(ProtocolError) as exc_info:
        decode_control("nope")
    assert exc_info.value.close_code == 1008


def test_status_encode_round_trip() -> None:
    status = Status(
        sim_time=1.25, fps=29.7, paused=False, n_actuators=18, backend="cpu",
        model="fly",
    )
    wire = json.loads(encode_status(status))
    assert wire == {
        "type": "status", "sim_time": 1.25, "fps": 29.7, "paused": False,
        "n_actuators": 18, "backend": "cpu", "model": "fly", "orbit": None,
    }


def test_status_carries_orbit_state() -> None:
    """Status reports the live orbit state (same Orbit type as the client
    message) so the browser adopts the server's initial view instead of
    drifting from its own hardcoded default."""
    orbit = Orbit(azimuth=30.0, elevation=-20.0, distance=6400.0)
    status = Status(
        sim_time=1.25, fps=29.7, paused=False, n_actuators=18, backend="cpu",
        model="fly", orbit=orbit,
    )
    wire = json.loads(encode_status(status))
    assert wire["orbit"] == {"azimuth": 30.0, "elevation": -20.0,
                             "distance": 6400.0}
