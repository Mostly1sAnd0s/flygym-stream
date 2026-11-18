"""Connectome integration checks runnable with unittest (no EGL or dataset)."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    from flygym_stream.flywire.session import EonSession
    from flygym_stream.flywire.shiu_model import ShiuModel
except ModuleNotFoundError as exc:
    if exc.name not in ('pandas', 'scipy'):
        raise
    raise unittest.SkipTest('Install requirements-flywire.txt for connectome tests.') from exc


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        profile = dict(base_config='base.json', sync_ms=15, sensory_hz=100,
                       forward_gain_per_hz=.025, turn_gain_per_hz=.0125,
                       maximum_drive=1.2,
                       readouts=dict(forward=['b'], steer_left=['b'], steer_right=['c']))
        cfg = dict(neural={'rate_filter_ms': 50}, populations={},
                   stimulus={'source_xy_mm': [10, 0], 'odor_sigma_mm': 20})
        (self.root / 'profile.json').write_text(json.dumps(profile))
        (self.root / 'base.json').write_text(json.dumps(cfg))

    def session(self, condition='sensory'):
        def load(cfg, folder):
            b = ShiuModel(['a', 'b', 'c'], [0, 0, 1], [1, 1, 2], [100, 100, 40],
                          ['exc', 'inh', 'exc'], cfg['neural'])
            return b, dict(input_left=['a'], input_right=['a']), {}
        with patch('flygym_stream.flywire.session.validate_annotations'), \
             patch('flygym_stream.flywire.session.load_network', side_effect=load):
            return EonSession(self.root / 'profile.json', 'cpu', 7, condition)

    def test_reset_repeats_neural_state_and_drive(self):
        s = self.session()
        first = s.advance([[10, 0], [20, 0]]).copy()
        voltage, counts = s.brain.v.copy(), s.counts.copy()
        s.advance([[0, 0], [0, 0]])
        s.reset()
        np.testing.assert_array_equal(first, s.advance([[10, 0], [20, 0]]))
        np.testing.assert_array_equal(voltage, s.brain.v)
        np.testing.assert_array_equal(counts, s.counts)
        self.assertEqual(s.brain.tick, 150)

    def test_disconnect_preserves_neural_activity(self):
        a, b = self.session(), self.session('disconnect-output')
        for _ in range(3):
            a.advance([[10, 0], [10, 0]])
            b.advance([[10, 0], [10, 0]])
            np.testing.assert_array_equal(a.counts, b.counts)
            np.testing.assert_array_equal(b.drive, [0, 0])

    def test_zero_input_and_invalid_positions(self):
        s = self.session('no-input')
        np.testing.assert_array_equal(s.advance([[10, 0], [10, 0]]), [0, 0])
        self.assertEqual(s.spikes, 0)
        with self.assertRaises(FloatingPointError):
            s.advance([[float('nan'), 0], [10, 0]])

    def test_streaming_schedule_and_reset(self):
        from flygym_stream.flywire.streaming import FlywireBackend
        from flygym_stream.protocol import Neutral, Orbit, Actuator
        data = SimpleNamespace(time=0., qpos=np.zeros(1), xpos=np.zeros((3, 3)))
        scene = SimpleNamespace(mj_model=SimpleNamespace(opt=SimpleNamespace(timestep=.001)),
                                mj_data=data, n_actuators=1, sim=object())
        scene.step = lambda: setattr(data, 'time', data.time + .001)
        scene.reset = lambda: setattr(data, 'time', 0.)
        session = self.session()
        session.manifest = lambda: {}
        from unittest.mock import Mock
        controller = Mock()
        with patch('flygym_demo.complex_terrain.HybridControllerObservation.from_sim'), \
             patch('flygym_demo.complex_terrain.apply_locomotion_action'):
            backend = FlywireBackend(scene, session, SimpleNamespace(name='fly'),
                                     controller, 2, [0, 1], self.root / 'runs')
            try:
                for _ in range(15):
                    backend.step(.001)
                self.assertEqual(session.brain.tick, 150)
                self.assertEqual(controller.step.call_count, 15)
                self.assertAlmostEqual(backend.time, .015)
                self.assertEqual(len((backend.output / 'trace.jsonl').read_text().splitlines()), 1)
                backend.reset()
                self.assertEqual(session.brain.tick, 0)
                self.assertEqual(backend.time, 0)
                self.assertEqual(backend.body_tick, 0)
                backend.step(.001)
                self.assertEqual(session.brain.tick, 150)
                backend.apply_input(Neutral())
                self.assertEqual(backend.body_tick, 0)
                self.assertAlmostEqual(backend.time, .001)
                with self.assertRaises(ValueError):
                    backend.step(.0001)
            finally:
                backend.trace.close()


if __name__ == '__main__':
    unittest.main()
