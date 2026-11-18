"""Equation, intervention, decoder and timing checks without Brian2."""
import json
from pathlib import Path
import unittest
import numpy as np
from .shiu_model import ShiuModel
from .eon_interface import decode_rates, integer_steps, validate_annotations


def model(backend='cpu'):
    return ShiuModel(['a', 'b', 'c'], [0, 0, 1], [1, 1, 2], [100, 100, 40],
                     ['exc', 'inh', 'exc'], {'backend': backend, 'rate_filter_ms': 50, 'seed': 7})


class EonTests(unittest.TestCase):
    def test_impulse_delay_and_duplicate_weights(self):
        brain = model()
        brain.v[0] = -44
        for tick in range(19):
            spikes = brain.step_poisson(np.zeros(3), .0001)
            self.assertEqual(bool(spikes[0]), tick == 0)
            self.assertAlmostEqual(brain.g[1], 55 if tick == 18 else 0)
        self.assertEqual(brain.v[1], -52)
        brain.step_poisson(np.zeros(3), .0001)
        expected = -52 + 55 * (np.exp(-.1/20) - np.exp(-.1/5)) / 3
        self.assertAlmostEqual(brain.v[1], expected, places=12)

    def test_inhibitory_impulse(self):
        brain = model()
        brain.v[1] = -44
        for _ in range(20):
            brain.step_poisson(np.zeros(3), .0001)
        self.assertLess(brain.v[2], -52)

    def test_declared_zero_rate_target_and_event_schedule(self):
        brain = model()
        brain.set_input_targets(['a'])
        # An externally stimulated neuron has no refractory even at zero rate.
        events = np.array([True, False, False])
        fired = brain.step_poisson(np.zeros(3), .0001, input_events=events)
        self.assertFalse(fired.any())  # Poisson events occur after thresholds.
        self.assertAlmostEqual(brain.v[0], -52+68.75)
        self.assertTrue(brain.step_poisson(np.zeros(3), .0001, input_events=events)[0])
        self.assertEqual(brain.v[0], -52)  # reset discards a same-tick external event
        brain.step_poisson(np.zeros(3), .0001, input_events=events)
        self.assertTrue(brain.step_poisson(np.zeros(3), .0001, input_events=np.zeros(3))[0])

    def test_invalid_inputs(self):
        for rates in ([np.nan, 0, 0], [-1, 0, 0], [10001, 0, 0], [0]):
            with self.assertRaises(ValueError):
                model().step_poisson(rates, .0001)
        brain = model()
        brain.set_input_targets(['a'])
        with self.assertRaises(ValueError):
            brain.step_poisson([0, 1, 0], .0001)

    def test_reproducible_poisson(self):
        a, b = model(), model()
        for _ in range(100):
            np.testing.assert_array_equal(a.step_poisson([150, 0, 0], .0001),
                                          b.step_poisson([150, 0, 0], .0001))
        np.testing.assert_array_equal(a.v, b.v)

    def test_refractory_end(self):
        brain = model()
        brain.v[0] = -44
        brain.step_poisson(np.zeros(3), .0001)
        brain.g[0] = 10
        for _ in range(21):
            brain.step_poisson(np.zeros(3), .0001)
            self.assertEqual(brain.g[0], 10)
        brain.step_poisson(np.zeros(3), .0001)
        self.assertLess(brain.g[0], 10)

    def test_decoder_and_sync(self):
        p = dict(forward_gain_per_hz=.025, turn_gain_per_hz=.0125, maximum_drive=1.2)
        r = dict(forward=0, steer_left=0, steer_right=0)
        np.testing.assert_array_equal(decode_rates(r, p), [0, 0])
        r.update(forward=20, steer_left=10)
        left = decode_rates(r, p)
        self.assertGreater(left[1], left[0])
        r.update(steer_left=0, steer_right=10)
        np.testing.assert_array_equal(decode_rates(r, p), left[::-1])
        self.assertEqual(integer_steps(.015, .0001), 150)
        with self.assertRaises(ValueError):
            integer_steps(.01505, .0001)

    def test_local_readout_annotations(self):
        folder = Path(__file__).parent
        p = json.loads((folder / 'eon_config.json').read_text())
        cfg = json.loads((folder / p['base_config']).read_text())
        validate_annotations(p, cfg, folder)


if __name__ == '__main__':
    unittest.main()
