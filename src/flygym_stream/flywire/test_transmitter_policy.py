import copy
import json
from pathlib import Path
import unittest
import numpy as np
from .load_network import classify
from .neural_model import NeuralModel


class TransmitterPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path(__file__).with_name('hybrid_config.json').read_text())

    def test_labels_and_fallbacks(self):
        for label in ['GLUT', 'glutamate', 'HIST', 'histamine', 'acetylcholine,histamine']:
            self.assertEqual(classify(label, self.cfg), 'inh')
        for label in ['DA', 'dopamine', 'SER', 'serotonin', 'OCT', 'TYR', '', 'unrecognized']:
            self.assertEqual(classify(label, self.cfg), 'exc')
        self.assertEqual(classify('acetylcholine,dopamine', self.cfg), 'exc')

    def test_overrides_and_legacy_policies(self):
        self.cfg['nt_class_map']['GLUT'] = 'exc'
        self.assertEqual(classify('glut', self.cfg), 'exc')
        self.cfg['transmitter_guesses']['unknown_class'] = 'inh'
        self.assertEqual(classify('', self.cfg), 'inh')
        for policy in ['disable', 'error']:
            self.cfg['data']['unmapped_transmitters'] = policy
            self.assertEqual(classify('unrecognized', self.cfg), 'disabled')

    def test_new_classes_propagate_with_expected_sign(self):
        for label, sign in [('GLUT', -1), ('HIST', -1), ('DA', 1), ('SER', 1), ('OCT', 1), ('TYR', 1), ('unknown', 1)]:
            params = copy.deepcopy(self.cfg['neural'])
            params['backend'] = 'cpu'
            brain = NeuralModel(['source', 'target'], [0], [1], [10],
                                [classify(label, self.cfg), 'exc'], params)
            brain.gate[0] = 0.5
            brain.step(np.zeros(2), 0.0001)
            self.assertGreater(sign * (brain.v[1] - params['rest_mV']), 0, label)


if __name__ == '__main__':
    unittest.main()
