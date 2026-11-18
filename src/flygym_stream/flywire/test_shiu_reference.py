"""Compare the port to Brian2's linear solver and event schedule on a fixture.

Brian2 is an optional validation dependency, not required to run the GPU port.
"""
import unittest
import numpy as np
from .shiu_model import ShiuModel


class ShiuReferenceTests(unittest.TestCase):
    def test_against_brian2(self):
        import brian2 as b
        b.start_scope()
        b.prefs.codegen.target = 'numpy'
        b.defaultclock.dt = .1*b.ms
        neurons = b.NeuronGroup(3,
            'dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)\n'
            'dg/dt=-g/(5*ms):volt (unless refractory)',
            threshold='v > -45*mV', reset='v=-52*mV; g=0*mV',
            refractory=2.2*b.ms, method='linear')
        neurons.v = [-44, -52, -52]*b.mV
        syn = b.Synapses(neurons, neurons, 'w:volt', on_pre='g+=w', delay=1.8*b.ms)
        syn.connect(i=[0, 1], j=[1, 2]); syn.w = [55, -11]*b.mV
        monitor = b.StateMonitor(neurons, ['v', 'g'], record=True, when='end')
        spikes = b.SpikeMonitor(neurons)
        network = b.Network(neurons, syn, monitor, spikes)
        network.run(15*b.ms)
        model = ShiuModel(['a','b','c'], [0,1], [1,2], [200,40],
                          ['exc','inh','exc'], {'backend':'cpu','rate_filter_ms':50})
        model.v[0] = -44
        vs, gs, events = [], [], []
        for tick in range(150):
            fired = model.step_poisson(np.zeros(3), .0001)
            events.extend((tick, int(i)) for i in np.flatnonzero(fired))
            vs.append(model.v.copy()); gs.append(model.g.copy())
        np.testing.assert_allclose(np.array(vs).T, monitor.v/b.mV, atol=1e-10)
        np.testing.assert_allclose(np.array(gs).T, monitor.g/b.mV, atol=1e-10)
        self.assertEqual(events, list(zip(np.rint(spikes.t/b.defaultclock.dt).astype(int), spikes.i[:].tolist())))

    def test_zero_input_and_invalid_timestep(self):
        model = ShiuModel(['a'], [], [], [], ['exc'], {'backend':'cpu','rate_filter_ms':50})
        for _ in range(50):
            self.assertFalse(model.step_poisson(np.zeros(1), .0001).any())
        self.assertEqual(model.v[0], -52)
        with self.assertRaises(ValueError): model.step_poisson(np.zeros(1), .0002)

    def test_external_event_replay_against_brian2(self):
        """Same external events avoid confounding solver parity with different RNGs."""
        import brian2 as b
        b.start_scope()
        b.prefs.codegen.target = 'numpy'
        b.defaultclock.dt = .1*b.ms
        neurons = b.NeuronGroup(3,
            'dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)\n'
            'dg/dt=-g/(5*ms):volt (unless refractory)\n'
            'rfc:second', threshold='v > -45*mV', reset='v=-52*mV; g=0*mV',
            refractory='rfc', method='linear')
        neurons.v = -52*b.mV
        neurons.rfc = [0, 2.2, 2.2]*b.ms
        recurrent = b.Synapses(neurons, neurons, 'w:volt', on_pre='g+=w', delay=1.8*b.ms)
        recurrent.connect(i=[0, 1], j=[1, 2])
        recurrent.w = [55, -11]*b.mV
        event_ticks = [0, 1, 2, 7, 22, 30, 60, 70, 71, 99]
        external = b.SpikeGeneratorGroup(1, np.zeros(len(event_ticks), dtype=int),
                                        np.asarray(event_ticks)*b.defaultclock.dt)
        injection = b.Synapses(external, neurons, on_pre='v_post += 68.75*mV')
        injection.connect(i=0, j=0)
        monitor = b.StateMonitor(neurons, ['v', 'g'], record=True, when='end')
        spikes = b.SpikeMonitor(neurons)
        b.Network(neurons, recurrent, external, injection, monitor, spikes).run(15*b.ms)
        model = ShiuModel(['a','b','c'], [0,1], [1,2], [200,40],
                          ['exc','inh','exc'], {'backend':'cpu','rate_filter_ms':50})
        model.set_input_targets(['a'])
        vs, gs, events = [], [], []
        for tick in range(150):
            fired = model.step_poisson(np.zeros(3), .0001,
                                      input_events=[tick in event_ticks, False, False])
            events.extend((tick, int(i)) for i in np.flatnonzero(fired))
            vs.append(model.v.copy()); gs.append(model.g.copy())
        np.testing.assert_allclose(np.asarray(vs).T, monitor.v/b.mV, atol=1e-10)
        np.testing.assert_allclose(np.asarray(gs).T, monitor.g/b.mV, atol=1e-10)
        self.assertEqual(events, list(zip(np.rint(spikes.t/b.defaultclock.dt).astype(int), spikes.i[:].tolist())))


if __name__ == '__main__': unittest.main()
