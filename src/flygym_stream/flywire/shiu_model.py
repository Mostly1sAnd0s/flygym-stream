"""GPU/CPU implementation of the Shiu reference equations and Brian2 schedule.

Source: philshiu/Drosophila_brain_model/model.py (reviewed 2026-11-04).
g is voltage, not conductance. This port does not reproduce the published dataset
or prove behavioral validity on the local connectivity export.
"""
import numpy as np
from .neural_model import NeuralModel


class ShiuModel(NeuralModel):
    def __init__(self, ids, pre, post, counts, source_class, params):
        p = dict(params)
        p.update(rest_mV=-52., reset_mV=-52., threshold_mV=-45.,
                 refractory_ms=2.2, synapse_decay_ms=5., unit_conductance_nS=0.275)
        super().__init__(ids, pre, post, counts, source_class, p)
        # Parent sparse constructor is reused; these matrices now have mV units.
        self.weights = self.exc - self.inh
        self.g = self.xp.zeros(len(ids))
        self.tick = 0
        self.dt = None
        self.queue = None
        self.last_spike = self.xp.full(len(ids), -1000000000, dtype=np.int64)
        self.rng = self.xp.random.RandomState(params.get('seed', 0))
        self.external = self.xp.zeros(len(ids), dtype=bool)
        self.explicit_targets = False

    def set_input_targets(self, ids):
        """Declare Poisson targets, including targets whose current rate is zero."""
        self.external.fill(False)
        self.external[self.xp.asarray(self.indices(ids))] = True
        self.explicit_targets = True

    def check_health(self):
        super().check_health()
        for state in (self.g, self.rates_hz):
            if not bool(self.to_cpu(self.xp.isfinite(state).all())):
                raise FloatingPointError('Non-finite Shiu synaptic or rate state.')

    def step(self, current_pA, dt_s):
        raise ValueError('ShiuModel uses step_poisson(rates_hz, dt_s), not pA input.')

    def step_poisson(self, rates_hz, dt_s, *, input_events=None):
        xp = self.xp
        if self.dt is None:
            if not np.isclose(dt_s, 0.0001, rtol=0, atol=1e-12):
                raise ValueError('Reference backend requires fixed 0.1 ms steps.')
            self.dt = dt_s
            self.delay_ticks = 18
            self.queue = xp.zeros((self.delay_ticks + 1, len(self.ids)))
        if not np.isclose(dt_s, self.dt, rtol=0, atol=1e-12):
            raise ValueError('Reference timestep cannot change during a trial.')
        rates = xp.asarray(rates_hz)
        if rates.shape != (len(self.ids),):
            raise ValueError('Poisson rates require one value per neuron.')
        if not bool(self.to_cpu((xp.isfinite(rates) & (rates >= 0) & (rates * dt_s <= 1)).all())):
            raise ValueError('Poisson rates must be finite and in [0, 1/dt].')
        # External targets have no refractory period, as in reference poi().
        if not self.explicit_targets:
            self.external |= rates > 0
        elif bool(self.to_cpu(((rates > 0) & ~self.external).any())):
            raise ValueError('Positive rate outside declared Poisson targets.')
        events = (self.rng.random_sample(len(self.ids)) < rates * dt_s
                  if input_events is None else xp.asarray(input_events, dtype=bool))
        if events.shape != rates.shape or bool(self.to_cpu((events & ~self.external).any())):
            raise ValueError('Input events must match the declared Poisson targets.')
        ready = (self.tick - self.last_spike >= 22) | self.external
        h = dt_s * 1000
        a, b = np.exp(-h / 20.), np.exp(-h / 5.)
        # Exact linear solution, with v and g frozen while refractory.
        v = -52. + (self.v + 52.) * a + self.g * (a-b) / 3.
        self.v = xp.where(ready, v, self.v)
        self.g = xp.where(ready, self.g * b, self.g)
        spikes = ready & (self.v > -45.)
        # Brian schedule: thresholds, synapses/PoissonInput, then resets.
        slot = self.tick % len(self.queue)
        self.g += self.queue[slot]
        self.queue[slot].fill(0)
        self.queue[(self.tick+self.delay_ticks) % len(self.queue)] += self.weights @ spikes.astype(float)
        self.v += events * 68.75
        self.v = xp.where(spikes, -52., self.v)
        self.g = xp.where(spikes, 0., self.g)
        self.last_spike = xp.where(spikes, self.tick, self.last_spike)
        tau_s = self.p['rate_filter_ms'] / 1000
        self.rates_hz *= np.exp(-dt_s/tau_s)
        self.rates_hz += spikes/tau_s
        self.tick += 1
        return spikes
