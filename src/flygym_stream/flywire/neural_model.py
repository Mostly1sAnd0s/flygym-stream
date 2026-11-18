"""Sparse conductance LIF model. Parameters are hypotheses, not measured BANC physiology.

Units: mV, ms, pF, nS, pA internally; public dt is seconds.
All outgoing synapses of one source share receptor kinetics and reversal class.
This permits a per-source saturated gate plus sparse matrix multiplication.
It does NOT implement receptor-specific co-transmission or neuromodulation.
"""
import numpy as np
from scipy.sparse import coo_matrix


class NeuralModel:
    def __init__(self, ids, pre, post, counts, source_class, params):
        self.ids = list(ids)
        self.index = {value: i for i, value in enumerate(ids)}
        self.p = params
        self.backend = params.get('backend', 'gpu')
        if self.backend not in ('cpu', 'gpu'):
            raise ValueError('backend must be cpu or gpu')
        self.xp = np
        if self.backend == 'gpu':
            try:
                import cupy as cp
                from cupyx.scipy.sparse import csr_matrix as gpu_csr
                cp.cuda.Device().use()
                cp.zeros(1).sum().get()
            except Exception as exc:
                raise RuntimeError('GPU initialization failed. Check your CuPy installation and driver, or use --backend cpu.') from exc
            self.xp = cp
        n = len(ids)
        pre, post = np.asarray(pre, int), np.asarray(post, int)
        counts = np.asarray(counts, dtype=float)
        if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
            raise ValueError('Synapse counts must be finite and positive.')
        source_class = np.asarray(source_class)
        matrices = []
        for kind in ('exc', 'inh'):
            mask = source_class[pre] == kind
            weight = counts[mask] * params['unit_conductance_nS']
            matrix = coo_matrix((weight, (post[mask], pre[mask])), shape=(n, n)).tocsr()
            matrix.sum_duplicates()
            matrices.append(gpu_csr(matrix) if self.backend == 'gpu' else matrix)
        self.exc, self.inh = matrices
        self.v = self.xp.full(n, params['rest_mV'], dtype=float)
        self.gate = self.xp.zeros(n)
        self.refractory_ms = self.xp.zeros(n)
        self.rates_hz = self.xp.zeros(n)

    def to_cpu(self, value):
        return self.xp.asnumpy(value) if self.backend == 'gpu' else np.asarray(value)

    def synchronize(self):
        if self.backend == 'gpu':
            self.xp.cuda.get_current_stream().synchronize()

    def check_health(self):
        if not bool(self.to_cpu(self.xp.isfinite(self.v).all())):
            raise FloatingPointError('Non-finite neural voltage: check parameters.')

    def indices(self, ids):
        missing = [x for x in ids if x not in self.index]
        if missing:
            raise ValueError(f'Configured neuron IDs not in loaded network: {missing[:10]}')
        return np.asarray([self.index[x] for x in ids], dtype=int)

    def step(self, current_pA, dt_s):
        p = self.p
        h = dt_s * 1000.0
        if not 0 < h <= 1.0:
            raise ValueError('Use neural timesteps in (0, 1] ms; 0.1 ms is the default.')
        # Existing release gates contribute now; new spikes act next timestep.
        self.gate *= np.exp(-h / p['synapse_decay_ms'])
        ge, gi = self.exc @ self.gate, self.inh @ self.gate
        leak = p['leak_nS']
        total = leak + ge + gi
        equilibrium = (leak * p['rest_mV'] + ge * p['exc_reversal_mV']
                       + gi * p['inh_reversal_mV'] + current_pA) / total
        xp = self.xp
        ready = self.refractory_ms <= 0
        self.refractory_ms = xp.maximum(0, self.refractory_ms - h)
        voltage = equilibrium + (self.v - equilibrium) * xp.exp(
            -h * total / p['capacitance_pF'])
        self.v = xp.where(ready, voltage, p['reset_mV'])
        spikes = ready & (self.v >= p['threshold_mV'])
        self.v = xp.where(spikes, p['reset_mV'], self.v)
        self.refractory_ms = xp.where(spikes, p['refractory_ms'], self.refractory_ms)
        self.gate += xp.where(spikes, (1 - self.gate) * p['release_fraction'], 0.0)
        tau_s = p['rate_filter_ms'] / 1000
        self.rates_hz *= np.exp(-dt_s / tau_s)
        self.rates_hz += spikes / tau_s
        # CPU checks each step. GPU runner checks at logging boundaries to
        # avoid a full device synchronization for this diagnostic every step.
        if self.backend == 'cpu':
            self.check_health()
        return spikes


class MotorDecoder:
    def __init__(self, brain, populations, params):
        if np.any(np.asarray(params.get('baseline', [0.0, 0.0])) != 0) or params['minimum'] != 0:
            raise ValueError('Brain-controlled movement requires zero decoder baseline and minimum.')
        self.brain = brain
        self.groups = [brain.xp.asarray(brain.indices(populations[k])) for k in ('output_left', 'output_right')]
        self.p = params

    def read(self, rates):
        population_rates = self.brain.to_cpu(self.brain.xp.stack([rates[g].mean() for g in self.groups]))
        drive = np.asarray(self.p['gain_per_hz']) * population_rates
        return np.clip(drive, self.p['minimum'], self.p['maximum']), population_rates
