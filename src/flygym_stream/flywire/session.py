"""Block-coupled Eon experiment shared by the streaming adapter.

The original 0.1 ms Shiu steps, annotation audit, Poisson targets and
15 ms brain-then-body schedule are retained. No body/hosting imports here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .eon_interface import EonReadout, integer_steps, validate_annotations
from .load_network import load_network


class EonSession:
    def __init__(self, path, backend='gpu', seed=0, condition='sensory'):
        self.path = Path(path).resolve()
        self.profile = json.loads(self.path.read_text(encoding='utf-8-sig'))
        self.base = (self.path.parent / self.profile['base_config']).resolve()
        self.cfg = json.loads(self.base.read_text(encoding='utf-8-sig'))
        self.seed, self.condition = seed, condition
        if backend not in ('cpu', 'gpu'):
            raise ValueError('Neural backend must be cpu or gpu.')
        if condition not in ('sensory', 'no-input', 'direct-output', 'disconnect-output'):
            raise ValueError('Unknown Flywire intervention.')
        self.cfg['neural'].update(dynamics='shiu', backend=backend, seed=seed)
        p = self.profile
        for name in ('sync_ms', 'forward_gain_per_hz', 'turn_gain_per_hz', 'maximum_drive'):
            if not np.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be positive and finite.')
        if not np.isfinite(p['sensory_hz']) or not 0 <= p['sensory_hz'] <= 1000:
            raise ValueError('sensory_hz must be in [0, 1000].')
        stimulus = self.cfg['stimulus']
        if not np.isfinite(stimulus['odor_sigma_mm']) or stimulus['odor_sigma_mm'] <= 0:
            raise ValueError('odor_sigma_mm must be positive and finite.')
        source = np.asarray(stimulus['source_xy_mm'])
        if source.shape != (2,) or not np.isfinite(source).all():
            raise ValueError('source_xy_mm requires two finite coordinates.')
        self.interval = p['sync_ms'] / 1000
        self.neural_dt = .0001
        self.neural_steps = integer_steps(self.interval, self.neural_dt)
        validate_annotations(p, self.cfg, self.base.parent)
        self.cfg['populations']['output_left'] = p['readouts']['steer_left']
        self.cfg['populations']['output_right'] = list(dict.fromkeys(
            x for name, ids in p['readouts'].items() if name != 'steer_left' for x in ids))
        self.brain, populations, self.audit = load_network(self.cfg, self.base.parent)
        b = self.brain
        self.decoder = EonReadout(b, p)
        self.inputs = [b.xp.asarray(b.indices(populations[key]))
                       for key in ('input_left', 'input_right')]
        targets = populations['input_left'] + populations['input_right']
        if condition == 'direct-output':
            targets = list(dict.fromkeys(x for key in ('forward', 'steer_left', 'steer_right')
                                        for x in p['readouts'][key]))
        b.set_input_targets(targets)
        self.targets = b.xp.asarray(b.indices(targets))
        self.counts = b.xp.zeros(len(b.ids), dtype=np.int64)
        self.external_rates = b.xp.zeros(len(b.ids))
        self.reset()

    def reset(self):
        """Reset neural time, delays and RNG without reloading the connectome."""
        b = self.brain
        b.v.fill(-52.)
        b.g.fill(0)
        b.rates_hz.fill(0)
        b.last_spike.fill(-1000000000)
        b.queue, b.dt, b.tick = None, None, 0
        b.rng = b.xp.random.RandomState(self.seed)
        self.counts.fill(0)
        self.external_rates.fill(0)
        self.blocks = 0
        self.drive = np.zeros(2)
        self.rates = {}
        self.spikes = 0

    def advance(self, antenna_xy):
        """Sample odor, evolve one complete neural block, hold the motor drive."""
        b, p = self.brain, self.profile
        xy = np.asarray(antenna_xy)
        if xy.shape != (2, 2) or not np.isfinite(xy).all():
            raise FloatingPointError('Expected finite bilateral antenna positions.')
        stimulus = self.cfg['stimulus']
        d2 = np.sum((xy - stimulus['source_xy_mm']) ** 2, axis=1)
        self.levels = np.exp(-d2 / (2 * stimulus['odor_sigma_mm'] ** 2))
        self.external_rates.fill(0)
        if self.condition == 'no-input':
            self.levels[:] = 0
        elif self.condition == 'direct-output':
            self.external_rates[self.targets] = p['sensory_hz']
            self.levels[:] = 0
        else:
            for group, level in zip(self.inputs, self.levels):
                self.external_rates[group] += p['sensory_hz'] * level
        self.counts.fill(0)
        for _ in range(self.neural_steps):
            self.counts += b.step_poisson(self.external_rates, self.neural_dt)
        b.check_health()
        self.drive, self.rates = self.decoder.read(self.counts, self.interval)
        if self.condition == 'disconnect-output':
            self.drive[:] = 0
        self.spikes = int(b.to_cpu(self.counts.sum()))
        self.blocks += 1
        return self.drive

    def manifest(self):
        paths = [self.path, self.base, Path(__file__),
                 *[Path(__file__).with_name(name) for name in
                   ('neural_model.py', 'shiu_model.py', 'eon_interface.py',
                    'load_network.py', 'run_hybrid.py', 'streaming.py')]]
        paths += [(self.base.parent / self.cfg['data'][key]).resolve()
                  for key in ('neurons_csv', 'connections_csv')]
        hashes = {}
        for path in paths:
            with path.open('rb') as stream:
                hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return dict(profile=self.profile, config=self.cfg, audit=self.audit,
                    condition=self.condition, seed=self.seed, sha256=hashes,
                    neural_backend=self.brain.backend, body_backend='cpu',
                    scope='Local Eon locomotion approximation; feed/groom readouts only.')
