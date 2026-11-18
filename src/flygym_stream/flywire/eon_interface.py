"""Auditable local approximation of Eon's published descending readout interface."""
import numpy as np
import pandas as pd


def integer_steps(seconds, dt):
    steps = round(seconds / dt)
    if steps < 1 or not np.isclose(steps * dt, seconds, rtol=0, atol=1e-10):
        raise ValueError('Duration must be a positive integer multiple of timestep.')
    return steps


def validate_annotations(profile, cfg, folder):
    attrs = pd.read_csv(folder / cfg['data']['neurons_csv'], dtype=str,
                        keep_default_na=False).set_index(cfg['columns']['neuron_id'])
    for name, ids in profile['readouts'].items():
        expected = profile['annotation_types'][name]
        if not ids or len(ids) != len(set(ids)) or len(ids) != len(expected):
            raise ValueError(f'Invalid readout population: {name}')
        for neuron, cell_type in zip(ids, expected):
            row = attrs.loc[neuron]
            labels = {row['Primary Cell Type'], *row['Alternative Cell Type(s)'].split(',')}
            if cell_type not in {x.strip() for x in labels}:
                raise ValueError(f'{neuron} does not have expected annotation {cell_type}')
            if name in ('steer_left', 'steer_right') and row['Soma side'] != name.split('_')[1]:
                raise ValueError(f'{neuron} has unexpected side')


def decode_rates(rates, profile):
    """Local gain choices; positive left steering raises right leg drive.

    oDN1's positive speed gain follows the coarse Eon description, not a fitted
    physiological rule. The associated primary study also implicates halting.
    """
    forward = profile['forward_gain_per_hz'] * rates['forward']
    turn = profile['turn_gain_per_hz'] * (rates['steer_left'] - rates['steer_right'])
    return np.clip([forward - turn, forward + turn], 0, profile['maximum_drive'])


class EonReadout:
    def __init__(self, brain, profile):
        self.brain, self.profile = brain, profile
        self.groups = {name: brain.xp.asarray(brain.indices(ids))
                       for name, ids in profile['readouts'].items()}

    def read(self, counts, duration):
        # Average spike counts over the just-completed brain block. Eon's exact
        # rate estimator is not published; keep this choice explicit.
        values = self.brain.to_cpu(self.brain.xp.stack(
            [counts[g].mean() / duration for g in self.groups.values()]))
        rates = dict(zip(self.groups, map(float, values)))
        return decode_rates(rates, self.profile), rates
