"""Verify a matched set of four completed neural control trials."""
import argparse
import json
from pathlib import Path
import numpy as np


def verify(paths):
    runs, manifests, arrays, traces = {}, {}, {}, {}
    for path in paths:
        m = json.loads((path / 'manifest.json').read_text())
        condition = m['arguments']['condition']
        if condition in runs:
            raise ValueError(f'Duplicate condition: {condition}')
        s = json.loads((path / 'summary.json').read_text())
        if s['status'] != 'completed' or s['completed_blocks'] != s['requested_blocks']:
            raise ValueError(f'Incomplete run: {path}')
        runs[condition] = dict(path=str(path), **s)
        manifests[condition] = m
        with np.load(path / 'spike_counts.npz') as values:
            arrays[condition] = {key: values[key].copy() for key in ('ids', 'counts')}
        traces[condition] = np.genfromtxt(path / 'trace.csv', delimiter=',', names=True)
    if set(runs) != {'sensory', 'no-input', 'direct-output', 'disconnect-output'}:
        raise ValueError('Supply one completed run for each of the four conditions.')
    reference = manifests['sensory']
    for condition, m in manifests.items():
        for key in ('resolved_config', 'profile', 'neural_dt_s', 'sync_s', 'effective_sensory_hz'):
            if m[key] != reference[key]:
                raise ValueError(f'{condition} differs in {key}')
        for key in ('seed', 'seconds', 'neural_only'):
            if m['arguments'][key] != reference['arguments'][key]:
                raise ValueError(f'{condition} differs in {key}')
        # Compare the actual neural implementation and input data, allowing
        # changes to runner metadata logging between runs.
        for path, digest in reference['sha256'].items():
            if Path(path).name in ('shiu_model.py', 'load_network.py', 'eon_interface.py',
                                  'neurons.csv', 'connections_princeton.csv'):
                if m['sha256'].get(path) != digest:
                    raise ValueError(f'{condition} source/data mismatch: {path}')
        np.testing.assert_array_equal(arrays[condition]['ids'], arrays['sensory']['ids'])
    np.testing.assert_array_equal(arrays['sensory']['counts'], arrays['disconnect-output']['counts'])
    for name in traces['sensory'].dtype.names:
        if name.endswith('_hz') or name == 'total_spikes':
            np.testing.assert_array_equal(traces['sensory'][name], traces['disconnect-output'][name])
    checks = dict(
        matched_neural_trials=True,
        identical_neuron_counts_and_block_readouts_after_motor_disconnection=True,
        no_input_quiet=runs['no-input']['total_spikes'] == 0 and runs['no-input']['maximum_drive'] == 0,
        direct_output_commands=runs['direct-output']['maximum_drive'] > 0,
        disconnected_commands_zero=runs['disconnect-output']['maximum_drive'] == 0,
        sensory_propagation=runs['sensory']['maximum_drive'] > 0)
    if not all(checks.values()):
        raise ValueError(f'Control checks failed: {checks}')
    return dict(runs=runs, checks=checks,
                scope='Neural control checks only; no biological or embodied validation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.runs)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
