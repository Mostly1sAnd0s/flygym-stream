"""Eon-inspired brain-then-body block coupling on the local connectome.

Default: headless odor-driven locomotion; --neural-only runs prescribed input.
See EON_METHODOLOGY.md for matched components and unresolved reproduction gaps.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import time
import numpy as np
from .eon_interface import EonReadout, integer_steps, validate_annotations
from .load_network import load_network


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('eon_config.json'))
    parser.add_argument('--seconds', type=float, default=1.5)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--neural-only', action='store_true')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--condition', choices=['sensory', 'no-input', 'direct-output', 'disconnect-output'], default='sensory')
    parser.add_argument('--sensory-hz', type=float, default=None)
    args = parser.parse_args()
    path = args.config.resolve()
    profile = json.loads(path.read_text(encoding='utf-8-sig'))
    base = (path.parent / profile['base_config']).resolve()
    cfg = json.loads(base.read_text(encoding='utf-8-sig'))
    cfg['neural'].update(dynamics='shiu', seed=args.seed,
                         backend=args.backend or cfg['neural'].get('backend', 'gpu'))
    hz = profile['sensory_hz'] if args.sensory_hz is None else args.sensory_hz
    if not np.isfinite(hz) or not 0 <= hz <= 1000:
        raise ValueError('sensory-hz must be finite and in [0,1000].')
    for name in ('sync_ms', 'forward_gain_per_hz', 'turn_gain_per_hz', 'maximum_drive'):
        if not np.isfinite(profile[name]) or profile[name] <= 0:
            raise ValueError(f'{name} must be positive and finite.')
    neural_dt = 0.0001
    interval = profile['sync_ms'] / 1000
    neural_steps = integer_steps(interval, neural_dt)
    blocks = integer_steps(args.seconds, interval)
    if args.neural_only and args.viewer:
        raise ValueError('--viewer requires a body.')
    validate_annotations(profile, cfg, base.parent)
    # Audit every selected readout using the loader's existing output checks.
    cfg['populations']['output_left'] = profile['readouts']['steer_left']
    cfg['populations']['output_right'] = list(dict.fromkeys(
        x for name, ids in profile['readouts'].items() if name != 'steer_left' for x in ids))
    brain, populations, audit = load_network(cfg, base.parent)
    print(f'Neural backend: {brain.backend}; precision: float64', flush=True)
    if brain.backend == 'gpu':
        device = brain.xp.cuda.runtime.getDeviceProperties(brain.xp.cuda.Device().id)
        name = device['name']
        if isinstance(name, bytes):
            name = name.decode()
        print(f'GPU: {name}', flush=True)
    decoder = EonReadout(brain, profile)
    inputs = [brain.xp.asarray(brain.indices(populations[key]))
              for key in ('input_left', 'input_right')]
    targets = populations['input_left'] + populations['input_right']
    if args.condition == 'direct-output':
        targets = list(dict.fromkeys(x for key in ('forward', 'steer_left', 'steer_right')
                                    for x in profile['readouts'][key]))
    brain.set_input_targets(targets)
    target_indices = brain.xp.asarray(brain.indices(targets))
    output = path.parent / 'Outputs' / ('eon_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True)
    source_files = [path, base, Path(__file__), Path(__file__).with_name('shiu_model.py'),
                    Path(__file__).with_name('eon_interface.py'), Path(__file__).with_name('load_network.py'),
                    Path(__file__).with_name('neural_model.py'), Path(__file__).with_name('run_hybrid.py')]
    source_files += [(base.parent / cfg['data'][key]).resolve() for key in ('neurons_csv', 'connections_csv')]
    versions = {}
    for package in ('numpy', 'scipy', 'pandas', 'flygym', 'mujoco', 'brian2', 'cupy-cuda12x'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    manifest = dict(arguments={**vars(args), 'config': str(path)}, profile=profile,
                    resolved_config=cfg, effective_sensory_hz=hz, network_audit=audit,
                    packages=versions,
                    effective_neural_constants=dict(rest_mV=-52, reset_mV=-52, threshold_mV=-45,
                        membrane_ms=20, synapse_decay_ms=5, refractory_ms=2.2,
                        delay_ms=1.8, weight_per_synapse_mV=.275, input_event_mV=68.75),
                    sha256={str(p.resolve()): sha256(p) for p in source_files},
                    neural_dt_s=neural_dt, sync_s=interval,
                    scope='Local locomotion approximation; feed/groom readouts only; not biological validation.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Output: {output}', flush=True)
    sim = None
    start = time.perf_counter()
    population_counts = brain.xp.zeros(len(brain.ids), dtype=np.int64)
    total_counts = brain.xp.zeros(len(brain.ids), dtype=np.int64)
    external_rates = brain.xp.zeros(len(brain.ids))
    completed = 0
    max_drive = 0.
    initial_position = final_position = None
    try:
        viewer_context = nullcontext(None)
        if not args.neural_only:
            from .run_hybrid import build_body
            from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action
            sim, fly, controller, thorax, antennae = build_body(cfg['stimulus']['source_xy_mm'])
            body_steps = integer_steps(interval, sim.timestep)
            initial_position = sim.mj_data.xpos[thorax].copy()
            if args.viewer:
                import mujoco.viewer
                viewer_context = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data)
        with (output / 'trace.csv').open('w', newline='') as stream, viewer_context as viewer:
            writer = csv.writer(stream)
            names = list(profile['readouts'])
            writer.writerow(['time_s', 'input_left', 'input_right', *[k + '_hz' for k in names],
                             'drive_left', 'drive_right', 'x_mm', 'y_mm', 'z_mm', 'total_spikes'])
            for block in range(blocks):
                if viewer and not viewer.is_running():
                    break
                # Sample environment once, evolve brain, read outputs, then evolve
                # body for the SAME interval with a held motor command.
                if sim is None:
                    levels = np.array([1., .35])
                else:
                    xy = sim.mj_data.xpos[antennae, :2]
                    d2 = np.sum((xy - cfg['stimulus']['source_xy_mm']) ** 2, axis=1)
                    levels = np.exp(-d2 / (2 * cfg['stimulus']['odor_sigma_mm'] ** 2))
                external_rates.fill(0)
                if args.condition == 'no-input':
                    levels[:] = 0
                elif args.condition == 'direct-output':
                    external_rates[target_indices] = hz
                    levels[:] = 0
                else:
                    for group, level in zip(inputs, levels):
                        external_rates[group] += hz * level
                population_counts.fill(0)
                for _ in range(neural_steps):
                    population_counts += brain.step_poisson(external_rates, neural_dt)
                brain.check_health()
                total_counts += population_counts
                drive, rates = decoder.read(population_counts, interval)
                if args.condition == 'disconnect-output':
                    drive[:] = 0
                max_drive = max(max_drive, float(drive.max()))
                position = [np.nan] * 3
                if sim is not None:
                    for _ in range(body_steps):
                        obs = HybridControllerObservation.from_sim(sim, fly.name)
                        apply_locomotion_action(sim, fly.name, controller.step(drive, obs))
                        sim.step()
                    if not np.isfinite(sim.mj_data.qpos).all():
                        raise FloatingPointError('Non-finite body state.')
                    final_position = sim.mj_data.xpos[thorax].copy()
                    position = final_position
                writer.writerow([(block+1)*interval, *levels, *[rates[k] for k in names],
                                 *drive, *position, int(brain.to_cpu(population_counts.sum()))])
                stream.flush()
                completed += 1
                if viewer:
                    with viewer.lock():
                        viewer.cam.lookat[:] = position
                        viewer.cam.distance = 9
                    viewer.sync()
                if block % max(1, round(.15/interval)) == 0:
                    print(f'{(block+1)*interval:.3f}s: {rates}; drive={drive}', flush=True)
    finally:
        if sim is not None:
            sim.close()
    counts = brain.to_cpu(total_counts)
    np.savez_compressed(output / 'spike_counts.npz', ids=np.asarray(brain.ids), counts=counts)
    summary = dict(completed_blocks=completed, requested_blocks=blocks,
                   simulated_s=completed*interval, wall_s=time.perf_counter()-start,
                   total_spikes=int(counts.sum()), active_neurons=int((counts > 0).sum()),
                   maximum_drive=max_drive,
                   displacement_mm=None if final_position is None else float(np.linalg.norm(final_position[:2]-initial_position[:2])),
                   status='completed' if completed == blocks else 'viewer_closed')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
