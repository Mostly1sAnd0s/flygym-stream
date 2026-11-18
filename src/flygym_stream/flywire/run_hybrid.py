"""Run toy/BANC neural dynamics connected to FlyGym 2.1's hybrid controller."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time
import numpy as np
from .load_network import load_network
from .neural_model import MotorDecoder


def inspect_data(cfg, root):
    import pandas as pd
    for key in ('neurons_csv', 'connections_csv'):
        path = (root / cfg['data'][key]).resolve()
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=5)
        print(f'\n{path}\nHEADERS: {list(frame.columns)}')
        print(frame.to_string(index=False))


def build_body(source_xy, *, tracking_camera=False):
    import mujoco
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld
    from flygym.anatomy import ContactBodiesPreset
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import (
        HybridTurningController, LocomotionAction, PreprogrammedSteps,
        apply_locomotion_action, make_locomotion_fly,
    )
    fly = make_locomotion_fly(name='hybrid_fly', add_adhesion=True, colorize=True)
    if tracking_camera:
        fly.add_tracking_camera(name='trackingcam')
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.8], Rotation3D('quat', [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                  add_ground_contact_sensors=False)
    world.mjcf_root.worldbody.add_geom(
        name='odor_source_marker', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[*source_xy, 0.025], size=[0.6, 0.025, 0], rgba=[0.2, 0.9, 0.3, 1],
        contype=0, conaffinity=0)
    sim = Simulation(world)
    steps = PreprogrammedSteps()
    dofs = fly.get_actuated_jointdofs_order('position')
    controller = HybridTurningController(timestep=sim.timestep,
        preprogrammed_steps=steps, output_dof_order=dofs)
    sim.reset(); controller.reset(seed=0)
    action = LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dofs),
                             adhesion_onoff=np.ones(6, dtype=bool))
    apply_locomotion_action(sim, fly.name, action)
    sim.warmup()
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    def body_id(suffix):
        matches = [i for i in range(1, sim.mj_model.nbody)
                   if sim.mj_model.body(i).name.endswith('/' + suffix)]
        if len(matches) != 1:
            raise ValueError(f'Expected one body ending /{suffix}; found {matches}')
        return matches[0]
    thorax = body_id('c_thorax')
    # Body origins approximate bilateral odor sampling locations, not receptor geometry.
    antennae = [body_id('l_funiculus'), body_id('r_funiculus')]
    return sim, fly, controller, thorax, antennae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['demo', 'banc', 'controller'], default='demo')
    parser.add_argument('--config', default=str(Path(__file__).with_name('hybrid_config.json')))
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default=None,
                        help='Neural backend (default: config setting, otherwise gpu)')
    parser.add_argument('--seconds', type=float, default=5.0)
    parser.add_argument('--dynamics', choices=['conductance', 'shiu'], default=None)
    parser.add_argument('--sensory-hz', type=float, default=150., help='Shiu Poisson rate at unit stimulus')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-current-pa', type=float, default=0.0,
                        help='Diagnostic current injected directly into both motor output populations')
    parser.add_argument('--motor-dt-ms', type=float, default=1.0,
                        help='Hold motor commands between readouts (default: 1 ms; 0.1 restores per-step readout)')
    parser.add_argument('--input', choices=['pulse', 'odor'], default='pulse')
    parser.add_argument('--no-viewer', action='store_true')
    parser.add_argument('--neural-only', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    root = cfg_path.parent
    if args.inspect:
        inspect_data(cfg, root)
        return
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('--seconds must be positive and finite.')
    if not np.isfinite(args.motor_dt_ms) or args.motor_dt_ms <= 0:
        raise ValueError('--motor-dt-ms must be positive and finite.')
    if args.neural_only and (args.input == 'odor' or args.mode == 'controller'):
        raise ValueError('--neural-only supports demo/banc with pulse input only.')
    if args.mode == 'controller' and args.input != 'pulse':
        raise ValueError('Controller test uses fixed commands, not odor input.')
    if args.mode == 'controller' and args.output_current_pa:
        raise ValueError('Output stimulation requires demo or banc mode.')
    args.backend = args.backend or cfg['neural'].get('backend', 'gpu')
    if not np.isfinite(args.output_current_pa) or args.output_current_pa < 0:
        raise ValueError('--output-current-pa must be finite and nonnegative')
    if args.output_current_pa:
        print('DIRECT OUTPUT STIMULATION TEST: bypasses the sensory pathway.', flush=True)
    cfg['neural']['backend'] = args.backend
    cfg['neural']['dynamics'] = args.dynamics or cfg['neural'].get('dynamics', 'conductance')
    cfg['neural']['seed'] = args.seed
    is_shiu = cfg['neural']['dynamics'] == 'shiu'
    if not np.isfinite(args.sensory_hz) or not 0 <= args.sensory_hz <= 1000:
        raise ValueError('--sensory-hz must be in [0,1000]')
    if is_shiu and args.output_current_pa:
        raise ValueError('pA output stimulation is only supported by conductance dynamics.')
    p = cfg['neural']
    for key in ('unit_conductance_nS', 'capacitance_pF', 'leak_nS', 'synapse_decay_ms',
                'rate_filter_ms', 'refractory_ms'):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if not 0 < p['release_fraction'] <= 1 or p['threshold_mV'] <= p['reset_mV']:
        raise ValueError('Invalid release fraction or spike threshold/reset.')
    for key in ('switch_every_s', 'odor_sigma_mm'):
        value = cfg['stimulus'][key]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    sim = None
    brain = None
    if args.mode != 'controller':
        brain, populations, report = load_network(cfg, root, demo=args.mode == 'demo')
        decoder = MotorDecoder(brain, populations, cfg['decoder'])
        input_groups = [brain.xp.asarray(brain.indices(populations[k])) for k in ('input_left', 'input_right')]
        if set(populations['output_left']) & set(populations['output_right']):
            raise ValueError('Left/right output populations overlap; choose distinct readouts.')
        print(json.dumps(report, indent=2))
        current = brain.xp.zeros(len(brain.ids))
        print(f'Neural backend: {brain.backend}; precision: float64', flush=True)
    else:
        report = {'mode': 'FIXED CONTROLLER TEST (NO NEURAL MODEL)'}
    output = root / 'Outputs' / ('hybrid_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True)
    (output / 'run_config.json').write_text(json.dumps({'arguments': vars(args), 'config': cfg,
                                                      'network_report': report}, indent=2))
    print(f'Run output: {output}', flush=True)
    if args.mode == 'demo':
        print('SYNTHETIC FOUR-NEURON DEMO: not BANC, not a biological behavior prediction.', flush=True)
    elif args.mode == 'banc':
        print('BANC model uses assumed physiology and manually configured input/output populations.', flush=True)
    viewer_context = nullcontext(None)
    if args.neural_only:
        dt = 0.0001
    else:
        print('Building body (first use may download meshes)...', flush=True)
        sim, fly, controller, thorax, antennae = build_body(cfg['stimulus']['source_xy_mm'])
        dt = sim.timestep
        from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action
        if not args.no_viewer:
            import mujoco.viewer
            viewer_context = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data)
    nsteps = int(np.ceil(args.seconds / dt))
    render_every = max(1, round(1 / (60 * dt)))
    log_every = max(1, round(0.01 / dt))
    motor_every = max(1, round(args.motor_dt_ms / (1000 * dt)))
    print(f'Motor readout interval: {motor_every * dt * 1000:g} ms; neural timestep: {dt * 1000:g} ms', flush=True)
    if brain is not None:
        brain.synchronize()
    completed_steps = 0
    neural_wall = body_wall = 0.0
    motor_reads = 0
    drive = rates = np.zeros(2)
    previous_levels = None
    start_wall = time.perf_counter()
    try:
        with (output / 'trace.csv').open('w', newline='') as stream, viewer_context as viewer:
            writer = csv.writer(stream)
            writer.writerow(['time_s', 'input_L', 'input_R', 'rate_L_hz', 'rate_R_hz',
                             'drive_L', 'drive_R', 'x_mm', 'y_mm', 'z_mm', 'spikes_since_log'])
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = sim.mj_data.xpos[thorax]
                    viewer.cam.distance = 9
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -35
            spikes_since_log = brain.xp.zeros((), dtype=np.int64) if brain is not None else 0
            for step in range(nsteps):
                if viewer and not viewer.is_running():
                    break
                t = step * dt
                stimulus = cfg['stimulus']
                if args.input == 'pulse':
                    # Alternating controlled input, no environmental feedback in pulse mode.
                    levels = np.array([1.0, 0.35]) if int(t / stimulus['switch_every_s']) % 2 == 0 else np.array([0.35, 1.0])
                else:
                    xy = sim.mj_data.xpos[antennae, :2]
                    source = np.asarray(stimulus['source_xy_mm'])
                    d2 = np.sum((xy - source) ** 2, axis=1)
                    levels = np.exp(-d2 / (2 * stimulus['odor_sigma_mm'] ** 2))
                if brain is not None:
                    tic = time.perf_counter()
                    # Pulse input is constant between switches: reuse its device buffer.
                    if previous_levels is None or not np.array_equal(levels, previous_levels):
                        current.fill(0)
                        for indices, level in zip(input_groups, levels):
                            current[indices] += args.sensory_hz * level if is_shiu else stimulus['baseline_pA'] + stimulus['amplitude_pA'] * level
                        for indices in decoder.groups:
                            current[indices] += args.output_current_pa
                        previous_levels = levels.copy()
                    spikes_since_log += (brain.step_poisson(current, dt) if is_shiu else brain.step(current, dt)).sum()
                    if step % motor_every == 0:
                        drive, rates = decoder.read(brain.rates_hz)
                        motor_reads += 1
                    neural_wall += time.perf_counter() - tic
                else:
                    drive = np.array([1.2, 0.4]) if t < args.seconds / 2 else np.array([0.4, 1.2])
                    rates = np.array([np.nan, np.nan])
                if sim:
                    tic = time.perf_counter()
                    obs = HybridControllerObservation.from_sim(sim, fly.name)
                    action = controller.step(drive, obs)
                    apply_locomotion_action(sim, fly.name, action)
                    sim.step()
                    if not np.isfinite(sim.mj_data.qpos).all():
                        raise FloatingPointError('Non-finite body state.')
                    position = sim.mj_data.xpos[thorax].copy()
                    body_wall += time.perf_counter() - tic
                else:
                    position = [np.nan] * 3
                completed_steps = step + 1
                if step % log_every == 0 or step == nsteps - 1:
                    if brain is not None:
                        tic = time.perf_counter()
                        brain.check_health()
                        neural_wall += time.perf_counter() - tic
                    logged_spikes = int(brain.to_cpu(spikes_since_log)) if brain is not None else 0
                    writer.writerow([t + dt, *levels, *rates, *drive, *position, logged_spikes])
                    if brain is not None:
                        spikes_since_log.fill(0)
                if step % max(1, round(1 / dt)) == 0:
                    print(f'{t:.1f}s | rates={rates.round(1)} Hz | drive={drive.round(2)}', flush=True)
                if viewer and step % render_every == 0:
                    with viewer.lock():
                        viewer.cam.lookat[:] = sim.mj_data.xpos[thorax]
                    viewer.sync()
                    remaining = (step + 1) * dt - (time.perf_counter() - start_wall)
                    if remaining > 0:
                        time.sleep(min(remaining, 1 / 60))
    finally:
        if sim:
            sim.close()
    if brain is not None:
        tic = time.perf_counter()
        brain.synchronize()
        neural_wall += time.perf_counter() - tic
    elapsed = time.perf_counter() - start_wall
    simulated = completed_steps * dt
    timing = dict(simulated_s=simulated, wall_s=elapsed,
                  neural_host_and_wait_s=neural_wall, body_compute_s=body_wall,
                  motor_reads=motor_reads, motor_interval_ms=motor_every * dt * 1000,
                  neural_dt_ms=dt * 1000,
                  timing_note='Neural time measures host submission and waits, not isolated GPU kernel time; GPU work can overlap body work.')
    (output / 'timing.json').write_text(json.dumps(timing, indent=2), encoding='utf-8')
    print(f'Neural submission/waits: {neural_wall:.2f}s; body: {body_wall:.2f}s; motor readouts: {motor_reads}', flush=True)
    print(f'Simulation loop: {simulated:.4f}s simulated in {elapsed:.2f}s wall time; {simulated / elapsed:.4f}x real time')
    print(f'Finished. Read {output / "trace.csv"}')


if __name__ == '__main__':
    main()
