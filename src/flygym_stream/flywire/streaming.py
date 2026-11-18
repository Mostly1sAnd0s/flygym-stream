"""Adapt the original Flywire body and Eon brain to the existing server.

Transport, authentication/bind rules, renderer, idle pacing and deployment
remain owned by flygym-stream. The original hybrid controller owns actuators.
"""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import mujoco as mj
import numpy as np

from flygym_stream.protocol import Camera, Neutral, Orbit
from flygym_stream.scene import Scene
from flygym_stream.simulation import MuJoCoBackend
from .eon_interface import integer_steps
from .run_hybrid import build_body
from .session import EonSession

log = logging.getLogger(__name__)


class FlywireScene(Scene):
    def __init__(self, sim, fly):
        self.sim = sim
        m, d = sim.mj_model, sim.mj_data
        d.time = 0.0
        super().__init__(m, d, range(m.nu), target_site=fly.name,
                         camera_name=f'{fly.name}/trackingcam',
                         distance_mm=9000, rest_anchor=False)
        self._neutral_ctrl = d.ctrl.copy()
        self.initial = mj.MjData(m)
        mj.mj_copyData(self.initial, m, d)

    def step(self):
        self.sim.step()

    def reset(self):
        mj.mj_copyData(self.mj_data, self.mj_model, self.initial)
        mj.mj_forward(self.mj_model, self.mj_data)


class FlywireBackend(MuJoCoBackend):
    def __init__(self, scene, session, fly, controller, thorax, antennae, output):
        super().__init__(scene)
        self.session, self.fly, self.controller = session, fly, controller
        self.thorax, self.antennae = thorax, antennae
        self.body_steps = integer_steps(session.interval, self.dt)
        self.body_tick = 0
        self.name = f'flywire-{session.brain.backend}-brain/cpu-body'
        self.controller.reset(seed=session.seed)
        self.output = Path(output) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.output.mkdir(parents=True)
        (self.output / 'manifest.json').write_text(json.dumps(session.manifest(), indent=2))
        self.trace = (self.output / 'trace.jsonl').open('w', encoding='utf-8')
        self.episode = 0
        log.warning('Flywire enabled: %s; run artifacts: %s', self.name, self.output)

    def step(self, dt):
        if abs(dt - self.dt) > 1e-12:
            raise ValueError('Body timestep cannot change.')
        from flygym_demo.complex_terrain import (
            HybridControllerObservation, apply_locomotion_action,
        )
        sim = self._scene.sim
        if self.body_tick % self.body_steps == 0:
            self.session.advance(self.data.xpos[self.antennae, :2])
        obs = HybridControllerObservation.from_sim(sim, self.fly.name)
        apply_locomotion_action(sim, self.fly.name,
                               self.controller.step(self.session.drive, obs))
        super().step(dt)
        self.body_tick += 1
        if self.body_tick % self.body_steps == 0:
            if not np.isfinite(self.data.qpos).all():
                raise FloatingPointError('Non-finite Flywire body state.')
            self.trace.write(json.dumps(dict(
                episode=self.episode, time_s=float(self.time),
                inputs=self.session.levels.tolist(), readouts_hz=self.session.rates,
                drive=self.session.drive.tolist(), spikes=self.session.spikes,
                position_mm=self.data.xpos[self.thorax].tolist())) + '\n')
            self.trace.flush()

    def reset(self):
        super().reset()
        self.session.reset()
        self.controller.reset(seed=self.session.seed)
        self.body_tick = 0
        self.episode += 1
        self.trace.write(json.dumps({'event': 'reset', 'episode': self.episode}) + '\n')
        self.trace.flush()

    def apply_input(self, msg):
        if isinstance(msg, (Camera, Orbit)):
            super().apply_input(msg)
        elif isinstance(msg, Neutral):
            # Neutral is not the server's Reset message: SimThread does not
            # rebase its wall-clock pacing for it. Keep the body clock intact.
            elapsed = self.time
            self.reset()
            self.data.time = elapsed
        else:
            log.warning('Flywire owns actuators; manual actuator input ignored.')

    def apply_state(self, msg):
        raise ValueError('Flywire owns the body state; use pause/reset and camera controls.')

    def close(self):
        self.trace.close()
        self._scene.sim.close()


def build_streaming_backend(args):
    """Opt-in factory; no second HTTP server, viewer window or physics loop."""
    if args.backend != 'cpu':
        raise ValueError('Flywire body requires --backend cpu; select the brain with --neural-backend gpu.')
    if (args.musculoskeletal or args.vision or args.world != 'flat'
            or args.spawn is not None or args.actuator != 'position'
            or args.dof != 'legs-active' or args.mesh != 'fullsize' or args.adhesion):
        raise ValueError('Flywire uses its original flat-ground body, pose and adhesion; omit scene overrides.')
    session = EonSession(args.flywire_config, args.neural_backend,
                         args.flywire_seed, args.flywire_condition)
    sim, fly, controller, thorax, antennae = build_body(
        session.cfg['stimulus']['source_xy_mm'], tracking_camera=True)
    try:
        scene = FlywireScene(sim, fly)
        backend = FlywireBackend(scene, session, fly, controller, thorax,
                                 antennae, args.flywire_output)
        return scene, backend
    except BaseException:
        sim.close()
        raise
