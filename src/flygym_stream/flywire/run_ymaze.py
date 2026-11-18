"""Y-maze experiment for FlyGym 2.1. Default: scripted body test, no connectome."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time
import numpy as np
from .ymaze_arena import YMaze, ChoiceTracker, scripted_drive


def build_body(maze, odor, seed):
    import mujoco
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld
    from flygym.anatomy import ContactBodiesPreset
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import (HybridTurningController, LocomotionAction,
        PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly)
    fly=make_locomotion_fly(name='ymaze_fly',add_adhesion=True,colorize=True)
    # FlyGym disables automatic collision masks by default. Enable body-to-wall
    # collisions only; keep its explicit foot-to-floor pairs unchanged.
    for geoms in fly.bodyseg_to_mjcfgeom.values():
        for geom in geoms:
            geom.contype=2; geom.conaffinity=0
    world=FlatGroundWorld()
    maze.add_to_world(world,odor)
    world.add_fly(fly,maze.start.tolist(),Rotation3D('quat',[1,0,0,0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,add_ground_contact_sensors=False)
    sim=Simulation(world)
    steps=PreprogrammedSteps(); dofs=fly.get_actuated_jointdofs_order('position')
    controller=HybridTurningController(timestep=sim.timestep,preprogrammed_steps=steps,output_dof_order=dofs)
    sim.reset(); controller.reset(seed=seed)
    apply_locomotion_action(sim,fly.name,LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dofs),adhesion_onoff=np.ones(6,dtype=bool)))
    sim.warmup(); mujoco.mj_forward(sim.mj_model,sim.mj_data)
    def body_id(suffix):
        matches=[i for i in range(1,sim.mj_model.nbody) if sim.mj_model.body(i).name.endswith('/'+suffix)]
        if len(matches)!=1: raise ValueError(f'Body {suffix}: found {matches}')
        return matches[0]
    return sim,fly,controller,body_id('c_thorax'),[body_id('l_funiculus'),body_id('r_funiculus')]


def plot_run(output, maze, rows, condition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,7))
    p=np.vstack([maze.polygon,maze.polygon[0]])
    ax.fill(p[:,0],p[:,1],color='#f1f4f8'); ax.plot(p[:,0],p[:,1],color='#334155',lw=3)
    for side,xy in maze.sources.items():
        ax.scatter(*xy,c='#22a855' if condition in (side,'both') else '#999999',s=80)
        ax.annotate(side,xy,xytext=(5,8),textcoords='offset points')
    if rows:
        trajectory=np.array(rows)
        ax.plot(trajectory[:,0],trajectory[:,1],color='#b54b1b',lw=1.4,label='Thorax trajectory')
        ax.scatter(*trajectory[0],marker='o',color='#b54b1b')
        ax.scatter(*trajectory[-1],marker='x',color='#b54b1b')
        ax.legend()
    ax.set(aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title=f'Y-maze: odor {condition}')
    fig.tight_layout(); fig.savefig(output/'trajectory.png',dpi=180); plt.close(fig)


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['scripted','banc'],default='scripted')
    p.add_argument('--arm',choices=['left','right'],default='left',help='Scripted target only; never used by BANC')
    p.add_argument('--odor',choices=['left','right','both','none'],default='left')
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--maze-config',default=str(here/'ymaze_config.json'))
    p.add_argument('--brain-config',default=str(here/'hybrid_config.json'))
    p.add_argument('--backend',choices=['cpu','gpu'],default=None,
        help='Neural backend (default: config setting, otherwise gpu); BANC only')
    p.add_argument('--neural-dt-ms',type=float,default=0.1,help='BANC only; 0.5 is faster but changes numerical approximation')
    p.add_argument('--motor-dt-ms',type=float,default=1.0,help='BANC motor readout interval; commands held between readouts (ms)')
    p.add_argument('--no-viewer',action='store_true')
    p.add_argument('--fast',action='store_true',help='Do not pace to real time')
    p.add_argument('--preview',action='store_true',help='Save top-down maze image without loading FlyGym')
    p.add_argument('--output',default=None)
    args=p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0: p.error('--seconds must be positive')
    if not 0<args.neural_dt_ms<=1: p.error('--neural-dt-ms must be in (0,1]')
    if not np.isfinite(args.motor_dt_ms) or args.motor_dt_ms<=0: p.error('--motor-dt-ms must be positive and finite')
    maze_path=Path(args.maze_config).resolve()
    cfg=json.loads(maze_path.read_text(encoding='utf-8-sig')); maze=YMaze(cfg)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=Path(args.output).resolve() if args.output else here/'Outputs'/f'ymaze_{stamp}'
    output.mkdir(parents=True,exist_ok=True)
    print(f'Output: {output}',flush=True)
    if args.preview:
        plot_run(output,maze,[],args.odor); print(output/'trajectory.png'); return
    brain=None; report=None; brain_cfg=None
    if args.mode=='banc':
        from .load_network import load_network
        from .neural_model import MotorDecoder
        bp=Path(args.brain_config).resolve(); brain_cfg=json.loads(bp.read_text(encoding='utf-8-sig'))
        args.backend=args.backend or brain_cfg['neural'].get('backend','gpu')
        brain_cfg['neural']['backend']=args.backend
        brain,populations,report=load_network(brain_cfg,bp.parent)
        decoder=MotorDecoder(brain,populations,brain_cfg['decoder'])
        inputs=[brain.xp.asarray(brain.indices(populations[k])) for k in ('input_left','input_right')]
        current=brain.xp.zeros(len(brain.ids))
        print(f'Neural backend: {brain.backend}; precision: float64',flush=True)
        print('BANC: odor drives selected inputs; no scripted steering or collision sensory input.',flush=True)
    else:
        print(f'SCRIPTED BODY TEST: target={args.arm}; brain NOT loaded; odor does not steer this mode.',flush=True)
    print('Building maze and body...',flush=True)
    sim,fly,controller,thorax,antennae=build_body(maze,args.odor,args.seed)
    from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action
    dt=sim.timestep
    stride=max(1,round(args.neural_dt_ms/1000/dt))
    motor_stride=max(1,round(args.motor_dt_ms/1000/(stride*dt)))
    if brain is not None and not np.isclose(stride*dt,args.neural_dt_ms/1000,rtol=0,atol=1e-10):
        sim.close(); raise ValueError('Neural dt must be an integer multiple of physics dt.')
    metadata=dict(arguments=vars(args),maze=cfg,brain_config=brain_cfg,network_report=report,
        physics_dt_s=dt,neural_dt_s=stride*dt if brain is not None else None,
        motor_interval_s=motor_stride*stride*dt if brain is not None else None,
        limitations=['Gaussian odor proxy has no wall occlusion or airflow.',
        'Hybrid leg controller, not direct biological VNC-to-muscle control.',
        'No wall-touch signal is injected into the brain. Scripted mode uses privileged position feedback.'])
    (output/'trial_config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    context=nullcontext(None)
    if not args.no_viewer:
        import mujoco.viewer
        context=mujoco.viewer.launch_passive(sim.mj_model,sim.mj_data)
    tracker=ChoiceTracker(cfg['choice_dwell_s'])
    wall_ids=np.array([i for i in range(sim.mj_model.ngeom) if sim.mj_model.geom(i).name.startswith('ymaze_wall_')])
    wall_contact_samples=0
    path=[]; status='timeout'; t=0.0; samples=0; neural_steps=0; pending=0
    rates=np.zeros(2); drive=np.zeros(2); input_integral=np.zeros(2)
    neural_wall=body_wall=0.0; elapsed=0.0
    log_stride=max(1,round(.02/dt)); frame_stride=max(1,round(1/(30*dt)))
    nsteps=int(np.ceil(args.seconds/dt))
    start=time.perf_counter()
    try:
        with context as viewer,(output/'trace.csv').open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['time_s','x_mm','y_mm','z_mm','heading_rad','odor_L','odor_R',
                'drive_L','drive_R','output_channel_L_hz','output_channel_R_hz','zone','choice','wall_contacts'])
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:]=[1,0,0]; viewer.cam.distance=62
                    viewer.cam.azimuth=90; viewer.cam.elevation=-75
            for step in range(nsteps):
                if viewer and not viewer.is_running(): status='viewer_closed'; break
                pos=sim.mj_data.xpos[thorax].copy()
                antenna_xy=sim.mj_data.xpos[antennae,:2]
                forward=antenna_xy.mean(axis=0)-pos[:2]
                heading=np.arctan2(forward[1],forward[0])
                odor=maze.odor(antenna_xy,args.odor)
                if brain is None:
                    drive=scripted_drive(maze,pos[:2],heading,args.arm)
                else:
                    # Accumulate physical sensor samples over a completed neural interval.
                    # Outputs are held until the next neural update (causal sample/hold).
                    input_integral+=odor*dt; pending+=1
                tic=time.perf_counter()
                obs=HybridControllerObservation.from_sim(sim,fly.name)
                apply_locomotion_action(sim,fly.name,controller.step(drive,obs)); sim.step()
                body_wall+=time.perf_counter()-tic
                t=(step+1)*dt
                if brain is not None and (pending==stride or step==nsteps-1):
                    tic=time.perf_counter(); interval=pending*dt
                    levels=input_integral/interval
                    current.fill(0)
                    for indices,level in zip(inputs,levels):
                        current[indices]+=brain_cfg['stimulus']['baseline_pA']+brain_cfg['stimulus']['amplitude_pA']*level
                    brain.step(current,interval)
                    if neural_steps%motor_stride==0:
                        drive,rates=decoder.read(brain.rates_hz)
                    neural_steps+=1; input_integral.fill(0); pending=0
                    neural_wall+=time.perf_counter()-tic
                pos=sim.mj_data.xpos[thorax].copy()
                if not np.isfinite(sim.mj_data.qpos).all():
                    status='nonfinite_physics'; break
                zone=maze.zone(pos[:2]); tracker.update(zone,t)
                if pos[2]>cfg['wall_height_mm']+1 or not maze.contains(pos[:2]): status='escaped'
                elif tracker.choice: status='choice'
                if step%log_stride==0 or status!='timeout' or step==nsteps-1:
                    if brain is not None: brain.check_health()
                    contacts=sim.mj_data.contact
                    wall_contacts=int(np.count_nonzero(np.isin(contacts.geom1,wall_ids) | np.isin(contacts.geom2,wall_ids)))
                    wall_contact_samples+=int(wall_contacts>0)
                    writer.writerow([t,*pos,heading,*odor,*drive,*rates,zone or '',tracker.choice or '',wall_contacts])
                    path.append(pos[:2].tolist()); samples+=1
                if step%max(1,round(1/dt))==0:
                    print(f'{t:.2f}s / {args.seconds:g}s | xy={pos[:2].round(1)} | drive={drive.round(2)} | zone={zone}',flush=True)
                if viewer and step%frame_stride==0:
                    viewer.sync()
                    if not args.fast:
                        delay=t-(time.perf_counter()-start)
                        if delay>0: time.sleep(min(delay,1/30))
                if status!='timeout': break
    finally:
        elapsed=time.perf_counter()-start
        if status=='timeout' and t < args.seconds-dt: status='interrupted'
        result=dict(status=status,choice=tracker.choice,decision_time_s=tracker.decision_time_s,
            simulated_s=t,wall_s=elapsed,simulated_seconds_per_wall_second=t/max(elapsed,1e-9),
            neural_compute_s=neural_wall,body_compute_s=body_wall,neural_updates=neural_steps,
            mode=args.mode,scripted_target=args.arm if args.mode=='scripted' else None,
            odor_condition=args.odor,seed=args.seed,logged_samples=samples,wall_contact_samples=wall_contact_samples)
        (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        sim.close(); plot_run(output,maze,path,args.odor)
    print(json.dumps(result,indent=2)); print(f'Results: {output}')


if __name__=='__main__':
    main()
