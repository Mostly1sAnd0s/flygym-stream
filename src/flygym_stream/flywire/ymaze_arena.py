"""Procedural Y-maze geometry, abstract odor field, and choice detection.

Coordinates and dimensions are mm, matching the FlyGym 2.1 model.
The stem points along +x; anatomical left is +y at the start.
No CAD files or additional mesh assets are needed for the arena.
"""
import numpy as np


class YMaze:
    def __init__(self, config):
        self.cfg = config
        for k, v in config.items():
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                raise ValueError(f'{k} must be a finite number.')
        for k in ('stem_length_mm','arm_length_mm','corridor_width_mm','wall_height_mm',
                  'wall_thickness_mm','odor_sigma_mm','choice_dwell_s','waypoint_lookahead_mm'):
            if config[k] <= 0:
                raise ValueError(f'{k} must be positive.')
        if not 20 <= config['branch_angle_deg'] <= 70:
            raise ValueError('Use branch_angle_deg in [20,70].')
        if not 0 < config['choice_start_fraction'] < config['source_fraction'] < 1:
            raise ValueError('Require 0 < choice_start_fraction < source_fraction < 1.')
        if not 0 < config['start_margin_mm'] < config['stem_length_mm']:
            raise ValueError('Start margin must lie inside the stem.')
        self.h = config['corridor_width_mm']/2
        self.a = np.deg2rad(config['branch_angle_deg'])
        self.c, self.s = np.cos(self.a), np.sin(self.a)
        self.length = config['arm_length_mm']
        if self.h/self.s >= self.length*config['choice_start_fraction']:
            raise ValueError('Arms too short/wide for nonoverlapping choice zones.')
        if config['wall_thickness_mm'] >= self.h:
            raise ValueError('Wall thickness too large for corridor.')
        self.directions = {'left': np.array([self.c,self.s]), 'right': np.array([self.c,-self.s])}
        h, c, s, l = self.h,self.c,self.s,self.length
        t = h*np.tan(self.a/2)
        stem = config['stem_length_mm']
        # Boundary of the union of three equal-width corridors (CCW).
        self.polygon = np.array([[-stem,-h],[-t,-h],[l*c-h*s,-l*s-h*c],
            [l*c+h*s,-l*s+h*c],[h/s,0],[l*c+h*s,l*s-h*c],
            [l*c-h*s,l*s+h*c],[-t,h],[-stem,h]])
        self.start = np.array([-stem+config['start_margin_mm'],0.0,0.8])
        self.sources = {k:v*l*config['source_fraction'] for k,v in self.directions.items()}

    def contains(self, xy):
        """Point in polygon (thorax center, not complete body containment)."""
        x,y = xy
        inside = False
        for p,q in zip(self.polygon, np.roll(self.polygon,-1,axis=0)):
            if (p[1] > y) != (q[1] > y):
                if x < (q[0]-p[0])*(y-p[1])/(q[1]-p[1])+p[0]:
                    inside = not inside
        return inside

    def zone(self, xy):
        xy = np.asarray(xy)
        for side,d in self.directions.items():
            along = float(xy@d)
            across = abs(float(xy@np.array([-d[1],d[0]])))
            if self.length*self.cfg['choice_start_fraction'] <= along <= self.length and across < self.h-self.cfg['wall_thickness_mm']/2:
                return side
        return None

    def odor(self, xy, condition):
        """Gaussian proxy, no airflow/transport/wall occlusion model. Values 0..1."""
        xy = np.asarray(xy)
        sides = {'left':['left'],'right':['right'],'both':['left','right'],'none':[]}[condition]
        total = np.zeros(xy.shape[:-1])
        for side in sides:
            d2 = np.sum((xy-self.sources[side])**2,axis=-1)
            total += np.exp(-d2/(2*self.cfg['odor_sigma_mm']**2))
        return np.clip(total,0,1)

    def add_to_world(self, world, odor_side):
        import mujoco
        root = world.mjcf_root.worldbody
        h = self.cfg['wall_height_mm']
        thickness = self.cfg['wall_thickness_mm']
        for i,(p,q) in enumerate(zip(self.polygon,np.roll(self.polygon,-1,axis=0))):
            vector=q-p; midpoint=(p+q)/2; angle=np.arctan2(vector[1],vector[0])
            root.add_geom(name=f'ymaze_wall_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[*midpoint,h/2], size=[np.linalg.norm(vector)/2+thickness/2,thickness/2,h/2],
                quat=[np.cos(angle/2),0,0,np.sin(angle/2)],rgba=[0.35,0.42,0.50,0.55],
                friction=[0.5,0.005,0.0001],contype=0,conaffinity=2)
        # Identical physical markers on both arms; odor condition changes color only.
        for side,xy in self.sources.items():
            active=odor_side in (side,'both')
            root.add_geom(name=f'{side}_source',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[*xy,0.015],size=[0.8,0.015,0],rgba=([0.2,0.8,0.3,1] if active else [0.6,0.6,0.6,1]),
                contype=0,conaffinity=0)
        for side,d in self.directions.items():
            begin=self.length*self.cfg['choice_start_fraction']; end=self.length
            angle=np.arctan2(d[1],d[0]); xy=d*(begin+end)/2
            root.add_geom(name=f'{side}_choice_zone',type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[*xy,0.005],size=[(end-begin)/2,self.h-thickness,0.005],
                quat=[np.cos(angle/2),0,0,np.sin(angle/2)],rgba=[0.3,0.6,0.9,0.18],contype=0,conaffinity=0)


class ChoiceTracker:
    def __init__(self, dwell_s):
        self.dwell_s=dwell_s; self.candidate=None; self.entered=None
        self.choice=None; self.decision_time_s=None

    def update(self, zone, t):
        if self.choice is not None:
            return self.choice
        if zone != self.candidate:
            self.candidate=zone; self.entered=t if zone is not None else None
        if zone is not None and t-self.entered >= self.dwell_s:
            self.choice=zone; self.decision_time_s=float(t)
        return self.choice


def scripted_drive(maze, xy, heading, side):
    """Position-guided engineering test: never interpreted as brain behavior."""
    xy=np.asarray(xy); look=maze.cfg['waypoint_lookahead_mm']
    if xy[0] < -2:
        target=np.array([min(xy[0]+look,0),0.0])
    else:
        d=maze.directions[side]
        target=d*np.clip(xy@d+look,look,maze.length*0.85)
    diff=target-xy
    error=np.arctan2(np.sin(np.arctan2(diff[1],diff[0])-heading),np.cos(np.arctan2(diff[1],diff[0])-heading))
    turn=np.clip(maze.cfg['steering_gain']*error,-0.75,0.75)*maze.cfg['steering_sign']
    speed=maze.cfg['scripted_speed']
    return np.clip([speed-turn,speed+turn],0.05,1.5)
