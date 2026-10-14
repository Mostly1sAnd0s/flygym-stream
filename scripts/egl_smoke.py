"""EGL headless-rendering smoke test (PLAN.md Phase 0, exploratory).

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run python scripts/egl_smoke.py [--flygym]

Steps:
1. Render one frame of a tiny inline MuJoCo model via ``mujoco.Renderer``
   (EGL), assert shape ``(240, 320, 3)`` and non-trivial content.
2. Report which EGL implementation the process actually loaded
   (NVIDIA vs mesa) via ``/proc/self/maps`` and ``eglinfo -B``.
3. With ``--flygym``, build the default flygym fly scene (triggers the
   one-time S3 mesh fetch), run 3 s of sim, and render a few frames.

Exits 0 on success; the exit code and printed notes feed the README.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import mujoco as mj
import numpy as np
from pathlib import Path

from PIL import Image

MINI_XML = """<mujoco>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.5">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.9 0.2 0.1 1"/>
    </body>
  </worldbody>
</mujoco>"""


def check_env() -> None:
    """Verify the required headless-rendering env vars are set."""
    import os

    for var in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        value = os.environ.get(var, "")
        if value.lower() != "egl":
            print(f"ERROR: {var}={value!r}, expected 'egl'. Re-run with:")
            print("  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ...")
            sys.exit(1)
    print("env OK: MUJOCO_GL=egl PYOPENGL_PLATFORM=egl")


def report_gl_backend() -> None:
    """Print which EGL libraries are loaded in this process."""
    with open("/proc/self/maps") as f:
        maps = f.read()
    for lib in ("libEGL_nvidia", "libEGL_mesa"):
        print(f"  {lib}: {'LOADED' if lib in maps else 'not loaded'}")
    try:
        out = subprocess.run(
            ["eglinfo", "-B"], capture_output=True, text=True, timeout=15
        )
        for line in out.stdout.splitlines():
            if any(k in line for k in ("Device0", "Device1", "GL_VENDOR", "GL_RENDERER")):
                print(f"  eglinfo: {line.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  eglinfo: not available")


def render_mini() -> None:
    """Render one frame of a tiny model and sanity-check it."""
    m = mj.MjModel.from_xml_string(MINI_XML)
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    r = mj.Renderer(m, 240, 320)
    try:
        r.update_scene(d)
        frame = r.render()
        assert frame.shape == (240, 320, 3), f"unexpected shape {frame.shape}"
        assert frame.dtype == np.uint8, f"unexpected dtype {frame.dtype}"
        assert float(frame.std()) > 1.0, "frame is trivially constant"
        print(
            f"mini render OK: shape={frame.shape} dtype={frame.dtype} "
            f"mean={float(frame.mean()):.1f} std={float(frame.std()):.1f}"
        )
    finally:
        r.close()


def render_flygym() -> None:
    """Build the default flygym scene, run 3 s of sim, render frames."""
    import flygym.rendering
    from flygym.anatomy import (
        ActuatedDOFPreset,
        AxisOrder,
        ContactBodiesPreset,
        JointPreset,
        Skeleton,
    )
    from flygym.compose import (
        ActuatorType,
        FlatGroundWorld,
        KinematicPosePreset,
        NeuroMechFly,
    )
    from flygym.utils.assets_lazy_loading import get_cache_root
    from flygym.utils.math import Rotation3D

    t0 = time.perf_counter()
    fly = NeuroMechFly()
    skeleton = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL, axis_order=AxisOrder.YAW_PITCH_ROLL)
    fly.add_joints(skeleton, KinematicPosePreset.NEUTRAL)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=KinematicPosePreset.NEUTRAL,
        kp=50.0,
        ctrlrange=(-3.14, 3.14),
    )
    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        (0, 0, 0.8),
        Rotation3D("quat", (1, 0, 0, 0)),  # identity, wxyz
        bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
    )
    mj_model, mj_data = world.compile()
    t1 = time.perf_counter()
    print(f"scene built in {t1 - t0:.1f} s (asset cache: {get_cache_root()})")

    dt = mj_model.opt.timestep
    n_steps = int(3.0 / dt)  # 3 s of sim
    with flygym.rendering.Renderer(
        mj_model,
        f"{fly.name}/trackingcam",  # camera names are namespaced by fly name
        camera_res=(240, 320),
        playback_speed=1.0,
        output_fps=30,
        buffer_frames=False,
        render_rgb=True,
    ) as renderer:
        t2 = time.perf_counter()
        n_frames = 0
        last = None
        for _ in range(n_steps):
            mj.mj_step(mj_model, mj_data)
            if renderer.render_as_needed(mj_data):
                last = renderer.mj_renderer.render()
                n_frames += 1
        t3 = time.perf_counter()
        wall = t3 - t2
    print(
        f"flygym render OK: {n_steps} sim steps in {wall:.2f} s wall "
        f"({n_steps / wall:.0f} steps/s, {n_frames} frames @30fps) "
        f"last frame shape={last.shape} std={float(last.std()):.1f}"
    )
    assert n_frames >= 1, "no frames rendered"
    assert float(last.std()) > 1.0, "frame is trivially constant"
    out = Path("scripts/egl_smoke_frame.png")
    Image.fromarray(last).save(out)
    print(f"sample frame saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flygym", action="store_true", help="also build+render the flygym fly scene"
    )
    args = parser.parse_args()

    check_env()
    render_mini()
    report_gl_backend()
    if args.flygym:
        render_flygym()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
