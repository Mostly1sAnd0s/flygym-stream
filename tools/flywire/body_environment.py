from pathlib import Path
import time

import mujoco
import mujoco.viewer

from flygym.anatomy import (
    AxisOrder,
    ContactBodiesPreset,
    JointPreset,
    Skeleton,
)
from flygym.compose import (
    FlatGroundWorld,
    KinematicPosePreset,
    NeuroMechFly,
)
from flygym.utils.math import Rotation3D


def main():
    project_folder = Path(__file__).resolve().parent
    output_folder = project_folder / "Outputs" / "body_environment"
    output_folder.mkdir(parents=True, exist_ok=True)

    print("Creating fly body...")
    print("The first run may need to download the body meshes.")

    fly = NeuroMechFly()

    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL,
        axis_order=AxisOrder.ROLL_PITCH_YAW,
    )

    fly.add_joints(
        skeleton,
        neutral_pose=KinematicPosePreset.NEUTRAL,
    )
    fly.colorize()

    print("Creating ground and adding fly...")

    world = FlatGroundWorld()

    world.add_fly(
        fly,
        [0, 0, 0.7],
        Rotation3D(format="quat", values=[1, 0, 0, 0]),
        bodysegs_with_ground_contact=(
            ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD
        ),
    )

    print("Compiling physics model...")
    model, data = world.compile()
    print(f"Meshes: {model.nmesh}, bodies: {model.nbody}, joints: {model.njnt}")

    if model.nmesh == 0:
        raise RuntimeError("No meshes were included in the compiled model.")

    # Start in the saved neutral pose, if available.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print("Saving model and mesh assets...")
    world.save_xml_with_assets(str(output_folder))

    print(f"Saved to: {output_folder}")
    print("Opening viewer. Close its window to stop.")

    # Preserve the initial camera focus even if the body later moves.

    # Locate the thorax explicitly so the camera targets the fly.
    thorax_ids = [
        i for i in range(1, model.nbody)
        if (model.body(i).name or "").endswith("c_thorax")
    ]

    if not thorax_ids:
        raise RuntimeError("No fly thorax found in the compiled model.")

    thorax_id = thorax_ids[0]
    initial_center = data.xpos[thorax_id].copy()
    print("Fly thorax position:", initial_center)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = initial_center
            viewer.cam.distance = 8.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

        # Update the display around 60 times per second.
        # Physics still uses the model's smaller simulation timestep.
        frame_duration = 1.0 / 60.0

        while viewer.is_running():
            frame_start = time.perf_counter()
            simulation_target = data.time + frame_duration

            # Temporarily display the initial body pose without running physics.
            mujoco.mj_forward(model, data)

            viewer.sync()

            remaining = frame_duration - (
                time.perf_counter() - frame_start
            )
            if remaining > 0:
                time.sleep(remaining)

    print("Viewer closed.")


if __name__ == "__main__":
    main()