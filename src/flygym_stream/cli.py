"""Command-line interface (PLAN.md Phase 4-5).

Commands:
    render-preview  headless MP4 preview of the fly (Phase 4 DoD)
    serve           run the MJPEG-over-WebSocket streaming service (Phase 5)
    fetch-assets    force the one-time flygym S3 asset prefetch (Phase 5)

All commands refuse to start without MUJOCO_GL=egl /
PYOPENGL_PLATFORM=egl (PLAN.md §6) — mujoco reads them at import time, so
they must be exported before this process starts.
"""

from __future__ import annotations

import argparse
import os
import sys

REQUIRED_GL_VARS = ("MUJOCO_GL", "PYOPENGL_PLATFORM")


def require_gl_env() -> None:
    """Exit with a helpful message unless the headless GL env is set."""
    missing = [v for v in REQUIRED_GL_VARS if os.environ.get(v) != "egl"]
    if missing:
        sys.exit(
            f"refusing to start: {', '.join(missing)} must be 'egl' (headless "
            f"GPU rendering; PLAN.md §6). Export first: "
            + " ".join(f"{v}=egl" for v in REQUIRED_GL_VARS)
        )


def render_preview(args: argparse.Namespace) -> None:
    """Simulate + render ``--duration`` sim-seconds of the fly to an MP4.

    Unpaced (as fast as the hardware allows); frame count follows the
    sim-time pacing at ``--fps`` (PLAN.md Phase 4 DoD).
    """
    import imageio.v3 as iio

    from flygym_stream.config import Config
    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.scene import build_default_scene

    scene = build_default_scene()
    config = Config(width=args.width, height=args.height, output_fps=args.fps)
    frames = []
    with OffscreenRenderer(scene, config) as renderer:
        n_steps = int(args.duration / scene.mj_model.opt.timestep)
        for _ in range(n_steps):
            scene.step()
            if renderer.render_as_needed():
                frames.append(renderer.render_frame())
    iio.imwrite(args.out, frames, fps=config.output_fps, codec="libx264", quality=8)
    print(f"wrote {len(frames)} frames ({args.duration} sim s) -> {args.out}")


def scene_config_from_args(args: argparse.Namespace) -> object:
    """Map serve flags to a SceneConfig (PLAN.md Phase 7, M7.1).

    Args: parsed serve namespace (dof/mesh/actuator/world/spawn/vision/
        adhesion/musculoskeletal). Returns a ``flygym_stream.scenes.
        SceneConfig`` (imported lazily with the other scene-layer imports).
    """
    from flygym.anatomy import ActuatedDOFPreset
    from flygym.compose import ActuatorType, MeshType
    from flygym_stream.scenes import SceneConfig

    dof = {"all": ActuatedDOFPreset.ALL,
           "legs": ActuatedDOFPreset.LEGS_ONLY,
           "legs-active": ActuatedDOFPreset.LEGS_ACTIVE_ONLY}[args.dof]
    mesh = {"fullsize": MeshType.FULLSIZE,
            "simplified": MeshType.SIMPLIFIED_MAX2000FACES}[args.mesh]
    try:
        spawn = tuple(float(v) for v in args.spawn.split(","))
    except ValueError:
        sys.exit(f"--spawn expects x,y,z numbers, got {args.spawn!r}")
    if len(spawn) != 3:
        sys.exit(f"--spawn expects x,y,z, got {len(spawn)} values")
    return SceneConfig(actuated_preset=dof, mesh_type=mesh,
                       actuator_type=ActuatorType[args.actuator.upper()],
                       world=args.world, spawn=spawn, vision=args.vision,
                       adhesion=args.adhesion,
                       musculoskeletal=args.musculoskeletal)


def serve(args: argparse.Namespace) -> None:
    """Run the MJPEG-over-WebSocket streaming service (Phase 5 DoD).

    Builds the configured scene (Phase 7: --dof/--world/...), starts the
    sim thread (with the built-in locomotion engine, M7.3), and serves
    until Ctrl-C; the sim thread and renderer are torn down on exit.
    """
    import uvicorn

    from flygym_stream.app import Producer, create_app
    from flygym_stream.config import Config
    from flygym_stream.locomotion import LocController
    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.simulation import SimThread, make_backend
    from flygym_stream.stream import FrameQueue, InputQueue
    from flygym_stream.telemetry import (
        TelemetryGatherer, TelemetryQueue, TelemetrySink, gather_model_info,
    )

    if args.bind == "0.0.0.0" and not args.insecure:
        sys.exit("refusing to bind 0.0.0.0 without --insecure (PLAN.md §6)")

    config = Config(bind=args.bind, port=args.port, width=args.width,
                    height=args.height, output_fps=args.fps,
                    backend=args.backend, playback_speed=args.speed,
                    overlay=args.overlay, transport=args.transport,
                    telemetry_hz=args.telemetry_hz)
    from flygym_stream.scenes import build_scene
    scene = build_scene(scene_config_from_args(args))
    backend = make_backend(scene, config)
    renderer = OffscreenRenderer(scene, config)
    frames, inputs = FrameQueue(), InputQueue()
    loco = LocController(scene.mj_model, scene.neutral_actuators(),
                         fly_name=scene.fly_name)
    model_info = gather_model_info(scene.mj_model, fly_name=scene.fly_name,
                                   world=args.world)
    telemetry = TelemetrySink(
        gatherer=TelemetryGatherer(scene.mj_model, scene.mj_data,
                                   model_info=model_info),
        queue=TelemetryQueue(), hz=config.telemetry_hz)
    sim = SimThread(backend, renderer, frames, inputs, config, loco=loco,
                    telemetry=telemetry)
    sim.start()
    app = create_app(config, Producer(frames, inputs, backend, sim,
                                      telemetry_queue=telemetry.queue,
                                      n_dof=scene.mj_model.nq,
                                      model_info=model_info))
    print(f"flygym-stream serving on http://{config.bind}:{config.port}")
    try:
        uvicorn.run(app, host=config.bind, port=config.port, log_level="warning")
    finally:
        sim.stop()
        renderer.close()


def fetch_assets(args: argparse.Namespace) -> None:
    """Force the one-time flygym S3 asset prefetch (Phase 5)."""
    from flygym.utils.assets_lazy_loading import get_cache_root, prefetch_meshes

    dirs = prefetch_meshes()
    print(f"assets cached under: {get_cache_root()}")
    for d in dirs:
        print(f"  {d}")


def main(argv: list[str] | None = None) -> None:
    """Parse argv and dispatch the subcommand (serve lands in Phase 5)."""
    parser = argparse.ArgumentParser(
        prog="flygym-stream", description="FlyGym remote streaming service"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("render-preview", help="render a headless MP4 of the fly")
    rp.add_argument("--duration", type=float, default=3.0, help="sim seconds")
    rp.add_argument("--out", default="preview.mp4", help="output MP4 path")
    rp.add_argument("--width", type=int, default=1280)
    rp.add_argument("--height", type=int, default=720)
    rp.add_argument("--fps", type=int, default=30)
    rp.set_defaults(func=render_preview)

    sv = sub.add_parser("serve", help="run the streaming service (Phase 5)")
    sv.add_argument("--bind", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--width", type=int, default=1280)
    sv.add_argument("--height", type=int, default=720)
    sv.add_argument("--fps", type=int, default=30)
    sv.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "auto"])
    sv.add_argument("--speed", type=float, default=1.0, help="playback speed")
    sv.add_argument("--overlay", default="none", choices=["none", "depth"],
                    help="depth overlay on the streamed frames (Phase 6)")
    sv.add_argument("--transport", default="mjpeg",
                    choices=["mjpeg", "webrtc"],
                    help="frame transport; webrtc needs the 'webrtc' extra "
                         "and ?webrtc in the browser URL (Phase 6)")
    # Scene variants (PLAN.md Phase 7, M7.1)
    sv.add_argument("--dof", default="legs-active",
                    choices=["all", "legs", "legs-active"],
                    help="actuated DOF preset: all = 126 actuators "
                         "(incl. wings), default = 42 active leg DOFs")
    sv.add_argument("--mesh", default="fullsize",
                    choices=["fullsize", "simplified"],
                    help="mesh fidelity (simplified = <=2000 faces/geom)")
    sv.add_argument("--actuator", default="position",
                    choices=["motor", "position", "velocity", "intvelocity",
                             "damper", "cylinder", "muscle", "adhesion",
                             "tendon"],
                    help="actuator type for the actuated DOFs")
    sv.add_argument("--world", default="flat",
                    choices=["flat", "blocks", "gapped", "mixed", "tethered"],
                    help="world: terrain variants, or tethered = body pinned "
                         "(motor-control experiments)")
    sv.add_argument("--spawn", default="0,0,0.8",
                    help="root spawn position x,y,z (model units, mm)")
    sv.add_argument("--vision", action="store_true",
                    help="add the fly's eye cameras (compound-eye sensors)")
    sv.add_argument("--adhesion", action="store_true",
                    help="add leg adhesion actuators (6 extra)")
    sv.add_argument("--musculoskeletal", action="store_true",
                    help="muscle/tendon model (M7.6; rejected until then)")
    sv.add_argument("--telemetry-hz", type=float, default=10.0,
                    help="telemetry message rate in Hz (0 = off; M7.4)")
    sv.add_argument("--insecure", action="store_true",
                    help="allow binding 0.0.0.0 (PLAN.md §6)")
    sv.set_defaults(func=serve)

    fa = sub.add_parser("fetch-assets", help="prefetch flygym assets (Phase 5)")
    fa.set_defaults(func=fetch_assets)

    args = parser.parse_args(argv)
    require_gl_env()
    if args.func is None:
        sys.exit(f"{args.cmd} is not implemented yet")
    args.func(args)


if __name__ == "__main__":
    main()
