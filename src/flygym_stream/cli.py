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


def serve(args: argparse.Namespace) -> None:
    """Run the MJPEG-over-WebSocket streaming service (Phase 5 DoD).

    Builds the default scene, starts the sim thread, and serves until
    Ctrl-C; the sim thread and renderer are torn down on exit.
    """
    import uvicorn

    from flygym_stream.app import Producer, create_app
    from flygym_stream.config import Config
    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.scene import build_default_scene
    from flygym_stream.simulation import SimThread, make_backend
    from flygym_stream.stream import FrameQueue, InputQueue

    if args.bind == "0.0.0.0" and not args.insecure:
        sys.exit("refusing to bind 0.0.0.0 without --insecure (PLAN.md §6)")

    config = Config(bind=args.bind, port=args.port, width=args.width,
                    height=args.height, output_fps=args.fps,
                    backend=args.backend, playback_speed=args.speed,
                    overlay=args.overlay, transport=args.transport)
    scene = build_default_scene()
    backend = make_backend(scene, config)
    renderer = OffscreenRenderer(scene, config)
    frames, inputs = FrameQueue(), InputQueue()
    sim = SimThread(backend, renderer, frames, inputs, config)
    sim.start()
    app = create_app(config, Producer(frames, inputs, backend, sim))
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
