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

    sub.add_parser("serve", help="run the streaming service (Phase 5)",
                   add_help=False).set_defaults(func=None)
    sub.add_parser("fetch-assets", help="prefetch flygym assets (Phase 5)",
                   add_help=False).set_defaults(func=None)

    args = parser.parse_args(argv)
    require_gl_env()
    if args.func is None:
        sys.exit(f"{args.cmd} is not implemented yet (Phase 5)")
    args.func(args)


if __name__ == "__main__":
    main()
