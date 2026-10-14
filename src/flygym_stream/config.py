"""Runtime configuration for the FlyGym streaming service (PLAN.md §3).

One ``Config`` dataclass is the single source of truth for all tunables —
no magic numbers scattered across modules. Values come from defaults,
CLI flags, or the environment at process start.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

#: Backends accepted for the simulation loop (PLAN.md §3).
BACKENDS = ("cpu", "gpu", "auto")
#: Scenes accepted by the CLI (PLAN.md §3).
SCENES = ("fly_flat_ground",)
#: Bind addresses accepted by the server. ``0.0.0.0`` requires the
#: explicit ``--insecure`` CLI flag (PLAN.md §6).
BINDS = ("127.0.0.1", "0.0.0.0")
#: Depth-overlay modes (PLAN.md Phase 6 stretch).
OVERLAYS = ("none", "depth")
#: Frame-stream transports (PLAN.md Phase 6): MJPEG binary WS frames or
#: WebRTC (aiortc 'webrtc' extra). The JSON control channel is unchanged.
TRANSPORTS = ("mjpeg", "webrtc")


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        output_fps: Target MJPEG frames per second.
        jpeg_quality: JPEG encode quality, 1-100.
        playback_speed: Sim-time multiplier relative to wall clock
            (1.0 = real time).
        bind: Interface the server binds.
        port: Port the server listens on.
        backend: Simulation backend: "cpu" (MuJoCo), "gpu" (MJWarp),
            or "auto" (pick the fastest available).
        scene: Scene identifier built by ``scene.build_default_scene``.
        overlay: "none" or "depth" (blue-cyan depth tint over the frame;
            PLAN.md Phase 6 stretch).
        transport: "mjpeg" (binary JPEG frames on the WS) or "webrtc"
            (PLAN.md Phase 6 stretch; needs the 'webrtc' extra).
    """

    width: int = 1280
    height: int = 720
    output_fps: int = 30
    jpeg_quality: int = 80
    playback_speed: float = 1.0
    bind: str = "127.0.0.1"
    port: int = 8765
    backend: str = "cpu"
    scene: str = "fly_flat_ground"
    overlay: str = "none"
    transport: str = "mjpeg"

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def camera_res(self) -> tuple[int, int]:
        """MuJoCo / flygym camera resolution: ``(height, width)``."""
        return (self.height, self.width)

    def replace(self, **changes: object) -> "Config":
        """Return a copy with the given fields changed (re-validated)."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in changes.items():
            if key not in values:
                raise TypeError(f"unknown Config field: {key!r}")
            values[key] = value
        return Config(**values)  # type: ignore[arg-type]


def _validate(cfg: Config) -> None:
    """Check every field range; raise ``ValueError`` on the first violation."""
    if not (1 <= cfg.width <= 16384):
        raise ValueError(f"width must be 1..16384, got {cfg.width}")
    if not (1 <= cfg.height <= 16384):
        raise ValueError(f"height must be 1..16384, got {cfg.height}")
    if not (1 <= cfg.output_fps <= 240):
        raise ValueError(f"output_fps must be 1..240, got {cfg.output_fps}")
    if not (1 <= cfg.jpeg_quality <= 100):
        raise ValueError(f"jpeg_quality must be 1..100, got {cfg.jpeg_quality}")
    if not (0 < cfg.playback_speed <= 100.0):
        raise ValueError(
            f"playback_speed must be > 0, got {cfg.playback_speed}"
        )
    if not (1 <= cfg.port <= 65535):
        raise ValueError(f"port must be 1..65535, got {cfg.port}")
    if cfg.bind not in BINDS:
        raise ValueError(f"bind must be one of {BINDS}, got {cfg.bind!r}")
    if cfg.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {cfg.backend!r}")
    if cfg.scene not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}, got {cfg.scene!r}")
    if cfg.overlay not in OVERLAYS:
        raise ValueError(f"overlay must be one of {OVERLAYS}, got {cfg.overlay!r}")
    if cfg.transport not in TRANSPORTS:
        raise ValueError(
            f"transport must be one of {TRANSPORTS}, got {cfg.transport!r}")
