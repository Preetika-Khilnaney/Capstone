"""
Central configuration for the Track 1 pipeline.
All constants are derived from context.md Sections 5-6.
"""

from pathlib import Path
from dataclasses import dataclass, field

# ── Project root (one level above app/) ──────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VideoConfig:
    """Timing and FPS constants for event clips."""
    pre_buffer_seconds: float = 10.0
    post_trigger_seconds: float = 6.0
    target_fps: int = 10
    jpeg_quality: int = 85

    @property
    def clip_duration(self) -> float:
        return self.pre_buffer_seconds + self.post_trigger_seconds

    @property
    def total_frames(self) -> int:
        return int(self.clip_duration * self.target_fps)


@dataclass(frozen=True)
class ThresholdConfig:
    """MOG2 entropy-based event trigger parameters (fallback path only)."""
    warmup_seconds: float = 120.0
    percentile: float = 95.0
    multiplier: float = 1.5
    absolute_floor: float = 0.05
    cooldown_seconds: float = 15.0


@dataclass(frozen=True)
class IncidentConfig:
    """Motion-native "burst-then-stop" incident anchor parameters.

    The primary trigger is a frame-difference motion state machine: a short,
    sharp motion burst followed by stillness (the crash/obstruction signature).
    The historical MOG2 entropy-ratio breach is retained only as a fallback for
    sources that never produce a burst-then-stop pattern (e.g. streams).
    All values are documented defaults to be validated/tuned per camera.
    """
    search_start_s: float = 2.0        # skip early adaptation zone (frame-diff inflated)
    baseline_window_s: float = 5.0     # rolling motion baseline window
    burst_factor: float = 1.6          # burst = motion > this × baseline
    burst_min_s: float = 0.5           # burst must persist this long
    burst_reject_s: float = 10.0       # still-high after this → not an incident, revert
    stop_window_s: float = 1.5         # stillness window required to confirm incident
    stop_factor: float = 0.65          # stillness = median motion < this × baseline
    terminate_factor: float = 0.7      # post-trigger end threshold
    settle_s: float = 2.5              # sustained low motion to end capture
    post_margin_s: float = 2.0         # extra tail after settle
    min_post_s: float = 6.0            # floor on post-trigger length
    max_post_s: float = 12.0           # conservative ceiling on post-trigger length
    max_search_s: float = 300.0        # give up motion searching past this source time
    motion_downscale: int = 4          # downsample frames by this factor for the motion score


@dataclass(frozen=True)
class SceneConfig:
    """Deterministic scene-semantics parameters (rider/pillion roles, stage rules)."""
    person_bike_overlap_frac: float = 0.35   # bbox overlap for rider/pillion role
    person_bike_min_frames: int = 5          # frames of overlap to assign a role
    rider_reaction_min_s: float = 0.5        # response-latency window bounds
    rider_reaction_max_s: float = 3.0
    stage_boundary_tol_s: float = 1.0        # boundary proximity tolerance (eval)
    skid_lateral_m: float = 3.0              # lateral deviation threshold for skid
    recovery_speed_thresh_mps: float = 2.0   # speeds above this = traffic resumed


@dataclass(frozen=True)
class YOLOConfig:
    """Detection model parameters."""
    model_name: str = "yolo11s.pt"
    confidence: float = 0.30
    # COCO class indices: car=2, motorcycle=3, bus=5, truck=7, bicycle=1, person=0
    class_whitelist: tuple[int, ...] = (0, 1, 2, 3, 5, 7)
    # Inference backend: "openvino" (INT8 IR, fast + best recall) or "pytorch".
    # OpenVINO falls back to the .pt weights if the exported IR dir is absent.
    backend: str = "openvino"
    openvino_model_dir: str = "yolo11s_int8_openvino_model"
    # Pin the inference device. Must be explicit: a CUDA-built torch with no usable
    # GPU otherwise mis-detects one and raises "Invalid device id".
    device: str = "cpu"


@dataclass(frozen=True)
class TrackerConfig:
    """BoT-SORT tracker parameters.

    Tuned to reduce duplicate/fragmented Object_IDs. match_thresh reverted from
    a non-standard 0.99 (caused ID switches) to the 0.8 default; new_track_thresh
    raised to suppress ghost tracks. ReID was verified active but gave no benefit
    on our footage, so it stays off. See handoff.py for the complementary
    min-lifespan track filter.
    """
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.5
    track_buffer: int = 60
    match_thresh: float = 0.8
    with_reid: bool = False


@dataclass(frozen=True)
class InterpolationConfig:
    """Occluded-track interpolation policy."""
    max_gap_frames: int = 10  # 1.0 second at 10 FPS
    min_track_frames: int = 8  # drop tracks observed in fewer frames (ghost/fragment filter)
    velocity_smooth_window: int = 11  # Savitzky-Golay window (odd) for world positions before differencing; 0/<5 disables
    max_speed_mps: float = 40.0       # physical speed cap (~144 km/h) — clamps residual BEV projection spikes


@dataclass(frozen=True)
class ProjectionConfig:
    """
    Reliable-region gate for the bird's-eye-view projection.

    Monocular homography extrapolates non-linearly outside the calibration quad
    (near the horizon, 1px of jitter → metres of error), so world positions far
    from the calibrated region are unreliable. Positions outside these bounds are
    set to NaN and excluded from kinematics/causal analysis. Bounds are generous
    around the quad (X∈[0,3.5], Y∈[0,14] for the current calibration).
    """
    min_depth_m: float = -100.0       # world Y (longitudinal) lower bound
    max_depth_m: float = 100.0        # world Y beyond this is extrapolation garbage
    max_abs_lateral_m: float = 25.0   # |world X| beyond this → NaN


@dataclass(frozen=True)
class CropConfig:
    """Entity crop selection parameters."""
    min_area_ratio: float = 0.4


@dataclass(frozen=True)
class RAGConfig:
    """RAG pipeline and LanceDB configuration."""
    model_name: str = "google/siglip-base-patch16-224"
    db_uri: str = "dataset/lancedb"
    table_name: str = "entity_crops"


@dataclass(frozen=True)
class CausalConfig:
    """Track 2 — Full multi-method causal engine parameters."""
    # PCMCI+ temporal lags
    tau_min: int = 1              # min lag in frames for PCMCI+
    tau_max: int = 10             # max lag in frames (1.0s at 10 FPS); adaptive clamped per series length
    tau_max_frame_frac: float = 0.12  # adaptive cap: min(tau_max, n_timesteps * this fraction)
    pc_alpha: float | None = None  # None = tigramite auto-selects via information criterion

    # Series quality filters
    min_series_len: int = 20      # min valid frames for an object to be included
    min_variable_presence_frac: float = 0.40  # variable must be finite in ≥ this fraction of frames

    # Pairwise interaction gating
    interaction_distance_m: float = 20.0  # max distance for a pair to be considered interacting
    lane_tolerance_m: float = 4.0         # lateral tolerance for "lead vehicle" (same-lane) detection

    # Speed/physics sanity checks
    max_plausible_speed_mps: float = 25.0  # reject entities whose peak speed exceeds this (~90 km/h)
    min_speed_drop_mps: float = 1.5        # minimum speed drop to flag as a braking event

    # Kinematic smoothing
    smooth_window: int = 5        # Savitzky-Golay window for smoothing speed/acceleration


@dataclass(frozen=True)
class SynthesisConfig:
    """Track 4 — LLM situation-report synthesis.

    Provider-agnostic: targets any OpenAI-compatible /chat/completions endpoint
    (hosted now, a local/edge server later — just change base_url). The API key is
    read from the environment variable named by api_key_env (never stored in code).
    """
    provider: str = "openai_compatible"
    # Defaults to Google Gemini's OpenAI-compatible endpoint (free tier via AI Studio).
    # Swap base_url/model for any other OpenAI-compatible provider or a local edge server.
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    model: str = "gemini-flash-latest"  # alias → current Gemini flash (avoids model-deprecation breakage)
    api_key_env: str = "LLM_API_KEY"
    max_tokens: int = 2500  # generous: "thinking" Gemini flash models spend tokens on internal reasoning
    temperature: float = 0.3
    max_entities: int = 10        # cap entities described in the evidence packet
    min_entity_frames: int = 8    # ignore fleeting tracks in the packet


@dataclass(frozen=True)
class FeedConfig:
    """Continuous live-monitoring + hybrid all-vehicle indexing."""
    index_all_vehicles: bool = True   # hybrid: index every tracked vehicle, not just event ones
    process_width: int = 640          # downscale incoming frames to this before processing (0 = native)
    process_height: int = 360         # keep at the homography calibration resolution; also bounds 4K memory
    track_end_frames: int = 15        # sampled frames a track may be unseen before it's finalized
    min_crop_area_px: int = 500       # ignore tiny far-field crops (noise)
    reconnect_delay_s: float = 2.0    # wait before reopening a dropped stream
    max_reconnect_attempts: int = 5   # give up on a stream after this many consecutive failures


@dataclass(frozen=True)
class PathConfig:
    """All filesystem paths."""
    base_dir: Path = BASE_DIR
    dataset_dir: Path = field(default_factory=lambda: BASE_DIR / "dataset")
    uploads_dir: Path = field(default_factory=lambda: BASE_DIR / "dataset" / "uploads")
    config_dir: Path = field(default_factory=lambda: BASE_DIR / "config")
    log_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")
    db_path: Path = field(default_factory=lambda: BASE_DIR / "event_registry.db")
    homography_path: Path = field(default_factory=lambda: BASE_DIR / "config" / "homography.npy")
    lancedb_path: Path = field(default_factory=lambda: BASE_DIR / "dataset" / "lancedb")


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    video: VideoConfig = field(default_factory=VideoConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    incident: IncidentConfig = field(default_factory=IncidentConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# ── Singleton config instance ────────────────────────────────────────────────
settings = PipelineConfig()
