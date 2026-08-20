"""
Phase 0 — Heuristic Ingestion.

Scans a frame source (a video file *or* a live stream) for incident events.

The primary trigger is a motion-native "burst-then-stop" state machine: a short
sharp motion burst (a crash / sudden obstruction) followed by stillness. A short
clip is captured around the confirmed incident with a dynamic, motion-terminated
tail. The historical MOG2 entropy-ratio breach is retained only as a *fallback*
for sources that never produce a burst-then-stop signature (e.g. long flowing
traffic or streams).

The core trigger logic lives in :class:`FrameEventDetector`, a push-driven
state machine fed one frame at a time. Both the batch scanner
(:func:`scan_for_events`) and the live feed worker drive the same detector,
so file and stream ingestion share identical event semantics.

Reference: context.md §4 Phase 0, §5.2, §6.1
"""

import logging
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EventFrameBlock:
    """Container for a triggered event's frames and metadata."""
    trigger_time_sec: float
    source_fps: float
    pre_frames: list[bytes] = field(default_factory=list)   # JPEG-encoded
    post_frames: list[bytes] = field(default_factory=list)  # JPEG-encoded

    @property
    def all_frames(self) -> list[bytes]:
        return self.pre_frames + self.post_frames

    @property
    def duration_sec(self) -> float:
        total = len(self.all_frames)
        return total / self.source_fps if self.source_fps > 0 else 0.0


def _encode_frame(frame: np.ndarray) -> bytes:
    """JPEG-encode a frame to reduce memory footprint."""
    ok, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.video.jpeg_quality]
    )
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return buf.tobytes()


def _foreground_ratio(mask: np.ndarray) -> float:
    """Fraction of foreground pixels in a MOG2 mask."""
    return float(np.count_nonzero(mask)) / mask.size


class FrameEventDetector:
    """
    Stateful, push-driven incident-event detector.

    Feed frames one at a time via :meth:`process_frame`; it returns an
    :class:`EventFrameBlock` on the frame that completes an event, else None.
    It is agnostic to the frame source, so it works identically for a finite
    video file and an unbounded live stream.

    Primary path — motion "burst-then-stop" machine:
        SCANNING ─(motion > burst_factor × baseline for burst_min_s)─▶ BURST
           ▲                                                            │
           │                        (motion stays high for               ▼
           │                         burst_reject_s → revert)          CONFIRM?
           │                                                            │
           │                     (stop_window_s of low motion)          ▼
           └──────────────────────────────────────────────────–  INCIDENT CONFIRMED
                                                                trigger = burst frame
        CAPTURING ── post frames until motion < terminate_factor × baseline for
                      settle_s, then + post_margin_s; clamp [min_post_s, max_post_s];
                      force-emit at the cap.

    Baseline is a rolling median of the frame-difference motion over the last
    ``baseline_window_s`` (excluding the first ``search_start_s`` adaptation zone).
    At capture time the baseline is pinned to its burst-frame value, so the
    post-trigger termination compares against pre-incident flow level.

    Fallback path — MOG2 entropy ratio breach, identical to the historical
    behaviour. It fires only while the motion machine is idle (no active burst
    candidate), and is the sole trigger for sources that never burst-then-stop.
    """

    def __init__(self, source_fps: float, total_frames: int | None = None):
        cfg_v = settings.video
        cfg_i = settings.incident
        cfg_t = settings.threshold

        self.source_fps = source_fps if source_fps and source_fps > 0 else 30.0
        self.pre_buffer_size = int(cfg_v.pre_buffer_seconds * self.source_fps)
        self.post_frame_count = int(cfg_v.post_trigger_seconds * self.source_fps)
        self.warmup_frame_count = int(cfg_t.warmup_seconds * self.source_fps)
        self.cooldown_frame_count = int(cfg_t.cooldown_seconds * self.source_fps)

        # Auto-scale warmup for short videos: cap to 30% of total frames so
        # there is always room left to scan for events after warmup.
        if total_frames is not None and self.warmup_frame_count >= total_frames:
            min_warmup = max(int(self.source_fps), 30)  # at least ~1 second
            scaled = max(int(total_frames * 0.3), min_warmup)
            logger.info(
                "Auto-scaled warmup: %d -> %d frames (video has %d frames)",
                self.warmup_frame_count, scaled, total_frames,
            )
            self.warmup_frame_count = scaled

        self._cfg_t = cfg_t
        self._cfg_i = cfg_i

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.buffer: deque[bytes] = deque(maxlen=self.pre_buffer_size)
        self.warmup_ratios: list[float] = []
        self.threshold: float | None = None
        self.frame_idx = 0
        self.cooldown_remaining = 0

        # Post-trigger capture state
        self._capturing = False
        self._pre_frames: list[bytes] | None = None
        self._post_frames: list[bytes] | None = None
        self._trigger_time = 0.0
        # Legacy (entropy) capture uses a fixed frame count; motion capture uses
        # a dynamic, motion-terminated tail keyed off _capture_start_idx.
        self._capture_start_idx: int | None = None
        self._settle_frame: int | None = None
        self._capture_baseline: float = 0.0

        # Motion "burst-then-stop" machine state
        self._state = "SCANNING"
        # Downsampled gray frames of the last two positions: diffing against the
        # frame 2 back strides over pulldown-duplicated frames, which otherwise
        # alternate near-zero diffs and reset the burst candidate every frame.
        self._recent_small: deque[np.ndarray] = deque(maxlen=2)
        self._motion_hist: deque[tuple[int, float]] = deque(
            maxlen=int((cfg_i.baseline_window_s + cfg_i.search_start_s + 1)
                       * self.source_fps)
        )
        self._candidate_start: int | None = None
        self._candidate_pre: list[bytes] | None = None
        self._candidate_baseline: float | None = None
        self._stop_med: deque[float] = deque(
            maxlen=int(cfg_i.stop_window_s * self.source_fps) + 2
        )
        run_max = int((cfg_i.burst_reject_s + cfg_i.stop_window_s
                       + cfg_i.settle_s + cfg_i.max_post_s + 2) * self.source_fps)
        self._run_frames: deque[bytes] = deque(maxlen=run_max)
        self._burst_start_idx: int | None = None
        self._burst_frame: int | None = None
        self._burst_baseline: float | None = None

    @property
    def warmed_up(self) -> bool:
        return self.threshold is not None

    # ── motion score ────────────────────────────────────────────────────────
    def _motion_update(self, frame: np.ndarray) -> float:
        """
        Mean (downsampled) frame-difference magnitude, diffed 2 frames back;
        0.5 ms/frame.
        """
        d = self._cfg_i.motion_downscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = gray[::d, ::d]
        if len(self._recent_small) < 2:
            self._recent_small.append(small)
            return 0.0
        prev2 = self._recent_small[0]
        self._recent_small.append(small)
        diff = np.abs(small.astype(np.float32) - prev2.astype(np.float32))
        motion = float(diff.mean())
        self._motion_hist.append((self.frame_idx, motion))
        return motion

    def _baseline(self) -> float | None:
        """
        Rolling median of motion over the last baseline_window_s, post-adaptation,
        with a search_start_s guard band so a nascent burst doesn't pollute the
        threshold it must exceed.
        """
        cfg_i = self._cfg_i
        start = int(cfg_i.search_start_s * self.source_fps)
        vals = [m for f, m in self._motion_hist if f >= start]
        win = int(cfg_i.baseline_window_s * self.source_fps)
        guard = int(cfg_i.search_start_s * self.source_fps)
        vals = vals[-win - guard:-guard] if len(vals) > guard else vals[-win:]
        if len(vals) < max(8, win // 4):
            return None
        return float(np.median(vals))

    # ── motion "burst-then-stop" machine ────────────────────────────────────
    def _reset_candidate(self) -> None:
        self._candidate_start = None
        self._candidate_pre = None
        self._candidate_baseline = None
        self._run_frames.clear()

    def _enter_burst(self) -> None:
        self._state = "BURST"
        self._burst_start_idx = self.frame_idx
        self._burst_frame = self._candidate_start
        self._burst_baseline = self._candidate_baseline
        self._stop_med.clear()
        logger.info(
            "BURST detected at frame %d (t=%.2fs) — waiting for stillness",
            self._burst_frame, self._burst_frame / self.source_fps,
        )

    def _confirm_incident(self) -> None:
        self._state = "CAPTURING"
        self._capturing = True
        self._trigger_time = self._burst_frame / self.source_fps
        self._pre_frames = self._candidate_pre or list(self.buffer)
        self._post_frames = list(self._run_frames)
        self._capture_start_idx = self._burst_frame
        self._capture_baseline = self._burst_baseline or 0.0
        self._settle_frame = None
        logger.info(
            "INCIDENT CONFIRMED at frame %d (t=%.2fs): burst->stop | "
            "%d pre + %d pending post frames",
            self.frame_idx, self.frame_idx / self.source_fps,
            len(self._pre_frames), len(self._post_frames),
        )

    def _revert_to_scan(self) -> None:
        logger.info("Burst did not stop within %.1fs — reverting to scan",
                    self._cfg_i.burst_reject_s)
        self._reset_candidate()
        self._state = "SCANNING"
        self._burst_start_idx = None
        self._burst_frame = None
        self._burst_baseline = None
        self._stop_med.clear()

    def _scan_motion(self, encoded: bytes, motion: float) -> None:
        """Run the motion incident machine for one frame (SCANNING or BURST)."""
        cfg_i = self._cfg_i
        if self.frame_idx / self.source_fps > cfg_i.max_search_s:
            self._reset_candidate()
            return

        if self._state == "BURST":
            b = self._burst_baseline
            stop_thr = cfg_i.stop_factor * b if b else float("inf")
            self._stop_med.append(motion)
            window = int(cfg_i.stop_window_s * self.source_fps)
            if len(self._stop_med) >= window:
                if float(np.median(self._stop_med)) <= stop_thr:
                    self._confirm_incident()
                elif (self.frame_idx - self._burst_start_idx) >= int(
                        cfg_i.burst_reject_s * self.source_fps):
                    self._revert_to_scan()
            self._run_frames.append(encoded)
            return

        baseline = self._baseline()
        if baseline is None:
            return
        thr = cfg_i.burst_factor * baseline
        if motion > thr:
            if self._candidate_start is None:
                self._candidate_start = self.frame_idx
                self._candidate_pre = list(self.buffer)
                self._candidate_baseline = baseline
            else:
                self._run_frames.append(encoded)
                if (self.frame_idx - self._candidate_start) >= int(
                        cfg_i.burst_min_s * self.source_fps):
                    self._enter_burst()
        else:
            self._reset_candidate()

    # ── fallback: MOG2 entropy breach ───────────────────────────────────────
    def _trigger_entropy(self, ratio: float) -> None:
        self._capturing = True
        self._state = "CAPTURING"
        self._trigger_time = self.frame_idx / self.source_fps
        self._pre_frames = list(self.buffer)
        self._post_frames = []
        self._capture_start_idx = None
        self._capture_baseline = 0.0
        self._settle_frame = None
        logger.info(
            "EVENT TRIGGERED (entropy fallback) at frame %d (t=%.2fs) | "
            "ratio=%.4f > threshold=%.4f",
            self.frame_idx, self._trigger_time, ratio, self.threshold,
        )

    def process_frame(self, frame: np.ndarray) -> EventFrameBlock | None:
        """Advance the state machine by one frame; return a completed event or None."""
        cfg_t = self._cfg_t
        encoded = _encode_frame(frame)
        motion = self._motion_update(frame)

        # ── Capturing post-trigger frames ────────────────────────────────
        if self._capturing:
            self.frame_idx += 1
            return self._capture_frame(encoded, motion)

        # ── Cooldown ─────────────────────────────────────────────────────
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.buffer.append(encoded)
            self.frame_idx += 1
            return None

        # ── Warmup ratios (feed the entropy-threshold fallback) ──────────
        ratio: float | None = None
        if self.frame_idx < self.warmup_frame_count:
            fg_mask = self.bg_sub.apply(frame)
            ratio = _foreground_ratio(fg_mask)
            self.warmup_ratios.append(ratio)
            if self.frame_idx + 1 == self.warmup_frame_count:
                raw_threshold = (
                    np.percentile(self.warmup_ratios, cfg_t.percentile) * cfg_t.multiplier
                )
                self.threshold = max(raw_threshold, cfg_t.absolute_floor)
                logger.info(
                    "Warmup complete (frame %d). Threshold=%.4f (raw=%.4f, floor=%.4f)",
                    self.frame_idx + 1, self.threshold, raw_threshold, cfg_t.absolute_floor,
                )
        elif self.threshold is not None:
            fg_mask = self.bg_sub.apply(frame)
            ratio = _foreground_ratio(fg_mask)

        # ── Scan phase ───────────────────────────────────────────────────
        self.buffer.append(encoded)

        if self.frame_idx >= int(self._cfg_i.search_start_s * self.source_fps):
            self._scan_motion(encoded, motion)

        # Entropy fallback fires only while the motion machine is idle.
        if (ratio is not None and self.threshold is not None
                and ratio > self.threshold and self._state == "SCANNING"
                and self._candidate_start is None):
            self._trigger_entropy(ratio)
            self.frame_idx += 1
            return None

        self.frame_idx += 1
        return None

    def _capture_frame(self, encoded: bytes, motion: float) -> EventFrameBlock | None:
        """Accumulate a post-trigger frame; emit when the tail conditions are met."""
        cfg_i = self._cfg_i
        self._post_frames.append(encoded)

        # Legacy entropy capture: fixed post frame count.
        if self._capture_start_idx is None:
            if len(self._post_frames) >= self.post_frame_count:
                return self._emit_event()
            return None

        post_len_s = (self.frame_idx - self._capture_start_idx) / self.source_fps
        target_end_s: float | None = None
        thr = cfg_i.terminate_factor * self._capture_baseline
        if motion < thr:
            if self._settle_frame is None:
                self._settle_frame = self.frame_idx
            sustained = (self.frame_idx - self._settle_frame) / self.source_fps
            if sustained >= cfg_i.settle_s:
                start_s = (self._settle_frame - self._capture_start_idx) / self.source_fps
                target_end_s = max(
                    cfg_i.min_post_s,
                    min(start_s + cfg_i.settle_s + cfg_i.post_margin_s, cfg_i.max_post_s),
                )
        else:
            self._settle_frame = None

        budget = target_end_s if target_end_s is not None else cfg_i.max_post_s
        if post_len_s >= budget:
            return self._emit_event()
        return None

    def flush(self) -> EventFrameBlock | None:
        """
        Emit a partial event if a capture was in progress when the source ended.
        Used by the batch scanner at EOF; live workers never call this.
        """
        if self._capturing and self._post_frames:
            return self._emit_event()
        return None

    def _emit_event(self) -> EventFrameBlock:
        event = EventFrameBlock(
            trigger_time_sec=self._trigger_time,
            source_fps=self.source_fps,
            pre_frames=self._pre_frames or [],
            post_frames=self._post_frames or [],
        )
        logger.info(
            "Captured event: %d pre + %d post frames (%.1fs, trigger at %.2fs)",
            len(event.pre_frames), len(event.post_frames), event.duration_sec,
            self._trigger_time,
        )
        self._capturing = False
        self._pre_frames = None
        self._post_frames = None
        self._capture_start_idx = None
        self._settle_frame = None
        self._reset_candidate()
        self._state = "SCANNING"
        self._burst_start_idx = None
        self._burst_frame = None
        self._burst_baseline = None
        self._stop_frame = None
        self.buffer.clear()
        self.cooldown_remaining = self.cooldown_frame_count
        return event


def scan_for_events(video_path: str) -> list[EventFrameBlock]:
    """
    Scan a finite video file and return all triggered EventFrameBlocks.

    Drives a :class:`FrameEventDetector` over every frame, then flushes any
    partial event captured at end-of-file. Behavior is unchanged from the
    original monolithic scanner.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        raise IOError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    detector = FrameEventDetector(source_fps, total_frames=total_frames)

    logger.info(
        "Scanning %s | src_fps=%.1f | buffer=%d frames | post=%d frames",
        video_path, source_fps, detector.pre_buffer_size, detector.post_frame_count,
    )

    events: list[EventFrameBlock] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        event = detector.process_frame(frame)
        if event is not None:
            events.append(event)

    tail = detector.flush()
    if tail is not None:
        events.append(tail)

    cap.release()

    if not events:
        logger.warning("No events triggered in %s", video_path)

    return events
