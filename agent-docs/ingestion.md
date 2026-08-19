# Track 1: Ingestion — How It Works

## Overview

Phase 0 scans video sources for anomaly events using **MOG2 background subtraction**. On trigger, it captures a 10-second clip (4s pre-buffer + 6s post-trigger) and passes it to the perception phase.

**Source:** `app/pipeline/ingestion.py`  
**Entry:** `scan_for_events()` for batch files; `FrameEventDetector.process_frame()` for both batch and live streams.

---

## State Machine (`FrameEventDetector`)

A push-driven detector fed one frame at a time. Works identically for finite video files and unbounded live streams.

```
WARMUP → SCANNING → CAPTURING → COOLDOWN → SCANNING → ...
```

| State | Behavior |
|-------|----------|
| **WARMUP** | Accumulate foreground ratios for `warmup_seconds` (default 120s). At the end, compute adaptive threshold: `percentile × multiplier`, floored at `absolute_floor`. |
| **SCANNING** | Maintain rolling pre-buffer (deque of JPEG-encoded frames). On threshold breach → freeze pre-buffer, switch to CAPTURING. |
| **CAPTURING** | Collect `post_trigger_seconds` (6s) of frames. On completion → emit `EventFrameBlock`, enter COOLDOWN. |
| **COOLDOWN** | Ignore triggers for `cooldown_seconds` after an event. Prevents duplicate events from the same incident. |

---

## Event Output (`EventFrameBlock`)

```python
@dataclass
class EventFrameBlock:
    trigger_time_sec: float   # time of threshold breach
    source_fps: float         # original video FPS
    pre_frames: list[bytes]   # JPEG-encoded (4s pre-buffer)
    post_frames: list[bytes]  # JPEG-encoded (6s post-trigger)
```

- `all_frames` → flat list of JPEG bytes for downstream phases
- `duration_sec` → total clip length (~10s)

---

## Key Design Decisions

- **Push-driven, not pull-driven.** The detector doesn't open video; callers feed frames. This makes it reusable for both batch (`scan_for_events`) and live (`FeedMonitor`).
- **JPEG encoding in memory** reduces memory footprint vs. keeping raw numpy arrays.
- **Only the first triggered event is processed** (prototype limitation).
- **Adaptive threshold** handles varying lighting/camera conditions rather than a fixed foreground-ratio cutoff.

---

## Trigger Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_seconds` | 120s | Frames to accumulate before computing threshold |
| `cooldown_seconds` | 30s | Minimum gap between events |
| `pre_buffer_seconds` | 4s | Pre-trigger frames to keep |
| `post_trigger_seconds` | 6s | Post-trigger frames to capture |
| `jpeg_quality` | 85 | JPEG encoding quality |
| `percentile` | 95th | Percentile of warmup ratios for threshold |
| `multiplier` | 1.5 | Threshold = percentile × multiplier |
| `absolute_floor` | 0.02 | Minimum threshold regardless of percentile |

---

## Usage

### Batch (video file)
```python
from app.pipeline.ingestion import scan_for_events
events = scan_for_events("/path/to/video.mp4")
```

### Live stream
The `FeedMonitor` in `app/pipeline/monitor.py` drives the same `FrameEventDetector` for each frame from a live camera feed.
