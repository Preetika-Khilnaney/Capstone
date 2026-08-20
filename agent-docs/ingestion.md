# Track 1: Ingestion — How It Works

## Overview

Phase 0 scans video sources for incident events. The **primary trigger is a motion-native "burst-then-stop" state machine**: a short, sharp motion burst (a crash / sudden obstruction) that culminates in stillness confirms the incident. A clip is captured around the confirmed burst — a pre-buffer before it and a dynamic, motion-terminated tail after it. The historical MOG2 foreground-ratio breach is retained only as a **fallback** for sources that never produce a burst-then-stop signature (e.g. long flowing traffic or streams).

**Source:** `app/pipeline/ingestion.py`  
**Entry:** `scan_for_events()` for batch files; `FrameEventDetector.process_frame()` for both batch and live streams.

---

## State Machine (`FrameEventDetector`)

A push-driven detector fed one frame at a time. Works identically for finite video files and unbounded live streams.

### Primary path — motion "burst-then-stop"

```
SCANNING ─(motion > burst_factor × baseline for burst_min_s)─▶ BURST
   ▲                                                            │
   │                       (motion stays high for                ▼
   │                        burst_reject_s → revert)          CONFIRM?
   │                                                            │
   │          (stop_window_s of stillness:                      ▼
   │           median motion < stop_factor × baseline)   INCIDENT CONFIRMED
   │                                                        trigger = burst frame
   └────────────── CAPTURING ─▶ COOLDOWN ─▶ SCANNING
```

| State | Behavior |
|-------|----------|
| **SCANNING** | Rolling median of frame-difference motion over `baseline_window_s` (first `search_start_s` skipped as an adaptation zone) forms the baseline. Maintain a rolling pre-buffer of JPEG-encoded frames. Motion > `burst_factor × baseline` for `burst_min_s` → burst candidate. |
| **BURST** | Candidate must be **confirmed**: median motion over within `stop_window_s` must stay ≤ `stop_factor × baseline` (stillness = the crash signature). If motion stays high past `burst_reject_s`, the burst **reverts** to SCANNING — rejects false starts (e.g. a truck passing). On confirmation → CAPTURING; the baseline is pinned to its burst-frame value. |
| **CAPTURING** | Collect post-trigger frames until motion drops below `terminate_factor × baseline` for `settle_s`, then keep `post_margin_s` extra; the tail is clamped to `[min_post_s, max_post_s]` and force-emitted at the cap. Emit `EventFrameBlock` with trigger = the burst frame (the **incident anchor**, t=0), then COOLDOWN. |
| **COOLDOWN** | Ignore triggers for `cooldown_seconds` after an event. Prevents duplicate events from the same incident. |

### Fallback path — MOG2 foreground-ratio breach

- During WARMUP (first `warmup_seconds` of frames), accumulate MOG2 foreground ratios; at the end compute an adaptive threshold `percentile × multiplier`, floored at `absolute_floor`.
- After warmup the ratio is checked every frame; a breach fires **only while the motion machine is idle** (no active burst candidate), capturing a fixed `post_trigger_seconds` tail. It is the sole trigger for sources that never burst-then-stop.

---

## Event Output (`EventFrameBlock`)

```python
@dataclass
class EventFrameBlock:
    trigger_time_sec: float   # time of incident confirmation = the burst frame
    source_fps: float         # original video FPS
    pre_frames: list[bytes]   # JPEG-encoded (pre-buffer before the burst)
    post_frames: list[bytes]  # JPEG-encoded (motion-terminated tail after the burst)
```

- `all_frames` → flat list of JPEG bytes for downstream phases
- `duration_sec` → total clip length (~16s typical: 10s pre + dynamic 6–12s post)

---

## Key Design Decisions

- **Push-driven, not pull-driven.** The detector doesn't open video; callers feed frames. This makes it reusable for both batch (`scan_for_events`) and live (`FeedMonitor`).
- **Burst-then-stop is the primary signal.** The rolling-median motion baseline adapts to the scene; stop confirmation rejects false starts (bursts that never culminate in stillness are reverted rather than triggering).
- **Burst anchor at t=0.** The confirmed burst frame becomes the clip's t=0 anchor, exported as the incident reference for downstream phase-2 staging.
- **Motion-terminated capture.** Post-trigger recording stops when the scene settles instead of on a fixed clock, clamped to `[min_post_s, max_post_s]`.
- **JPEG encoding in memory** reduces memory footprint vs. keeping raw numpy arrays.
- **Warmup auto-scales for short videos.** If `warmup_seconds` ≥ total frames, it is capped to 30% of the file (min ~1s) so scan room always remains.
- **Batch `scan_for_events` flushes at EOF** — a capture still in progress when the file ends is emitted as a partial event.
- **Only the first triggered event is processed** (prototype limitation; enforced in `app/routes/events.py`).

---

## Trigger Configuration

Clip shape (`video.*`) and MOG2 fallback (`threshold.*`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pre_buffer_seconds` | 10s | Pre-trigger (pre-burst) frames to keep |
| `post_trigger_seconds` | 6s | Fixed post tail — used only by the legacy fallback path |
| `jpeg_quality` | 85 | JPEG encoding quality |
| `warmup_seconds` | 120s | Frames to accumulate before computing the fallback threshold (auto-scaled for short videos) |
| `cooldown_seconds` | 15s | Minimum gap between events |
| `percentile` | 95th | Percentile of warmup ratios for the fallback threshold |
| `multiplier` | 1.5 | Fallback threshold = percentile × multiplier |
| `absolute_floor` | 0.05 | Minimum fallback threshold regardless of percentile |

Motion machine (`incident.*`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `search_start_s` | 2.0 | Skip early adaptation zone (frame-diff inflated) |
| `baseline_window_s` | 5.0 | Rolling motion baseline window |
| `burst_factor` | 1.6 | Burst = motion > this × baseline |
| `burst_min_s` | 0.5 | Burst must persist this long |
| `burst_reject_s` | 10.0 | Still high after this → not an incident, revert |
| `stop_window_s` | 1.5 | Stillness window required to confirm the incident |
| `stop_factor` | 0.65 | Stillness = median motion < this × baseline |
| `terminate_factor` | 0.7 | Post-trigger end threshold |
| `settle_s` | 2.5 | Sustained low motion to end capture |
| `post_margin_s` | 2.0 | Extra tail after settle |
| `min_post_s` | 6.0 | Floor on post-trigger length |
| `max_post_s` | 12.0 | Ceiling on post-trigger length |
| `max_search_s` | 300.0 | Give up motion searching past this source time |
| `motion_downscale` | 4 | Downsample factor for the motion score |

---

## Usage

### Batch (video file)
```python
from app.pipeline.ingestion import scan_for_events
events = scan_for_events("/path/to/video.mp4")
```

### Live stream
The `FeedMonitor` in `app/pipeline/monitor.py` drives the same `FrameEventDetector` for each frame from a live camera feed.