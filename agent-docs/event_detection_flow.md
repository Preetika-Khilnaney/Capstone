# Event Detection Mechanism — High-Level Flow

```
┌─────────────────────────────────────────────────────────┐
│                    FRAME ARRIVES                        │
│              (from file or live feed)                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              JPEG-ENCODE THE FRAME                      │
│          (reduces memory footprint)                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              COMPUTE TWO SCORES IN PARALLEL             │
│                                                         │
│  1. MOTION SCORE:                                       │
│     grayscale → downsample 4× → compare to frame        │
│     2 positions back → mean absolute pixel diff         │
│                                                         │
│  2. MOG2 FOREGROUND RATIO:                              │
│     background subtraction → fraction of foreground     │
│     pixels in the mask                                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              BUFFER THE FRAME                           │
│     Add to rolling deque (10s worth of frames)         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  WARMUP PHASE  │
              │  (first 120s)  │
              └───────┬────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Accumulate MOG2          Compute fallback
   foreground ratios        threshold
   (for threshold)          (95th %ile × 1.5)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│           CHECK CURRENT STATE                          │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │COOLDOWN │   │WARMUP   │   │SCANNING │
   │(15s)    │   │         │   │         │
   │         │   │         │   │         │
   │Skip     │   │No       │   │BOTH     │
   │detect.  │   │detect.  │   │MACHINES │
   │Buffer   │   │Buffer   │   │ACTIVE   │
   │only     │   │only     │   │         │
   └─────────┘   └─────────┘   └────┬────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │                             │
                      ▼                             ▼
          ┌──────────────────────┐    ┌──────────────────────┐
          │  MOTION MACHINE      │    │  MOG2 MACHINE        │
          │  (primary)           │    │  (fallback)          │
          │                      │    │                      │
          │  motion > 1.6×       │    │  fg_ratio >          │
          │  baseline for 0.5s?  │    │  adaptive_threshold? │
          └──────────┬───────────┘    └──────────┬───────────┘
                     │                           │
           ┌─────────┴─────────┐                 │
           │YES                │NO               │
           ▼                   ▼                 │
  ┌────────────────┐  ┌────────────────┐         │
  │  ENTER BURST   │  │  NO ACTION     │         │
  │  STATE         │  │  (wait)        │         │
  └───────┬────────┘  └────────────────┘         │
          │                                      │
          ▼                                      │
  ┌────────────────────────────┐                 │
  │  ROLLING MEDIAN OVER       │                 │
  │  1.5s DROPS TO             │                 │
  │  ≤ 0.65 × BASELINE?       │                 │
  └───────────┬────────────────┘                 │
              │                                  │
    ┌─────────┴─────────┐                        │
    │YES                │NO (after 10s)          │
    ▼                   ▼                        │
┌──────────────┐   ┌──────────────┐              │
│ CONFIRM      │   │ REVERT TO    │              │
│ INCIDENT     │   │ SCANNING     │              │
│              │   │ (false start)│              │
└──────┬───────┘   └──────────────┘              │
       │                                         │
       │    ┌────────────────────────────────────┘
       │    │
       ▼    ▼
┌─────────────────────────────────────────────────────────┐
│           TRIGGER CONFIRMED                            │
│                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐      │
│  │  MOTION PATH        │  │  MOG2 PATH          │      │
│  │                     │  │                     │      │
│  │  Pre: 10s buffer    │  │  Pre: 10s buffer    │      │
│  │  Post: dynamic      │  │  Post: fixed 6s     │      │
│  │  (6–12s, settle     │  │                     │      │
│  │   logic)            │  │                     │      │
│  └─────────────────────┘  └─────────────────────┘      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              EMIT EVENT                                │
│  Return EventFrameBlock with:                          │
│    - 10s pre-trigger frames (from buffer snapshot)     │
│    - Post-trigger frames (6–12s motion / 6s MOG2)     │
│    - Trigger time = burst frame = t=0                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              COOLDOWN (15s)                            │
│  Skip detection, buffer frames only                    │
│  Prevents duplicate triggers from same incident        │
└─────────────────────────────────────────────────────────┘
```

## Key Differences Between the Two Paths

| Aspect | Motion (primary) | MOG2 (fallback) |
|--------|------------------|------------------|
| **Trigger** | Spike + calm signature | Foreground ratio breach |
| **Priority** | First — runs always | Only when motion machine is idle |
| **Post-trigger** | Dynamic (6–12s, settle-based) | Fixed (6s) |
| **Use case** | Collisions, sudden stops | Restricted zone entry, continuous heavy activity |
| **Rejection** | 10s timeout reverts false starts | None — fires once and captures |

## Why Two Mechanisms?

- **Motion** detects incidents with a clear **spike-then-calm** pattern (collisions, emergency braking). This is the majority of traffic incidents.
- **MOG2** catches incidents **without** that pattern (e.g., a vehicle entering a no-go zone, or a gradual buildup of activity that crosses a threshold). It's a safety net for edge cases the motion machine would miss.
