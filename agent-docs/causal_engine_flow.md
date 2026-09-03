# Causal Engine (Track 2) — High-Level Flow

## What It Does

Given a traffic incident clip, it asks: **"Which vehicle's behavior most influenced the target vehicle's speed, and how?"** It uses a causal discovery algorithm (PCMCI+) to find lagged cause-effect relationships between vehicles.

---

## The Flow

```
┌─────────────────────────────────────────────────────────┐
│              INPUT: CAUSAL CSV                          │
│  Per-object kinematics from Phase 2 (perception):      │
│  Position (x,y), Velocity, Class, BBox per frame       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 1: SELECT TARGET VEHICLES                  │
│                                                         │
│  For each vehicle, compute:                             │
│  - Sustained speed drop (running max minus current)     │
│  - Lead fraction (% of frames with a vehicle ahead)     │
│                                                         │
│  Filter: speed drop >= 1.5 m/s                          │
│  Rank: followers first (lead_frac >= 0.3), then others  │
│  Cap: max 3 targets per event                           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 2: BUILD VARIABLES (per target)            │
│                                                         │
│  For each target, find per-frame:                       │
│  - tgt_speed:  target's own speed (the EFFECT)          │
│  - lead_gap:   distance to vehicle ahead in same lane   │
│  - rel_speed:  lead_speed − tgt_speed (car-following)   │
│  - nn_gap:     distance to nearest vehicle              │
│  - nn_speed:   nearest vehicle's speed                  │
│                                                         │
│  "Same lane" = within 4.0m lateral offset               │
│  "Ahead" = positive projection on travel direction      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 3: FILTER & ASSEMBLE                       │
│                                                         │
│  - Drop variables present in < 50% of frames            │
│  - Drop constant variables (std ≈ 0, ParCorr can't use) │
│  - Interpolate short gaps (≤ 5 frames)                  │
│  - Build observation mask for tigramite                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 4: RUN PCMCI+                              │
│                                                         │
│  tigramite's PCMCI+ with ParCorr independence test      │
│  - Adaptive tau_max: min(15, n_frames × 0.2)           │
│  - pc_alpha: auto-selected by tigramite                 │
│  - Searches for lagged causal links to tgt_speed        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 5: EXTRACT DRIVERS                         │
│                                                         │
│  Find all "-->" edges INTO tgt_speed                    │
│  Filter out self-links (autoregressive)                 │
│  Map abstract variables back to physical vehicles:      │
│    lead_gap/rel_speed → which vehicle was the lead?     │
│    nn_gap/nn_speed    → which vehicle was nearest?      │
│                                                         │
│  Each driver: { cause, lag, strength, cause_object }    │
│  Sort by |strength| descending                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 6: BUILD EPISODE (N1–N5)                   │
│                                                         │
│  Scene roles: initiator, rider, pillion, trailing,      │
│               upstream stream                           │
│  Narrative stages:                                      │
│    N1 Stable → N2 Trigger → N3 Incident →              │
│    N4 Hazard Response → N5 Recovery                     │
│  Relations: DIRECT_CAUSE, TRIGGERED_RESPONSE,           │
│             CONSEQUENCE                                 │
│  Root cause: traction_loss or abrupt_deceleration       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         OUTPUT: causal_graph.json                       │
│  - targets[] with drivers + lag + strength              │
│  - episode with entities, stages, relations, root cause │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Speed-primary** (no acceleration) | Monocular-BEV acceleration is too noisy to trust |
| **Relative speed** (`lead_speed − tgt_speed`) instead of raw `lead_speed` | Avoids multicollinearity between correlated vehicle speeds |
| **Followers ranked first** | Vehicles with a consistent lead are more likely to have causal links |
| **Adaptive tau_max** | Short clips need shorter lag windows; 100 frames × 0.2 = 20 lags max |
| **Self-links filtered out** | Autoregression is expected; we want inter-vehicle causality |
| **Variables mapped back to physical OIDs** | PCMCI+ works on abstract variable names; the mapping step tells you *which specific vehicle* caused the effect |

---

## Episode Staging (N1–N5)

The episode is a **deterministic narrative** built from kinematics, not from the causal graph:

| Stage | What Happens | How It's Detected |
|-------|-------------|-------------------|
| **N1 Stable** | Normal traffic flow | Before initiator's braking onset |
| **N2 Trigger** | Initiator begins braking | Speed drops below cruise by ≥ 25% or 1 m/s |
| **N3 Incident** | Initiator stops / track loss / skid | Speed ≤ 0.8 m/s, or track terminated, or lateral skid ≥ 3m |
| **N4 Hazard Response** | Trailing vehicles brake | Braking onset within 0.5–3.0s after N3 |
| **N5 Recovery** | Traffic resumes | Responders re-accelerate, or persons stand up, or fresh traffic arrives after responders bottom out |

---

## Scene Roles

| Role | How Assigned |
|------|-------------|
| **Initiator** | Motorcycle observed inside the crash window [-1s, +5s] (most frames visible) |
| **Rider** | Nearest person whose track starts near the crash anchor (within 25m) |
| **Pillion** | Second person within 15m of rider |
| **Trailing** | 4-wheelers (car/truck/bus) positioned behind the initiator (dot product with road direction < 0) |
| **Upstream stream** | Vehicles ahead of the initiator (dot product > 0), aggregate stats |

---

## Root Cause

Deterministic classification based on kinematic facts:

| Primary Factor | Condition |
|---------------|-----------|
| **`traction_loss`** | Skid detected (lateral deviation ≥ 3m) OR persons down (bbox aspect ratio < 0.65) |
| **`abrupt_deceleration`** | Otherwise — initiator decelerated sharply without loss of traction |

**Contributing factors**: traffic density, surrounding vehicle proximity, ambient speed.

**Mitigating factors**: following distance, lane position, occupants upright at end.

---

## Output Structure: `causal_graph.json`

```json
{
  "status": "ok",
  "event_id": "EVT_...",
  "targets": [
    {
      "target_object": "V_03",
      "target_class": "motorcycle",
      "target_speed_drop_mps": 12.5,
      "target_lead_fraction": 0.0,
      "variables": ["tgt_speed", "nn_gap", "nn_speed"],
      "n_timesteps": 98,
      "tau_max": 15,
      "pc_alpha_used": 0.01,
      "drivers_of_target_speed": [
        {
          "cause": "nn_speed",
          "lag": 3,
          "strength": -0.312,
          "cause_object": "V_01",
          "cause_object_frac": 0.92
        }
      ]
    }
  ],
  "episode": {
    "entities": [...],
    "stages": [...],
    "relations": [...],
    "root_cause": {...}
  }
}
```
