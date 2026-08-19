# Track 2: Causal Engine — How It Works

## Overview

The Causal Engine performs **multi-target causal discovery** over event kinematics using **PCMCI+** (tigramite). It answers: *what caused a vehicle's speed to change, and at what time lag?*

**Source:** `app/pipeline/causal.py`  
**Routes:** `app/routes/causal_routes.py`

---

## Pipeline

1. **Load** the event's causal CSV (`{event_id}_causal_data.csv`) produced by Phase 2 handoff.
2. **Rank** candidate target vehicles — those with a sustained speed drop (`min_speed_drop_mps`). Followers (lead present ≥ `follower_lead_fraction_threshold` of frames) rank ahead of non-followers, both sorted by drop magnitude. Capped at `max_targets`.
3. **Build target-centric variables** per candidate:
   - `tgt_speed` — the target's speed (the effect variable)
   - `lead_gap` — forward distance to nearest same-lane vehicle ahead
   - `lead_speed` — that lead vehicle's speed
   - `nn_gap` — distance to nearest vehicle (any direction)
   - `nn_speed` — that nearest vehicle's speed
4. **Filter** variables present in ≥ `min_variable_presence_frac` of frames and drop constant ones.
5. **Run PCMCI+** with adaptive `tau_max` (capped by `tau_max_frame_frac × n_timesteps`) and `pc_alpha=None` (auto significance).
6. **Extract driver links** — any `-->` edge into `tgt_speed` at lag τ > 0. Links sorted by strength.
7. **Persist** to `dataset/{event_id}/causal_graph.json`.

---

## Key Design Decisions

- **Speed-primary by design.** Monocular-BEV acceleration is too noisy; variables are speeds and gaps only.
- **Multi-target.** Each braking vehicle is analyzed independently; all share the same event-level CSV.
- **Adaptive parameters.** `tau_max` and `pc_alpha` scale with the target's usable timestep count, not fixed.
- **Non-fatal.** Causal failure doesn't fail the event — the event stays "Extracted" and can be re-analyzed manually.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/causal/analyze/{event_id}` | Run PCMCI+ on the event |
| `GET` | `/api/causal/{event_id}` | Fetch persisted causal graph |

---

## Output Schema (`causal_graph.json`)

```json
{
  "status": "ok",
  "event_id": "EVT_...",
  "targets": [
    {
      "target_object": "V_01",
      "target_class": "car",
      "target_speed_drop_mps": 5.2,
      "target_lead_fraction": 0.85,
      "variables": ["tgt_speed", "lead_gap", "lead_speed"],
      "n_timesteps": 98,
      "tau_max": 10,
      "drivers_of_target_speed": [
        {"cause": "lead_speed", "lag": 2, "strength": 0.412}
      ]
    }
  ]
}
```

---

## Caveats

- ~100-timestep clips are short for causal discovery; links are **ranked hypotheses**, not proof.
- On free-flow traffic, the correct result is *no* inter-vehicle causality (only autoregression).
- Requires at least 2 variables and `min_series_len` timesteps per target.
