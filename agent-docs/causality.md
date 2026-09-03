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
   - `rel_speed` — the lead vehicle's speed minus the target's (car-following causation is driven by relative speed, not raw speeds; replaces the raw `lead_speed` to avoid multicollinearity in ParCorr)
   - `nn_gap` — distance to nearest vehicle (any direction)
   - `nn_speed` — that nearest vehicle's speed
4. **Filter** variables present in ≥ `min_variable_presence_frac` of frames and drop constant ones.
5. **Run PCMCI+** with adaptive `tau_max` (capped by `tau_max_frame_frac × n_timesteps`) and `pc_alpha=None` (auto significance).
6. **Extract driver links** — any `-->` edge into `tgt_speed` at lag τ > 0. Links sorted by strength.
7. **Persist** to `dataset/{event_id}/causal_graph.json`.

Runs **automatically** at the end of Phase 2/3 for every event — batch pipeline (`_run_pipeline` in `app/routes/events.py`) and live feeds (`app/pipeline/monitor.py`) — non-fatally. `POST /api/causal/analyze/{event_id}` re-runs it manually.

---

## Key Design Decisions

- **Speed-primary by design.** Monocular-BEV acceleration is too noisy; variables are speeds and gaps only.
- **Multi-target.** Each braking vehicle is analyzed independently; all share the same event-level CSV.
- **Adaptive parameters.** `tau_max` and `pc_alpha` scale with the target's usable timestep count, not fixed.
- **Deterministic episode staging.** Every event also gets a staged N1–N5 narrative + typed relations + root cause (see below) — no LLM, same CSV → same episode.
- **Automatic, non-fatal.** Runs at the end of Phase 2 for every event; failure never fails the event — it stays "Extracted" and can be re-analyzed manually.

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
      "variables": ["tgt_speed", "lead_gap", "rel_speed"],
      "n_timesteps": 98,
      "tau_max": 10,
      "pc_alpha_used": 0.01,
      "drivers_of_target_speed": [
        {"cause": "rel_speed", "lag": 2, "strength": 0.412}
      ]
    }
  ],
  "episode": {
    "entities": [{"id": "E1", "role": "initiator", "object_ids": ["V_03"], "class": "motorcycle", ...}],
    "nodes": [{"node_id": "N1", "state": "Stable", "window_s": [-10.0, -4.0], ...}],
    "relations": [{"source_node": "N2", "target_node": "N3", "relation_type": "DIRECT_CAUSE", ...}],
    "root_cause": {"primary_factor": {...}, "contributing_factors": [...], "summary": "..."}
  },
  "note": "PCMCI+ over a short (~100-step) clip — treat links as ranked hypotheses, not proof."
}
```

---

## Episode Staging (deterministic)

Alongside the PCMCI+ targets, every event carries a **situation episode** under `episode` in the same `causal_graph.json` — a staged narrative over the clip (t=0 = the incident anchor):

- **Entities** — roles derived geometrically from the CSV: initiator (the motorcycle observed inside the crash window), rider/pillion (persons whose tracks start near the crash anchor), trailing four-wheelers, upstream traffic stream.
- **Nodes** — five stage windows: `N1 Stable → N2 Trigger/Critical → N3 Incident → N4 Hazard Response → N5 Recovery`.
- **Relations** — typed: `DIRECT_CAUSE` (N2→N3, same-actor kinematic collapse), `TRIGGERED_RESPONSE` (N3→N4, trailing braking within the reaction-lag window), `CONSEQUENCE` (N3→N5, recovery following the incident).
- **Root cause** — deterministic primary factor + contributing/mitigating factors from kinematics (speed collapse, skid, flanking proximity, trailing gaps).

Sources: `app/pipeline/scene.py` (entities/roles), `app/pipeline/stage.py` (stage windows + relations), `app/pipeline/rootcause.py` (primary factor). No LLM — same CSV yields the same episode; `test_true_labels.py` checks roles, stage order, boundary tolerance, relations and root-cause keywords against `testVideo2TrueLabels.json`.

---

## Caveats

- ~100-timestep clips are short for causal discovery; links are **ranked hypotheses**, not proof.
- On free-flow traffic, the correct result is *no* inter-vehicle causality (only autoregression).
- Requires at least 2 variables and `min_series_len` timesteps per target.
