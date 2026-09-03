# Implementation Plan — Incident-Anchored Events + Scene-Level Causal Narrative

Goal: make the system produce the **same causal meaning** as `testVideo2TrueLabels.json`
(entity roles, staged narrative, typed stage-level relations, root-cause analysis) —

- without a whole-video perception pass (YOLO/BoT-SORT stays a single-window cost),
- keeping the existing per-target PCMCI micro-links,
- deterministic by default (works with no `LLM_API_KEY`; LLM only phrases things),
- verified against the gold file on *meaning*, not schema.

---

## 1. Problem statement (measured on testVideo2.mp4)

- `testVideo2TrueLabels.json` story: **Stable (0–11 s) → Motorcycle skid/fall (12–16 s) → Trailing SUV/sedan defensive braking (15–18 s) → Recovery (19–25 s)**.
- Current behavior: MOG2 entropy trigger fires at **25.97 s** (pre-buffer 10 s) → clip = **[15.97, 29.6 s]** — the *recovery tail*, accident excluded. The causal engine then legitimately finds "own-past-speed only" and the SitRep says nothing about the crash.
- Measured cheap signals (888 frames @ 30 FPS, 640×360):
  - **Foreground ratio** (current trigger input): biggest spike at **~26 s** (mean 0.904 — traffic resumes). The crash interval (12–18 s) is low-ratio. → Pure entropy **cannot isolate this incident**.
  - **Frame-difference motion**: peak at **~14 s** (7.5) = crash; sustained stillness **15–22 s** (motion bottoms ~1.5) = everyone stopped; motion resumes **23–29 s**. → Incident signature = **motion burst → stop → (resume)**.
  - Video-start sees a motion bump (sec 1: 6.6) then *continuous* flow — never a burst→stop, so it must be rejected via `search_start_s`.
- Cost of the motion scorer (benchmarked): 0.5 ms/frame vs MOG2 7.3 ms/frame → **+7%** over the existing Phase 0 scan. Reuses the single frame pass; **no second video read**.

## 2. Locked design decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Perception budget | Single-window only; **no whole-video perception** (rejected) |
| 2 | Re-anchor search scope | Motion scorer over the **whole video** (free, +7%) |
| 3 | Dynamic post-trigger signal | **Motion-based** termination (not ratio — ratio peaks at recovery) |
| 4 | Post-trigger cap | **Conservative** `max_post_s` ≈ 12 s |
| 5 | Micro-links | **Keep** PCMCI per-target links; stage-level links are an added `episode` block |
| 6 | Stage/RCA production | **Hybrid rules**: deterministic always; LLM phrases only if key present |
| 7 | Output contract | Meaning-equivalent to TrueLabels; *not* schema-identical |

## 3. Capture rework — online "burst-then-stop" state machine

Replace the entropy-first trigger in `FrameEventDetector.process_frame` with a
motion-native state machine (entropy trigger retained only as a **fallback** for
sources that never produce a burst-then-stop signature, e.g. streams):

```
SCANNING ──(motion > burst_factor × baseline for burst_min_s)──▶ BURST
   ▲                                                                │
   │                                    (motion stays high for       ▼
   │                                     burst_reject_s → revert)  CONFIRM (expect stillness)
   │                                                                │
   │                                     (stop_window_s of low motion)
   └────────────────────────────────────────────────────────────────▼
                                                            INCIDENT CONFIRMED
                                                            trigger = burst frame
                                                                  │
                                                                  ▼
CAPTURING ── post-frames until motion < terminate_factor × baseline for settle_s
             then + post_margin_s; clamp [min_post_s, max_post_s]; force-emit at cap
```

Implementation notes:

- Baseline = rolling median of motion over the last `baseline_window_s` (excluding the first `search_start_s`, the adaptation zone where frame-diff is inflated).
- The rolling JPEG pre-buffer (`deque`, existing) is frozen at the **burst frame**, so the clip naturally contains the approach (N1–N2), the crash (N3), response braking (N4), and the start of recovery (N5) — with the dynamic tail governed by #3.
- `EventFrameBlock.trigger_time_sec` = burst-frame time (already a field).
- Fallback: if no incident confirmed by end of source or within `max_search_s`, use the historical entropy breach path exactly as today.
- Streaming path (`monitor.py`) already drives the same `FrameEventDetector` → gains the new behavior automatically; cooldown set to `max_post_s + settle_s` to bound latency.

Expected result on testVideo2: anchor ≈ **14 s**, clip ≈ **[4, 22 s]**.

## 4. Config additions (`app/config.py`)

```python
@dataclass(frozen=True)
class IncidentConfig:
    search_start_s: float = 2.0        # skip early adaptation zone
    baseline_window_s: float = 5.0     # rolling motion baseline window
    burst_factor: float = 1.6          # burst = motion > this × baseline
    burst_min_s: float = 0.5           # burst must persist this long
    burst_reject_s: float = 3.0        # still-high after this → not an incident
    stop_window_s: float = 1.5         # stillness window required to confirm
    stop_factor: float = 0.6           # stillness = motion < this × baseline
    terminate_factor: float = 0.7      # post-trigger end threshold
    settle_s: float = 2.5              # sustained low motion to end capture
    post_margin_s: float = 2.0         # extra tail after settle
    min_post_s: float = 6.0            # floor on post-trigger length
    max_post_s: float = 12.0           # conservative ceiling (per decision #4)
    max_search_s: float = 300.0        # give up searching past this source time

@dataclass(frozen=True)
class SceneConfig:
    person_bike_overlap_frac: float = 0.35  # bbox IoU-ish overlap for rider/pillion
    person_bike_min_frames: int = 5         # frames of overlap to assign role
    rider_reaction_min_s: float = 0.5       # response-latency window bounds
    rider_reaction_max_s: float = 3.0
    stage_boundary_tol_s: float = 1.0       # boundary proximity tolerance (eval)
    skid_lateral_m: float = 3.0             # lateral deviation threshold for skid
    recovery_speed_thresh_mps: float = 2.0  # speeds above this = traffic resumed
```

(All values to be validated and tuned against testVideo2; documented defaults only.)

**Amendments to existing config (incorporated from `causal_fixes.md`):**

- `CausalConfig.tau_max`: 5 → **15** (1.5 s) — matches real car-following reaction latency (1–2 s), so stage-response links (N3→N4) are within reach of the micro analysis.
- `CausalConfig.tau_max_frame_frac`: 0.12 → **0.2** — so clips ≥ ~75 frames actually use the full τ=15 (with 0.12 the cap was redundant: `min(15, 12) = 12`).
- `InterpolationConfig.min_track_frames`: 3 → **8** — drops BoT-SORT ghost/fragment tracks globally (cleaner scene entity table, crops, RAG, and causal data).

## 5. Variable-length clip support

- `perception.py` already down-samples to 10 FPS target frames from the actual block; `Frame_ID` derives from real block length. The only hard-coded grid is `all_frame_ids = range(total_frames)` in `handoff.py:54` (`settings.video.total_frames = 160`).
- Refactor: thread the actual clip length (`len(frames)` / `EventFrameBlock.duration_sec`) into `_interpolate_tracks` so the uniform grid matches the real clip; keep the 160 value as pure fallback/legacy default.
- The 13.6 s event ran fine with the current code — this is a correctness cleanup, not a behavioural rewrite.

## 6. Scene semantics (`app/pipeline/scene.py`, new — deterministic)

Input: event causal CSV + `entity_crops/` (+ SigLIP colour via `synthesis._entity_attributes`).

- **Entity table** `entities: [{id, name, kind, object_ids, colour, role}]`:
  - Motorcycle / Rider / Pillion: person tracks whose bbox overlaps a motorcycle track for ≥ `person_bike_min_frames` frames at ≥ `person_bike_overlap_frac` → assign `rider` (front) / `pillion` (rear, if a second person overlaps).
  - Trailing 4-wheelers: `car`/`truck`/`bus` tracks behind the incident position (greater depth along travel direction), named by class + SigLIP colour → e.g. *White SUV*, *Silver Sedan*.
  - Aggregate **Traffic Flow (Upstream)**: all vehicle tracks ahead of the incident lane during the pre-incident window; represented as a stream entity with mean speed/density.
- Role assignment is deterministic; names for *vehicles* use colour; *persons* use role.
- Benefits from `min_track_frames: 8` (Fix 5): ghost/fragment tracks never reach the entity table.

## 7. Stage segmenter (`app/pipeline/stage.py`, new — deterministic)

Given the re-anchored CSV, output `nodes: [{node_id, window_s, state, involved_entities, evidence}]` in TrueLabels *meaning*:

| Stage | State | Detection rule |
|-------|-------|----------------|
| N1 | Stable | Clip start → first sustained speed-fall of the initiator |
| N2 | Trigger / Critical | Initiator (motorcycle/person) speed begins sustained fall from cruising level |
| N3 | Incident | Initiator speed ≈ 0, or mid-window track loss; lateral skid if `|lateral Δ| > skid_lateral_m`; persons down (bbox aspect collapses) |
| N4 | Hazard Response | Downstream vehicles' deceleration onsets in `[rider_reaction_min_s, rider_reaction_max_s]` after N3 start (temporal precedence) |
| N5 | Recovery | Initiator persons re-acquire upright bbox aspect; speeds of responders resume above `recovery_speed_thresh_mps` |

Boundaries are snapped to frames; each node carries `involved_entities` resolved from scene roles + frame-level presence.

Stage rules consume the **NaN-aware Savitzky-Golay smoothed** kinematics (Fix 4, `handoff.py`) — depth-gated objects keep smoothing on their finite segments instead of being skipped wholesale, removing the spike contamination that would otherwise misplace boundaries.

## 8. Stage-level relations (in `stage.py`, persisted as `episode`)

- **DIRECT_CAUSE** N2→N3 — same actor, kinematic continuity (speed collapse → stop/skid).
- **TRIGGERED_RESPONSE** N3→N4 — obstruction precedes responder braking; corroborated by the mean reaction lag across responders (and optionally by narrow-window PCMCI/cross-correlation as support, not proof).
- **CONSEQUENCE** N3→N5 — incident precedes recovery; mechanism from kinematic evidence.
- Mechanism strings are generated from scene facts (e.g. *"Obstruction in active lane prompted evasive braking by trailing vehicles (mean lag 1.2 s)"*).
- Output: new top-level `"episode": {nodes, relations, root_cause}` in `causal_graph.json`. The existing `targets` / micro-links stay untouched (decision #5), but their *quality* is raised by the incorporated fixes (self-link filter, `rel_speed` variable, NaN mask — see §15), so any micro-link cited as corroboration for a stage relation is an external, non-trivial link.

## 9. Root-cause analysis (`app/pipeline/rootcause.py`, new — deterministic)

- **primary_factor** — from initiator kinematics (motorcycle abrupt braking / traction-loss signature; if the initiator is a person, phrased accordingly).
- **contributing_factors** — lane density near incident, mean flanking-vehicle proximity before onset, ambient traffic speed.
- **mitigating_factors** — minimum gap maintained by trailing 4-wheelers (did they stop short), skid lane position, occupancy ambulation (persons remain visible/upright after N3).
- Emitted as structured evidence always. If `$LLM_API_KEY` is present, the LLM *re-phrases* each factor; never fabricates new facts (same policy as `synthesis.py`).

## 10. Track 4 synthesis (`app/pipeline/synthesis.py`)

- `_build_evidence` gains `episode` (nodes + relations) and `root_cause` from §7–§9.
- `sitrep.json` persists the structured claims **without a key**; prose `sitrep.md` only when a key exists (current behaviour preserved).
- `_SYSTEM` prompt updated to reference the staged narrative when present.

## 11. Frontend (`frontend/src/components/CausalPanel.tsx`)

- Render the `episode` as a **stage chain** (N1→N5 with states, timestamps, typed edge labels).
- Keep the existing per-target micro-link tables and CausalGraph visualizations below it (decision #5), updated for the incorporated fixes:
  - CausalGraph legend **"Lead Speed" → "Rel. Speed"** (Fix 3 renamed the variable).
  - Driver tables no longer list "Target's own past speed" rows (Fix 1 filters all self-links); targets with no external drivers show an empty driver list + the absent-driver interpretation.

## 12. Evaluation harness (`test_true_labels.py`, new — no key)

Meaning-based PASS/FAIL vs `testVideo2TrueLabels.json`:

1. **Entity roles present**: motorcycle, ≥1 rider, ≥2 trailing 4-wheelers (SUV/sedan equivalents by class count, not colour text).
2. **Stage ordering + boundaries**: produced stage sequence ⊇ [Stable, Trigger, Incident, Hazard, Recovery] in order; each boundary within `stage_boundary_tol_s` (±~1 s) of the gold timestamps where comparable.
3. **Relation direction + type**: episode contains a DIRECT_CAUSE (trigger→incident), a TRIGGERED_RESPONSE (incident→hazard), and a CONSEQUENCE (incident→recovery).
4. **Root-cause primary alignment**: `primary_factor` attributes the incident to the initiator's abrupt deceleration/traction loss (keyword/heuristic match).
- Exit code 0 on PASS (mirrors `test_causal.py` convention).

## 13. Implementation order

1. `config.py` — add `IncidentConfig`, `SceneConfig`; register in `PipelineConfig`; apply config amendments (τ=15, frac 0.2, `min_track_frames` 8).
2. `ingestion.py` — motion series + burst-then-stop state machine + fallback entropy path.
3. `handoff.py` / `perception.py` — variable-length grid refactor **+ NaN-aware SavGol (Fix 4)**.
4. `scene.py`, `stage.py`, `rootcause.py` — scene/role, stages, relations, root cause.
5. `causal.py` — persist `episode` block (additive; micro-links unchanged) **+ self-link filter (Fix 1), `rel_speed` variable (Fix 3), NaN-mask `DataFrame` input (Fix 6), extraction logging (Fix 7)**.
6. `synthesis.py` — evidence packet + prompt updated.
7. `CausalPanel.tsx` — episode chain rendering + legend/driver-table updates.
8. `test_true_labels.py` — eval harness.

## 14. Verification

- End-to-end run of `testVideo2.mp4` with the previously used `src_pts` via the frontend (Playwright): expect anchor ≈ 14 s, window ≈ [4, 22 s], `episode` with N1–N5 and typed relations, micro-links still present.
- `python test_true_labels.py` → PASS.
- Regression: `test_causal.py` still passes; existing events in the registry unaffected (new behaviour applies to new triggers).

## 15. Incorporated causal-engine quality fixes (from `causal_fixes.md`)

| Fix | Incorporate? | Where in this plan |
|-----|--------------|--------------------|
| **1. Filter all self-links** (drop `tau==0` condition, `causal.py:267`) | ✅ Yes | §8, §11, §13(5) — keeps micro-links per decision #5 but removes the trivially-true autoregression that dominated output (velocity lag-1 autocorr ~0.88–0.91) |
| **2. `tau_max` 5→15, `tau_max_frame_frac` 0.12→0.2** | ✅ Yes | §4 — 1.5 s reach matches real reaction latency and the N3→N4 response window |
| **3. `lead_speed` → `rel_speed = lead_speed − tgt_speed`** | ✅ Yes | §8, §11, §13(5) — car-following causality is driven by relative speed; avoids near-collinear raw speeds in ParCorr (replace, not add: multicollinearity) |
| **4. NaN-aware Savitzky-Golay** (`handoff.py:28`) | ✅ Yes | §7, §13(3) — one NaN currently disables smoothing for the whole series; fix smooths finite segments only, killing velocity spikes (40 m/s caps observed) |
| **5. `min_track_frames` 3→8** | ✅ Yes | §4, §6, §13(1) — global change; drops ghost/fragment tracks from CSV, crops, RAG, and scene table |
| **6. NaN mask instead of `MISSING=999.0` sentinel** | ✅ Yes | §13(5) — tigramite-native `mask=` input; fragile 999.0 sentinel removed |
| **7. Result-extraction logging** | ✅ Yes | §13(5) — logs timesteps×variables, kept/dropped vars, pre-filter link count; enables debugging `episode` + micro output |
| **8. Stationarity / ADF first-differencing** | ❌ Defer | Keep out (file's own recommendation): velocity is already a first-difference; adds `statsmodels` dependency |

Notes / accepted tradeoffs:

- Fix 3 loses the raw lead-speed signal; relative speed is strictly more informative for car-following causality (IDM/Wiedemann), and the stage-level `episode` analysis (not the micro layer) is what carries the narrative.
- Fix 1 changes the meaning of "no external drivers": with self-links gone, an empty driver list now indicates *no in-scene inter-vehicle influence* — the interpretation text in `synthesis.py` remains valid as-is.
- Fix 5 may drop short-lived real vehicles (<8 frames), but those contribute no usable causal signal and yield poor crops — accepted.
- `test_causal.py` uses LAG=5 (0.5 s); Fix 2 (τ up to 15) does not regress it, but note the synthetic test's lag is faster than real traffic.

## 16. Out of scope (deliberately)

- Whole-video perception / multi-window perception.
- Schema-identical output to TrueLabels (meaning equivalence only).
- Schema changes to the causal CSV or event registry.
- LLM-dependent stage/RCA correctness (deterministic path must stand alone).