# Track 4: XAI / Synthesis — How It Works

## Overview

Track 4 is the **explainability layer** — it turns structured perception + causal outputs into an LLM-written **Situation Report (SitRep)**. The key architectural constraint: **no raw imagery ever reaches the LLM**. Everything is structured text derived by the perception, causal, and embedding layers.

**Source:** `app/pipeline/synthesis.py`  
**Routes:** `app/routes/synthesis_routes.py`

---

## Pipeline

### 1. Build Evidence Packet (`_build_evidence`)

**Per-entity kinematics** (from causal CSV):
- Class, frames tracked, mean/peak speed (km/h)
- Whether it decelerated sharply (drop > 3 m/s)
- Speed uncertainty flag (if peak > `max_plausible_speed_mps`)
- Entry/exit timestamps

**SigLIP zero-shot colour attributes:**
- For vehicle entities, embeds crops against colour prompts ("a red vehicle", "a blue vehicle", etc.)
- Returns best-matching colour per entity — text only, no image to LLM.

**Incident indicators:**
- Flags vehicles that decelerated sharply AND whose track terminated mid-window (possible collision/impact).
- Notes the nearest other entity at the exit instant.

**Scene summary:**
- Vehicle/person counts, class breakdown, time window.

**Causal findings (Track 2):**
- Per-target: speed drop, external drivers (links where cause ≠ `tgt_speed`), interpretation string.

### 2. Format Evidence (`_format_evidence`)

Converts the structured evidence dict into a plain-text packet for the LLM:
```
EVENT EVT_...  (source: /path/to/video.mp4)
Window: -4.0s to +6.0s around the trigger
Scene: 5 vehicles + 2 persons tracked.

ENTITIES (kinematics from monocular bird's-eye-view; approximate):
  V_01: red car — present -4.0s..6.0s, mean 35.2 km/h, peak 52.1 km/h, DECELERATED sharply

INCIDENT INDICATORS (sharp deceleration + abrupt mid-window track loss — possible collision/impact):
  V_01 (red car): decelerated, then track terminated at 4.2s — closest entity: truck V_03 (2.1 m away)

CAUSAL ASSESSMENT (Track 2 / PCMCI+):
  Target: car V_01 (speed drop 5.2 m/s).
    Reactive coupling detected: lead_speed at lag 2 (strength 0.412)
```

### 3. Call LLM (`_call_llm`)

- Provider-agnostic: any **OpenAI-compatible** `/chat/completions` endpoint.
- Default: **Google Gemini** (`gemini-flash-latest`, free tier).
- Configurable via `$LLM_MODEL`, `$LLM_BASE_URL`, `$LLM_API_KEY` env vars or a repo-root `.env`.
- Uses stdlib `urllib` — no SDK dependency.
- Without an API key, the evidence packet is still built and persisted; the report is generated when a key is present.

### 4. Persist

- `dataset/{event_id}/sitrep.json` — full evidence + prompt + report
- `dataset/{event_id}/sitrep.md` — the report text only

---

## System Prompt

The LLM is instructed to:
- Write a concise, factual SitRep from structured data only
- Never invent vehicles, colours, actions, or outcomes
- Not speculate about fault or injuries
- Lead the Summary with likely collision/impact if incident indicators are present
- Structure: Summary → Entities Involved → Kinematic Timeline → Causal Assessment → Confidence & Caveats
- Keep under ~250 words

---

## Key Design Decisions

- **No imagery to LLM.** Everything is structured text — colour comes from SigLIP zero-shot, kinematics from BEV projection, causality from PCMCI+. This is an explicit privacy/safety constraint.
- **Speed clamping.** Speed spikes > `max_plausible_speed_mps` are clamped/flagged so implausible kinematics don't reach the model.
- **Provider-agnostic.** Works with any OpenAI-compatible endpoint — hosted, local, or edge.
- **Graceful degradation.** Without an API key, the evidence packet is still built and persisted.
- **Lazy initialization.** The `SynthesisEngine` singleton is created on first use.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/synthesis/{event_id}` | Build evidence + generate SitRep |
| `GET` | `/api/synthesis/{event_id}` | Fetch persisted SitRep |

---

## Output Schema (`sitrep.json`)

```json
{
  "status": "ok",
  "event_id": "EVT_...",
  "report": "## Situation Report\n\n...",
  "evidence": { ... },
  "prompt": "EVENT EVT_..."
}
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `gemini-flash-latest` | LLM model name |
| `base_url` | Google Gemini endpoint | OpenAI-compatible base URL |
| `max_tokens` | 1024 | Max generation tokens |
| `temperature` | 0.3 | Generation temperature |
| `api_key_env` | `LLM_API_KEY` | Env var name for API key |
| `max_entities` | 10 | Max entities in evidence packet |
| `min_entity_frames` | 5 | Min frames for entity inclusion |
