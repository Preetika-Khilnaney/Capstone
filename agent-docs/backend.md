# Backend — How It Works

## Overview

A **FastAPI** application serving the video surveillance pipeline as a REST API. The backend runs the three-phase pipeline as a background task and exposes endpoints for all four tracks.

**Source:** `app/`  
**Entry point:** `app/main.py`  
**Start:** `uvicorn app.main:app --reload --port 8000`

---

## Application Lifecycle

```
Startup
  ├── _setup_logging()        # Dual handler: console + file (track1.log)
  ├── _ensure_directories()   # Create dataset/, config/, logs/ dirs
  ├── init_db()               # SQLite migrations + back-fill
  └── app.include_router()    # Register all route modules

Shutdown
  └── get_feed_manager().stop_all()  # Stop live feed monitors
```

---

## Route Modules

| Module | Prefix | Tags |
|--------|--------|------|
| `routes/events.py` | `/api` | Track 1 |
| `routes/causal_routes.py` | `/api/causal` | Track 2 |
| `routes/rag_routes.py` | `/api/rag` | Track 3 |
| `routes/synthesis_routes.py` | `/api/synthesis` | Track 4 |
| `routes/feed_routes.py` | `/api/feeds` | Camera Ingestion |

---

## Core Endpoints

### Pipeline (Track 1)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/pipeline/run` | Trigger pipeline on a video file. Returns `event_id` immediately; processing runs in background. |
| `GET` | `/api/events` | List all events |
| `GET` | `/api/events/{id}` | Event detail |
| `GET` | `/api/events/{id}/csv` | Download causal CSV |
| `GET` | `/api/events/{id}/video` | Download event clip |
| `GET` | `/api/events/{id}/crops` | List entity crops |
| `GET` | `/api/events/{id}/crops/{filename}` | Serve a crop image |
| `GET` | `/api/events/{id}/source-video` | Serve full source video |

### Video Sources

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/sources` | List registered video sources |
| `GET` | `/api/sources/{id}` | Source detail |
| `GET` | `/api/sources/{id}/stream` | Stream source video |
| `GET` | `/api/sources/{id}/events` | Events from a source |

### Causal (Track 2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/causal/analyze/{id}` | Run PCMCI+ causal discovery |
| `GET` | `/api/causal/{id}` | Fetch persisted causal graph |

### RAG (Track 3)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/rag/ingest/{id}` | Embed and index event crops |
| `POST` | `/api/rag/search` | Semantic search |

### Synthesis (Track 4)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/synthesis/{id}` | Generate SitRep |
| `GET` | `/api/synthesis/{id}` | Fetch persisted SitRep |

### Feeds (Live Monitoring)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/feeds` | Status of all feed monitors |
| `POST` | `/api/feeds/{id}/start` | Start monitoring a source |
| `POST` | `/api/feeds/{id}/stop` | Stop monitoring a source |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/logs` | Last 100 lines of track1.log |
| `GET` | `/api/config` | Current pipeline configuration |

### Dynamic Homography (API Payloads)

Endpoints that ingest video (`POST /api/pipeline/run` and `POST /api/feeds/{id}/start`) accept an optional JSON payload to dynamically calculate the homography matrix on the fly, skipping the static `config/homography.npy` file.

- `camera_id` (str, optional): The ID of the camera.
- `src_pts` (list of list of ints, optional): A 4x2 matrix of pixel coordinates representing a 3.5m x 14.0m physical lane segment. Format: `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` ordered as near-left, near-right, far-right, far-left.

---

## Pipeline Execution Flow

```
POST /api/pipeline/run
  ├── Insert event row (status="Processing") — synchronous
  ├── Background task: _run_pipeline()
  │   ├── Phase 0: scan_for_events() → EventFrameBlock
  │   ├── Phase 1: process_event() → PerceptionResult
  │   ├── Phase 2: finalize_event() → CSV, video, crops, status="Extracted"
  │   └── Phase 3: CausalEngine.analyze_event() → causal_graph.json (non-fatal)
  └── Return { event_id, status: "processing" }
```

---

## Storage

| Layer | Path | Purpose |
|-------|------|---------|
| SQLite | `event_registry.db` | Events + Video Sources (WAL mode) |
| Per-event | `dataset/{Event_ID}/` | Clip, CSV, crops, causal graph, SitRep |
| LanceDB | `dataset/lancedb/` | RAG vector table |

---

## Configuration

All tuning constants live in `app/config.py` as a frozen-dataclass `settings` singleton. Change constants there, not inline. Key config groups:
- `video` — FPS, buffer sizes, timing
- `threshold` — MOG2 entropy fallback trigger
- `incident` — motion "burst-then-stop" trigger (burst/stop/termination knobs)
- `scene` — episode-staging params (roles, skid, reaction-lag, recovery)
- `yolo` — model, backend, confidence, device
- `tracker` — BoT-SORT parameters
- `interpolation` — track lifespan filter, gap interpolation, velocity smoothing
- `projection` — reliable-region depth gate
- `crop` — crop selection
- `causal` — PCMCI+ parameters
- `rag` — SigLIP model + LanceDB
- `synthesis` — LLM endpoint, model, tokens
- `feed` — Live monitoring parameters
- `paths` — filesystem locations

---

## CORS

Fully open (`allow_origins=["*"]`) for development. Restrict in production.
