# Frontend — How It Works

## Overview

A **Next.js 16** (App Router, React 19) dashboard that provides the UI for the video surveillance platform. It communicates with the FastAPI backend at `http://localhost:8000`.

**Source:** `frontend/src/`  
**Dev server:** `npm run dev` (port 3000)

---

## Architecture

```
frontend/src/
├── app/                  # Next.js App Router pages
│   ├── page.tsx          # Home/dashboard
│   ├── events/           # Event list & detail views
│   ├── search/           # RAG semantic search UI
│   ├── upload/           # Video upload/trigger UI
│   └── config/           # System configuration view
├── components/
│   ├── Sidebar.tsx       # Navigation sidebar
│   ├── CausalPanel.tsx   # Causal analysis display
│   ├── CausalGraph.tsx   # Causal graph visualization
│   └── VideoAnnotator.tsx # Video player with annotations
└── lib/
    └── api.ts            # API client (all backend calls)
```

---

## Key Pages

| Route | Purpose |
|-------|---------|
| `/` | Dashboard — system status, recent events |
| `/events` | List all events with status filtering |
| `/events/[id]` | Event detail — crops, video, CSV, causal graph, SitRep |
| `/search` | RAG semantic search over indexed entity crops |
| `/upload` | Trigger pipeline on a video file path (now supports optional `cameraId` and `srcPts` for dynamic homography calibration) |
| `/config` | View current pipeline configuration |

---

## API Client (`lib/api.ts`)

The client wraps all backend endpoints:

### Track 1 (Events)
- `fetchEvents()` — GET /api/events
- `fetchEventDetail(eventId)` — GET /api/events/{id}
- `triggerPipeline(videoPath, cameraId?, srcPts?)` — POST /api/pipeline/run
- `fetchCrops(eventId)` — GET /api/events/{id}/crops
- URL builders: `getCropUrl()`, `getCsvUrl()`, `getVideoUrl()`, `getSourceVideoUrl()`

### Track 2 (Causal)
- `analyzeCausal(eventId)` — POST /api/causal/analyze/{id}
- `fetchCausalGraph(eventId)` — GET /api/causal/{id}

### Track 3 (RAG)
- `ragIngest(eventId)` — POST /api/rag/ingest/{id}
- `ragSearch(query, limit)` — POST /api/rag/search

### Track 4 (Synthesis)
- `generateSitrep(eventId)` — POST /api/synthesis/{id}
- `fetchSitrep(eventId)` — GET /api/synthesis/{id}

### System
- `fetchConfig()` — GET /api/config
- `fetchLogs()` — GET /api/logs

---

## Key Components

### `VideoAnnotator.tsx`
Renders event video clips with bounding box overlays and object labels. Used on event detail pages.

### `CausalPanel.tsx` / `CausalGraph.tsx`
Displays PCMCI+ causal discovery results — target vehicles, driver links with lag/strength, and a graph visualization.

### `Sidebar.tsx`
Navigation sidebar with links to all major sections.

---

## Configuration

- `NEXT_PUBLIC_API_URL` env var (defaults to `http://localhost:8000`).
- Backend CORS is fully open (dev mode).

---

## Caveats

- **Next.js 16** has breaking changes from earlier versions. Consult `frontend/node_modules/next/dist/docs/` before writing frontend code.
- The privacy/audit API functions (`fetchPrivacyStatus`, `fetchAuditLog`, etc.) are **unwired** — no backend implementation exists yet.
- `frontend/AGENTS.md` warns about React 19 / App Router differences from prior knowledge.
