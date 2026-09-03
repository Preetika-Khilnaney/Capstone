# Track 3: RAG (Retrieval-Augmented Generation) — How It Works

## Overview

The RAG pipeline provides **semantic image search** over entity crops using **SigLIP** embeddings stored in **LanceDB**. It enables natural-language queries like "find the red truck" to retrieve matching vehicle crops.

**Source:** `app/pipeline/rag.py`  
**Routes:** `app/routes/rag_routes.py`

---

## Pipeline

### Ingestion (`ingest_event_crops`)
1. Find all `{event_id}_*_crop.jpg` in `dataset/{event_id}/entity_crops/`.
2. **Idempotent**: delete existing rows for the event before inserting (no duplicates on re-ingest).
3. Embed all crops via SigLIP (`embed_images` → L2-normalized 768-d vectors).
4. Insert records into LanceDB table `entity_crops` with schema:
   - `vector` (float32[768])
   - `event_id` (string)
   - `object_id` (string)
   - `image_path` (string)

### Search (`search_crops`)
1. Embed the text query via SigLIP (`embed_text`).
2. Over-fetch 5× limit (for deduplication).
3. **Deduplicate**: collapse exact `(event_id, object_id)` repeats and near-identical embeddings (cosine > 0.95).
4. Return up to `limit` distinct entities sorted by distance.

### Zero-shot Classification (`zero_shot_batch`)
- Used by Track 4 (synthesis) to derive text attributes (colour) for entities.
- Embeds images and label prompts, returns best-matching label via cosine similarity.

### Live Feed Indexing (`index_vehicles`)
- Indexes continuously-observed vehicles from `FeedMonitor` into LanceDB under a `FEED_{video_id}` bucket.
- Every vehicle is indexed, not just event vehicles — the **hybrid** design.

---

## Key Design Decisions

- **Lazy initialization.** Model weights load on first use, not at import — keeps server startup fast.
- **SigLIP on PyTorch** (not OpenVINO). Running on CPU or CUDA depending on availability.
- **De-duplication at search time.** Handles both repeated ingests and fragmented track IDs for the same physical entity.
- **No auto-ingestion.** The pipeline does NOT auto-ingest into RAG — ingestion is a separate explicit call per event.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/rag/ingest/{event_id}` | Embed and index an event's crops |
| `POST` | `/api/rag/search` | Semantic search with natural language |

---

## Storage

- **LanceDB** at `dataset/lancedb/`, table `entity_crops`.
- Schema: `vector` (768-d float32), `event_id`, `object_id`, `image_path`.
- Event crops keyed by `event_id=EVT_...`; live-feed vehicles by `event_id=FEED_...`.
