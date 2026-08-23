"""
FastAPI router for Track 2 — Full Causal Engine endpoints.

Endpoints:
  POST /api/causal/analyze/{event_id}   — run full causal analysis
  GET  /api/causal/graph/{event_id}     — return persisted causal graph
  GET  /api/causal/methods              — list available causal methods
  GET  /api/causal/report/{event_id}    — return causal report summary
  GET  /api/causal/timeline/{event_id}  — return temporal event timeline
"""
import json

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.pipeline.causal import get_causal_engine

router = APIRouter(prefix="/api/causal", tags=["Causal (Track 2)"])


@router.post("/analyze/{event_id}")
async def analyze_event(event_id: str):
    """
    Run the full causal analysis pipeline:
      - Kinematic enrichment
      - PCMCI+ / LiNGAM / GES / Causal Forest
      - Multi-method consensus
      - Temporal event extraction
      - Event-level causal chain
      - Event onset localization
      - Human-readable explanation
    Returns the full causal graph JSON.
    """
    try:
        return get_causal_engine().analyze_event(event_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/graph/{event_id}")
async def get_causal_graph(event_id: str):
    """Return the persisted causal graph for an event (run POST /analyze first)."""
    path = settings.paths.dataset_dir / event_id / "causal_graph.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No causal graph for {event_id}; POST /api/causal/analyze/{event_id} first",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/methods")
async def get_methods():
    """List available causal discovery methods and their status."""
    statuses = {}
    try:
        import tigramite
        statuses["PCMCI+"] = {"available": True, "library": "tigramite"}
    except ImportError:
        statuses["PCMCI+"] = {"available": False, "library": "tigramite"}
    try:
        import lingam
        statuses["LiNGAM"] = {"available": True, "library": "lingam"}
    except ImportError:
        statuses["LiNGAM"] = {"available": False, "library": "lingam"}
    try:
        from causallearn.search.ScoreBased.GES import ges  # noqa: F401
        statuses["GES"] = {"available": True, "library": "causal-learn"}
    except ImportError:
        statuses["GES"] = {"available": False, "library": "causal-learn"}
    try:
        from econml.dml import CausalForestDML  # noqa: F401
        statuses["CausalForest"] = {"available": True, "library": "econml"}
    except ImportError:
        statuses["CausalForest"] = {"available": False, "library": "econml"}
    statuses["CycleNet"] = {"available": False, "library": "cyclenet", "reason": "Not installable as pip package"}
    return {"methods": statuses}


@router.get("/report/{event_id}")
async def get_causal_report(event_id: str):
    """Return the causal report summary for an event."""
    path = settings.paths.dataset_dir / event_id / "causal_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No causal report for {event_id}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/timeline/{event_id}")
async def get_event_timeline(event_id: str):
    """Return the temporal event timeline for an event."""
    path = settings.paths.dataset_dir / event_id / "event_timeline.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No event timeline for {event_id}")
    return json.loads(path.read_text(encoding="utf-8"))
