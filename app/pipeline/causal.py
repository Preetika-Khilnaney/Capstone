"""
causal.py — Track 2: Full Causal Engine orchestrator.

Orchestrates the complete pipeline:
  1. Load kinematic CSV
  2. Enrich kinematics (vx, vy, ax, ay, acceleration, heading)
  3. Build pairwise interaction features
  4. Build PCMCI+ / LiNGAM / GES / CausalForest variable matrices
  5. Run all available statistical causal discovery methods
  6. Build multi-method consensus graph
  7. Extract temporal events (per-entity + pairwise)
  8. Build event-level causal chain
  9. Localize event onset (earliest physically-supported frame)
  10. Construct final causal graph JSON
  11. Generate human-readable explanation
  12. Persist all outputs

API surface (unchanged):
  get_causal_engine().analyze_event(event_id) -> dict
  get_causal_engine().get_graph(event_id) -> dict
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import settings
from app.database import get_event

# Sub-modules
from app.pipeline.causal_kinematics import enrich_kinematics, compute_pairwise_features
from app.pipeline.causal_discovery import (
    run_pcmci_plus,
    run_lingam,
    run_ges,
    run_causal_forest,
    build_consensus,
)
from app.pipeline.causal_events import (
    extract_entity_events,
    extract_pairwise_events,
    build_event_causal_chain,
    EventOnsetLocalizer,
    generate_explanation,
    TemporalEvent,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_event_df(event_id: str) -> Optional[pd.DataFrame]:
    """Load the event's causal CSV from DB-registered path or default location."""
    ev = get_event(event_id)
    csv_path: Optional[Path] = None
    if ev and ev.get("Causal_CSV_Path"):
        p = Path(ev["Causal_CSV_Path"])
        csv_path = p if p.exists() else None
    if csv_path is None:
        cand = settings.paths.dataset_dir / event_id / f"{event_id}_causal_data.csv"
        csv_path = cand if cand.exists() else None
    if csv_path is None:
        return None
    return pd.read_csv(csv_path)


def _fps_from_df(df: pd.DataFrame) -> float:
    """Estimate FPS from timestamp column; fallback to config."""
    if "Timestamp" in df.columns:
        ts = pd.to_numeric(df["Timestamp"], errors="coerce").dropna().sort_values()
    elif "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().sort_values()
    else:
        return float(settings.video.target_fps)
    if len(ts) < 2:
        return float(settings.video.target_fps)
    # Median frame-to-frame interval (per object is tricky; use all rows sorted)
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return float(settings.video.target_fps)
    median_dt = float(diffs.median())
    return round(1.0 / median_dt, 2) if median_dt > 1e-6 else float(settings.video.target_fps)


def _build_causal_variables(
    entity_series: Dict[str, pd.DataFrame],
    pair_series: Dict[Tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge entity kinematics + pairwise features into a single wide DataFrame
    aligned on the UNION of frame_ids (NaN where an entity is absent), suitable
    for PCMCI+ / LiNGAM / GES.

    Column naming:
      {oid}_speed, {oid}_acceleration, {oid}_vx, {oid}_vy, {oid}_heading
      {oid_a}_{oid_b}_distance, ..._closing_speed, ..._relative_speed

    Only entities with enough valid frames are included.
    """
    # Find union of all frame IDs across all entities and pairs
    all_frames: set = set()
    for oid, sdf in entity_series.items():
        all_frames.update(sdf["frame_id"].astype(int).tolist())

    if not all_frames:
        return pd.DataFrame()

    all_frames_sorted = sorted(all_frames)
    base = pd.DataFrame({"frame_id": all_frames_sorted})
    new_cols = {}

    # Entity columns — only include entities with >= 10 valid speed frames
    for oid, sdf in entity_series.items():
        speed_valid = sdf["speed"].notna().sum()
        if speed_valid < 10:
            continue  # skip very sparse tracks
        sdf_idx = sdf.set_index("frame_id")
        safe_oid = str(oid).replace("-", "_").replace(" ", "_")
        for col in ["speed", "acceleration", "vx", "vy", "heading", "ax", "ay"]:
            if col in sdf_idx.columns:
                new_cols[f"{safe_oid}_{col}"] = base["frame_id"].map(
                    sdf_idx[col].to_dict()
                ).astype(float)

    # Pairwise columns
    for (id_a, id_b), pf in pair_series.items():
        if len(pf) < 10:
            continue
        pf_idx = pf.set_index("frame_id")
        label = f"{id_a}_{id_b}".replace("-", "_").replace(" ", "_")
        for col in ["distance", "closing_speed", "relative_speed", "relative_acceleration", "heading_difference"]:
            if col in pf_idx.columns:
                new_cols[f"{label}_{col}"] = base["frame_id"].map(
                    pf_idx[col].to_dict()
                ).astype(float)

    if new_cols:
        base = pd.concat([base, pd.DataFrame(new_cols)], axis=1)

    return base.sort_values("frame_id").reset_index(drop=True)


def _causal_forest_treatments(
    entity_series: Dict[str, pd.DataFrame],
    pair_series: Dict[Tuple[str, str], pd.DataFrame],
    wide_df: pd.DataFrame,
) -> List[Dict]:
    """
    Identify valid treatment/outcome formulations for Causal Forest.
    Runs the forest and returns a list of result dicts.
    """
    from app.pipeline.causal_discovery import run_causal_forest

    results = []

    # For each pair: treat "sudden braking of A" as treatment, "braking of B" as outcome
    for (id_a, id_b), pf in pair_series.items():
        sa = entity_series.get(id_a)
        sb = entity_series.get(id_b)
        if sa is None or sb is None:
            continue

        safe_a = str(id_a).replace("-", "_").replace(" ", "_")
        safe_b = str(id_b).replace("-", "_").replace(" ", "_")
        pair_label = f"{safe_a}_{safe_b}"

        treat_col = f"{safe_a}_acceleration"  # large deceleration → sudden braking
        outcome_col = f"{safe_b}_speed"
        cov_cols = [
            f"{pair_label}_distance",
            f"{pair_label}_closing_speed",
            f"{safe_a}_speed",
            f"{safe_b}_acceleration",
        ]
        cov_cols = [c for c in cov_cols if c in wide_df.columns]

        if treat_col not in wide_df.columns or outcome_col not in wide_df.columns:
            continue

        # Create binary treatment: A deceleration threshold
        tmp = wide_df.copy()
        tmp[treat_col] = tmp[treat_col].apply(lambda x: float(x) if pd.notna(x) else np.nan)
        # Treatment = strong deceleration of A
        acc_vals = tmp[treat_col].dropna()
        if len(acc_vals) < 30:
            continue
        thresh = float(np.percentile(acc_vals, 25))  # bottom quartile = hardest braking
        tmp["_treatment"] = (tmp[treat_col] < thresh).astype(float)

        res = run_causal_forest(
            tmp,
            treatment_col="_treatment",
            outcome_col=outcome_col,
            covariate_cols=cov_cols,
            treatment_threshold=0.5,
        )
        res["treatment_description"] = f"Hard braking of {id_a}"
        res["outcome_description"] = f"Speed of {id_b}"
        results.append(res)

    return results


def _select_stat_variables(
    all_var_names: List[str],
    entity_series: Dict[str, pd.DataFrame],
    pair_series: Dict[Tuple[str, str], pd.DataFrame],
    max_vars: int = 60,
) -> List[str]:
    """
    Select the most causally relevant subset of variables for statistical methods.

    Priority:
    1. Speed + acceleration for entities with significant speed drops (events)
    2. Pairwise distance + closing_speed for pairs with min_dist < 5m
    3. Remaining speed variables up to cap
    """
    import numpy as np

    selected: List[str] = []
    used_oids: set = set()

    # Score entities by speed drop magnitude
    entity_scores: List[Tuple[float, str]] = []
    for oid, sdf in entity_series.items():
        spd = sdf["speed"].to_numpy(dtype=float)
        valid = spd[np.isfinite(spd)]
        if len(valid) < 5:
            continue
        drop = float((np.maximum.accumulate(valid) - valid).max())
        entity_scores.append((drop, oid))
    entity_scores.sort(reverse=True)

    # Score pairs by minimum distance
    pair_scores: List[Tuple[float, Tuple[str, str]]] = []
    for pair, pf in pair_series.items():
        dist = pf["distance"].to_numpy(dtype=float)
        min_d = float(np.nanmin(dist)) if np.any(np.isfinite(dist)) else np.inf
        pair_scores.append((min_d, pair))
    pair_scores.sort()  # ascending distance = closest pair first

    # 1. Speed + acceleration for top entities
    for _, oid in entity_scores:
        safe = oid.replace("-", "_").replace(" ", "_")
        cols = [f"{safe}_speed", f"{safe}_acceleration"]
        for c in cols:
            if c in all_var_names and c not in selected:
                selected.append(c)
        used_oids.add(oid)
        if len(selected) >= max_vars // 2:
            break

    # 2. Pairwise features for close pairs
    for min_d, (pa, pb) in pair_scores:
        if min_d > 10.0:
            break  # only really close pairs
        label = f"{pa}_{pb}".replace("-", "_").replace(" ", "_")
        cols = [f"{label}_distance", f"{label}_closing_speed", f"{label}_relative_speed"]
        for c in cols:
            if c in all_var_names and c not in selected:
                selected.append(c)
        if len(selected) >= max_vars:
            break

    # 3. Fill remaining with speed vars
    for _, oid in entity_scores:
        safe = oid.replace("-", "_").replace(" ", "_")
        c = f"{safe}_speed"
        if c in all_var_names and c not in selected:
            selected.append(c)
        if len(selected) >= max_vars:
            break

    return selected[:max_vars]


# ─────────────────────────────────────────────────────────────────────────────
# CausalEngine
# ─────────────────────────────────────────────────────────────────────────────

class CausalEngine:
    """Full Track 2 Causal Engine with multi-method statistical discovery,
    temporal event extraction, event onset localization, and explanation."""

    def analyze_event(self, event_id: str) -> Dict[str, Any]:
        """
        Run the complete causal analysis pipeline for event_id.
        Returns a rich causal graph dict and persists it to disk.
        """
        df_raw = _load_event_df(event_id)
        if df_raw is None or df_raw.empty:
            return {"status": "error", "message": f"No causal CSV for {event_id}"}

        fps = _fps_from_df(df_raw)
        logger.info("Causal engine: event=%s  FPS=%.1f  rows=%d", event_id, fps, len(df_raw))

        # ── 1. Enrich kinematics ─────────────────────────────────────────────
        try:
            entity_series = enrich_kinematics(df_raw, smooth_window=5)
        except Exception as exc:
            return {"status": "error", "message": f"Kinematic enrichment failed: {exc}"}

        if len(entity_series) < 1:
            return {"status": "no_target", "message": "No tracked entities found in kinematic data."}

        # ── 2. Pairwise features ──────────────────────────────────────────────
        pair_series = compute_pairwise_features(entity_series)

        # ── 3. Build wide causal variable matrix ─────────────────────────────
        wide_df = _build_causal_variables(entity_series, pair_series)
        if wide_df.empty:
            return {
                "status": "insufficient",
                "message": "No overlapping frames between tracked entities.",
            }

        causal_var_names = [c for c in wide_df.columns if c != "frame_id"]

        # ── 4. Select most relevant variables for statistical methods ─────────
        # With 58 entities we can have 2000+ columns; PCMCI+ becomes too slow.
        # Select: speed + acceleration for each entity, plus the closest-pair
        # pairwise features. Cap at 60 variables.
        MAX_STAT_VARS = 60
        stat_var_names = _select_stat_variables(
            causal_var_names, entity_series, pair_series, max_vars=MAX_STAT_VARS
        )
        stat_df = wide_df[["frame_id"] + stat_var_names].copy()

        # ── 5. Statistical causal discovery ──────────────────────────────────
        tau_max_cfg = settings.causal.tau_max

        pcmci_res = run_pcmci_plus(stat_df, stat_var_names, tau_max=tau_max_cfg, fps=fps)
        lingam_res = run_lingam(stat_df, stat_var_names)
        ges_res = run_ges(stat_df, stat_var_names)

        # Causal Forest (per relevant pair)
        forest_results = _causal_forest_treatments(entity_series, pair_series, wide_df)
        # Merge all forest edges into a single result
        all_forest_edges = []
        forest_status = "ok" if forest_results else "skipped"
        for fr in forest_results:
            if fr.get("status") == "ok":
                all_forest_edges.extend(fr.get("edges", []))
        forest_combined = {"status": forest_status, "edges": all_forest_edges, "details": forest_results}

        # CycleNet: not available
        cyclenet_res = {"status": "unavailable", "reason": "CycleNet not installable as a pip package", "edges": []}

        method_results = {
            "PCMCI+": pcmci_res,
            "LiNGAM": lingam_res,
            "GES": ges_res,
            "CausalForest": forest_combined,
            "CycleNet": cyclenet_res,
        }

        # ── 5. Consensus ──────────────────────────────────────────────────────
        consensus_edges = build_consensus(method_results, min_support_ratio=0.25)

        # ── 6. Temporal event extraction ──────────────────────────────────────
        all_events: List[TemporalEvent] = []
        for oid, sdf in entity_series.items():
            all_events.extend(extract_entity_events(oid, sdf))
        for (id_a, id_b), pf in pair_series.items():
            all_events.extend(extract_pairwise_events(id_a, id_b, pf))

        # ── 7. Event-level causal chain ───────────────────────────────────────
        event_edges = build_event_causal_chain(all_events, entity_series, pair_series, fps=fps)

        # ── 8. Event onset localization ───────────────────────────────────────
        localizer = EventOnsetLocalizer()
        onset_map = localizer.localize(all_events, entity_series, pair_series, fps=fps)

        # ── 9. Build final graph nodes & edges ────────────────────────────────
        nodes = self._build_nodes(entity_series, all_events, onset_map)
        statistical_edges = consensus_edges
        temporal_edges = event_edges

        # ── 10. Explanation ───────────────────────────────────────────────────
        explanation = generate_explanation(all_events, event_edges, onset_map, entity_series)

        # ── 11. Compute overall confidence ───────────────────────────────────
        edge_confs = [e["final_confidence"] for e in consensus_edges]
        event_confs = [e["confidence"] for e in event_edges]
        all_confs = edge_confs + event_confs
        overall_conf = round(float(np.mean(all_confs)) if all_confs else 0.0, 4)

        # ── 12. Method status report ──────────────────────────────────────────
        available = [m for m, r in method_results.items() if r.get("status") == "ok"]
        failed = [m for m, r in method_results.items() if r.get("status") == "error"]
        skipped = [m for m, r in method_results.items() if r.get("status") in ("unavailable", "skipped", "insufficient")]

        # Primary event summary (largest collision/contact if exists)
        primary_evt = self._find_primary_event(all_events, onset_map)

        result = {
            "status": "ok",
            "event_id": event_id,

            "fps": fps,
            "n_entities": len(entity_series),
            "n_pairs": len(pair_series),
            "n_causal_vars": len(causal_var_names),
            "n_timesteps": len(wide_df),

            "primary_event": primary_evt,

            "entities": [
                {
                    "object_id": oid,
                    "class": str(sdf["class"].dropna().iloc[0]) if "class" in sdf.columns and not sdf["class"].dropna().empty else None,
                    "n_frames": int(sdf["frame_id"].nunique()),
                    "speed_range_mps": [
                        round(float(sdf["speed"].min(skipna=True)), 3),
                        round(float(sdf["speed"].max(skipna=True)), 3),
                    ] if "speed" in sdf.columns else None,
                }
                for oid, sdf in entity_series.items()
            ],

            "nodes": nodes,

            "statistical_causal_edges": statistical_edges,
            "temporal_event_edges": temporal_edges,

            "method_results": {
                m: {
                    "status": r.get("status"),
                    "n_edges": len(r.get("edges", [])),
                    "reason": r.get("reason"),
                }
                for m, r in method_results.items()
            },
            "available_methods": available,
            "failed_methods": failed,
            "skipped_methods": skipped,

            "consensus_edges": consensus_edges,

            "temporal_events": [
                {
                    "event_id": e.event_id,
                    "object_ids": e.object_ids,
                    "event_type": e.event_type,
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "start_timestamp": e.start_timestamp,
                    "end_timestamp": e.end_timestamp,
                    "features": e.features,
                    "confidence": e.confidence,
                    "onset": onset_map.get(e.event_id, {}),
                }
                for e in all_events
            ],

            "explanation": explanation,
            "confidence": overall_conf,

            "note": (
                "Multi-method causal engine (PCMCI+, LiNGAM, GES, CausalForest). "
                "Statistical results are ranked hypotheses, not causal proof."
            ),
        }

        self._persist(event_id, result, method_results, consensus_edges)
        return result

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_primary_event(
        self,
        all_events: List[TemporalEvent],
        onset_map: Dict[str, Dict],
    ) -> Optional[Dict]:
        """Return the most significant collision/contact event with onset info."""
        priority_types = ["COLLISION", "CONTACT", "NEAR_COLLISION", "FALL", "SUDDEN_BRAKING"]
        for ev_type in priority_types:
            candidates = [e for e in all_events if e.event_type == ev_type]
            if candidates:
                evt = max(candidates, key=lambda e: e.confidence)
                onset_info = onset_map.get(evt.event_id, {})
                return {
                    "event_id": evt.event_id,
                    "event_type": evt.event_type,
                    "object_ids": evt.object_ids,
                    "event_onset": {
                        "frame": onset_info.get("onset_frame", evt.start_frame),
                        "timestamp": onset_info.get("onset_timestamp", evt.start_timestamp),
                        "reason": onset_info.get("onset_reason", "event_start"),
                    },
                    "confirmation": {
                        "frame": onset_info.get("confirmation", {}).get("confirmation_frame", evt.end_frame),
                        "timestamp": onset_info.get("confirmation", {}).get("confirmation_timestamp", evt.end_timestamp),
                    },
                    "confidence": evt.confidence,
                }
        return None

    def _build_nodes(
        self,
        entity_series: Dict[str, pd.DataFrame],
        all_events: List[TemporalEvent],
        onset_map: Dict[str, Dict],
    ) -> List[Dict]:
        """Build node list for the causal graph."""
        nodes = []

        # Entity nodes
        for oid, sdf in entity_series.items():
            cls = str(sdf["class"].dropna().iloc[0]) if "class" in sdf.columns and not sdf["class"].dropna().empty else "vehicle"
            first_ts = float(sdf["timestamp"].min(skipna=True)) if "timestamp" in sdf.columns else None
            nodes.append({
                "id": f"entity_{oid}",
                "label": f"Vehicle {oid}",
                "type": "object",
                "object_ids": [oid],
                "class": cls,
                "timestamp": first_ts,
                "confidence": 1.0,
            })

        # Event nodes
        for evt in all_events:
            onset_info = onset_map.get(evt.event_id, {})
            onset_ts = onset_info.get("onset_timestamp", evt.start_timestamp)
            onset_frame = onset_info.get("onset_frame", evt.start_frame)
            nodes.append({
                "id": evt.event_id,
                "label": evt.event_type.replace("_", " ").title(),
                "type": "event",
                "object_ids": evt.object_ids,
                "event_type": evt.event_type,
                "timestamp": onset_ts,
                "frame": onset_frame,
                "confidence": evt.confidence,
                "features": evt.features,
            })

        return nodes

    @staticmethod
    def _persist(
        event_id: str,
        result: Dict,
        method_results: Dict,
        consensus_edges: List[Dict],
    ) -> None:
        out_dir = settings.paths.dataset_dir / event_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # causal_graph.json
        (out_dir / "causal_graph.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )

        # method_results.json (full, untruncated)
        (out_dir / "method_results.json").write_text(
            json.dumps(method_results, indent=2, default=str), encoding="utf-8"
        )

        # consensus_edges.csv
        if consensus_edges:
            pd.DataFrame([
                {
                    "source": e["source"],
                    "target": e["target"],
                    "relationship": e["relationship"],
                    "lag_frames": e["lag_frames"],
                    "lag_seconds": e["lag_seconds"],
                    "support_ratio": e["support_ratio"],
                    "final_confidence": e["final_confidence"],
                    "p_value": e.get("p_value"),
                }
                for e in consensus_edges
            ]).to_csv(out_dir / "consensus_edges.csv", index=False)

        # event_timeline.json
        (out_dir / "event_timeline.json").write_text(
            json.dumps(result.get("temporal_events", []), indent=2, default=str),
            encoding="utf-8",
        )

        # causal_report.json (summary)
        report = {
            "event_id": event_id,
            "status": result["status"],
            "fps": result["fps"],
            "n_entities": result["n_entities"],
            "n_pairs": result["n_pairs"],
            "available_methods": result["available_methods"],
            "failed_methods": result["failed_methods"],
            "skipped_methods": result["skipped_methods"],
            "n_consensus_edges": len(consensus_edges),
            "n_temporal_events": len(result.get("temporal_events", [])),
            "primary_event": result.get("primary_event"),
            "explanation": result.get("explanation"),
            "confidence": result.get("confidence"),
        }
        (out_dir / "causal_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

        logger.info(
            "Causal engine persisted: %d consensus edges, %d events, conf=%.3f",
            len(consensus_edges),
            len(result.get("temporal_events", [])),
            result.get("confidence", 0.0),
        )

    def get_graph(self, event_id: str) -> Optional[Dict]:
        """Load persisted causal graph."""
        path = settings.paths.dataset_dir / event_id / "causal_graph.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[CausalEngine] = None


def get_causal_engine() -> CausalEngine:
    global _engine
    if _engine is None:
        _engine = CausalEngine()
    return _engine
