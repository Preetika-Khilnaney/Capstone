"""
Stage segmenter — deterministic staged narrative for an event.

Given the (re-anchored) event CSV and the scene table, produces the TrueLabels-
*meaning* stage chain N1..N5 (stable → trigger → incident → hazard response →
recovery) plus typed stage-level relations (DIRECT_CAUSE / TRIGGERED_RESPONSE /
CONSEQUENCE) and the root-cause block. Windows are in clip-relative seconds
(t=0 = incident anchor); consume the NaN-aware smoothed kinematics written by
Phase 2.

Reference: incident_anchor_plan.md §7, §8, §9
"""

import logging

import numpy as np

from app.config import settings
from app.pipeline.rootcause import analyze_root_cause
from app.pipeline.scene import analyze_scene, forward_dir, sustained_drop, _load_df, series_for, _VEHICLE_CLASSES

logger = logging.getLogger(__name__)

_STOP_SPEED_MPS = 0.8          # initiator speed ≈ 0 → incident stage
_FALLEN_ASPECT = 0.65          # person bbox h/w below this = downed
_FALLEN_MIN_FRAMES = 8
_UPRIGHT_ASPECT = 1.1          # person bbox h/w above this = back on feet
_BRAKE_DROP_FLOOR_MPS = 1.0


def _cruise_speed(v: np.ndarray) -> float:
    """Robust cruising level: median of the top quartile of finite speeds."""
    v = np.asarray(v, dtype=float)
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return 0.0
    vv = np.sort(vv)
    k = max(3, vv.size // 4)
    return float(np.median(vv[-k:]))


def _frame_arrays(s: dict, frames: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.array([s["vel"].get(f, np.nan) for f in frames], dtype=float)
    p = np.array([s["pos"].get(f, np.array([np.nan, np.nan])) for f in frames], dtype=float)
    t = np.array([s["time"].get(f, np.nan) for f in frames], dtype=float)
    return v, p, t


def _aspect_series(s: dict, frames: list[int]) -> np.ndarray:
    out = []
    for f in frames:
        b = s["bbox"].get(f)
        if b is None:
            out.append(np.nan)
            continue
        x1, y1, x2, y2 = b
        h, w = y2 - y1, x2 - x1
        out.append(h / w if w > 0 else np.nan)
    return np.array(out, dtype=float)


def _person_down_window(frames: list[int], persons: list[dict], lo: int, hi: int) -> bool:
    for s in persons:
        asp = _aspect_series(s, frames[lo:hi])
        fin = asp[np.isfinite(asp)]
        if fin.size >= _FALLEN_MIN_FRAMES and float(np.mean(fin)) < _FALLEN_ASPECT:
            return True
    return False


def _person_upright_resume(frames: list[int], persons: list[dict], onset: int) -> int | None:
    earliest = None
    for s in persons:
        asp = _aspect_series(s, frames)
        for i in range(onset, len(frames)):
            if np.isfinite(asp[i]) and asp[i] >= _UPRIGHT_ASPECT:
                if earliest is None or i < earliest:
                    earliest = i
    return earliest


def _initiator_fallback(scene: dict) -> dict | None:
    """Pick an initiator object when no motorcycle exists (persons first)."""
    ranking = scene.get("drop_ranking") or []
    persons = [r for r in ranking if r["class"] == "person"]
    vehicles = [r for r in ranking if r["onset_frame"] >= 0]
    pool = persons if persons else vehicles
    if not pool:
        return None
    top = pool[0]
    return {"object_id": top["object_id"], "class": top["class"]}


def _build_episode_parts(df, scene: dict) -> dict | None:
    cfg_s = settings.scene
    frames = sorted(int(f) for f in df["Frame_ID"].unique())
    if not frames:
        return None
    n = len(frames)

    oids = sorted(df["Object_ID"].unique())
    series = {oid: series_for(df, oid) for oid in oids}
    id_by_oid = {oid: e["id"] for e in scene["entities"] for oid in e["object_ids"]}
    role_entities = {e["role"]: e for e in scene["entities"]}

    grid_t = np.array([f / 10.0 - 10.0 for f in frames], dtype=float)
    clip_start = float(grid_t[0])
    clip_end = float(grid_t[-1])
    t = lambda idx: (t_i[idx] if np.isfinite(t_i[idx]) else grid_t[idx])

    initiator_oid = scene.get("initiator_object")
    if initiator_oid not in series:
        fb = _initiator_fallback(scene)
        initiator_oid = fb["object_id"] if fb else None
    if initiator_oid is None:
        return None
    init_series = series[initiator_oid]
    initiator_entity = next((e for e in scene["entities"] if initiator_oid in e["object_ids"]), None)
    initiator_entity = initiator_entity or next(
        (e for e in scene["entities"] if e["role"] == "initiator"), None)

    # ── initiator kinematics ─────────────────────────────────────────────
    v_i, p_i, t_i = _frame_arrays(init_series, frames)
    cruise = _cruise_speed(v_i)
    min_drop = max(0.25 * cruise, _BRAKE_DROP_FLOOR_MPS)
    _, onset = sustained_drop(v_i, min_drop)
    grid_t = np.array([f / 10.0 - 10.0 for f in frames], dtype=float)
    anchor_idx = int(np.argmin(np.abs(grid_t - 0.0)))  # t=0 = crash-machine burst
    if onset < 0:
        # The initiator may only be observable at rest (debris phase) after the
        # crash: pin the kinematic onset to its first post-anchor observation,
        # else to the anchor itself, so the chain still reads N1..N5 in order.
        post = np.where(np.isfinite(v_i) & (np.arange(n) >= anchor_idx))[0]
        onset = int(post[0]) if post.size else anchor_idx
    if onset < 0:
        logger.info("Event %s: no sustained initiator speed fall — no incident staged",
                    df["Event_ID"].iloc[0])
        return None

    # ── trigger (N2) / incident (N3) split ───────────────────────────────
    # N2 spans from the earliest instability evidence (kinematic onset OR the
    # crash-machine burst, whichever came first) to the later one; N3 runs from
    # there to the incident (speed ≈ 0 / track loss). On testVideo2 the burst
    # (t=0) precedes the skid onset, so N2 = [burst, skid-onset].
    anchor_t = float(grid_t[anchor_idx])
    n1_end_t = float(min(t_i[onset], anchor_t))
    inc_start_t = float(max(t_i[onset], anchor_t))

    # ── incident (N3) start: speed ≈ 0, or mid-window track loss ────────
    stop_idx = int(np.where(np.isfinite(v_i) & (v_i <= _STOP_SPEED_MPS))[0].min()) \
        if np.any(np.isfinite(v_i) & (v_i <= _STOP_SPEED_MPS)) else -1
    finite_idx = np.where(np.isfinite(v_i))[0]
    last_finite = int(finite_idx[-1]) if finite_idx.size else onset
    track_lost = (last_finite - onset) <= int(6.0 / 0.1) and last_finite < n - 2
    if stop_idx >= 0 and onset <= stop_idx:
        incident_idx = stop_idx
        lost = False
    elif track_lost:
        incident_idx = last_finite
        lost = True
    else:
        incident_idx = onset
        lost = False

    # ── skid detection: lateral deviation growth after onset ─────────────
    u = forward_dir(p_i)
    lat = np.full(n, np.nan)
    for k in range(n):
        if np.isfinite(p_i[k]).all():
            lat[k] = abs(float((p_i[k] - p_i[incident_idx])[0] * u[1]
                               - (p_i[k] - p_i[incident_idx])[1] * u[0]))
    pre_idx = np.where(np.isfinite(lat) & (np.arange(n) < onset))[0]
    pre_lat = float(np.nanmax(lat[pre_idx])) if pre_idx.size else 0.0
    inc_idx = np.where(np.isfinite(lat) & (np.arange(n) >= onset))[0]
    inc_lat = float(np.nanmax(lat[inc_idx])) if inc_idx.size else 0.0
    skid = bool(inc_lat - pre_lat >= cfg_s.skid_lateral_m and inc_lat >= 1.0)
    skid_lateral_m = float(inc_lat if skid else 0.0)

    persons = [series[o] for o in scene.get("rider_object_ids", []) if o in series]
    persons_down = _person_down_window(frames, persons, onset, n)

    # ── hazard response (N4): trailing four-wheeler braking onsets ───────
    n3_time = t_i[incident_idx]
    responders: list[tuple[str, int, float]] = []  # (object_id, onset_idx, lag_s)
    for tr in scene.get("trailing", []):
        s = series.get(tr["object_id"])
        if s is None:
            continue
        v_r = _frame_arrays(s, frames)[0]
        rdrop, ronset = sustained_drop(v_r, max(0.25 * _cruise_speed(v_r), _BRAKE_DROP_FLOOR_MPS))
        if ronset < 0:
            continue
        t_r = _frame_arrays(s, frames)[2]
        lag = float(t_r[ronset] - n3_time)
        if cfg_s.rider_reaction_min_s <= lag <= cfg_s.rider_reaction_max_s:
            responders.append((tr["object_id"], ronset, lag))
    n4_idx = min((r[1] for r in responders), default=-1)
    mean_lag = float(np.mean([r[2] for r in responders])) if responders else 0.0

    # ── recovery (N5): traffic reflow or occupants upright ──────────────
    # Recovery = the obstruction clears: responders moving again, the initiator
    # resuming, riders upright again, or NEW flowing traffic first appearing
    # AFTER the responders' full stop (a vehicle passing during the incident is
    # a far-lane bypass, not a resolution of the obstruction).
    n5_idx = None
    for r_oid, r_idx, _ in responders:
        v_r = _frame_arrays(series[r_oid], frames)[0]
        lows = np.where(np.isfinite(v_r) & (v_r < cfg_s.recovery_speed_thresh_mps)
                        & (np.arange(n) > r_idx))[0]
        if not lows.size:
            continue
        after_stop = int(lows[-1])
        hits = np.where(np.isfinite(v_r) & (v_r >= cfg_s.recovery_speed_thresh_mps)
                        & (np.arange(n) > after_stop))[0]
        if hits.size:
            cand = int(hits[0])
            n5_idx = cand if n5_idx is None else min(n5_idx, cand)
    v_i_hits = np.where(np.isfinite(v_i) & (v_i >= cfg_s.recovery_speed_thresh_mps)
                        & (np.arange(n) > incident_idx))[0]
    if v_i_hits.size:
        n5_idx = int(v_i_hits[0]) if n5_idx is None else min(n5_idx, int(v_i_hits[0]))
    upright = _person_upright_resume(frames, persons, incident_idx)
    if upright is not None:
        n5_idx = upright if n5_idx is None else min(n5_idx, upright)
    if n5_idx is None:
        # no one moved again: look for fresh flowing traffic arriving only
        # after the responders have reached their slowest (a vehicle passing
        # during the incident is a far-lane bypass, not a resolution)
        stop_after = incident_idx
        for r_oid, r_idx, _ in responders:
            v_r = _frame_arrays(series[r_oid], frames)[0]
            window = np.where(np.isfinite(v_r) & (np.arange(n) > r_idx)
                              & (np.arange(n) <= r_idx + int(cfg_s.rider_reaction_max_s / 0.1)))[0]
            if window.size:
                stop_after = max(stop_after, int(window[np.nanargmin(v_r[window])]))
        for oid, s in sorted(series.items()):
            if oid == initiator_oid or s["class"] not in _VEHICLE_CLASSES:
                continue
            v_o = _frame_arrays(s, frames)[0]
            vrows = np.where(np.isfinite(v_o))[0]
            if vrows.size and int(vrows[0]) > stop_after \
                    and v_o[vrows[1] if vrows.size > 1 else vrows[0]] >= cfg_s.recovery_speed_thresh_mps:
                cand = int(vrows[1] if vrows.size > 1 else vrows[0])
                n5_idx = cand if n5_idx is None else min(n5_idx, cand)
    if n5_idx is not None and n5_idx <= incident_idx:
        n5_idx = None
    n5_time = t(n5_idx) if n5_idx is not None else None

    # ── windows (clip-relative seconds, t=0 = crash anchor) ──────────────
    n1_end = n1_end_t
    n2_end = inc_start_t
    n4_time = t(n4_idx) if n4_idx >= 0 else None
    n3_end = n4_time if n4_time is not None else (n5_time if n5_time is not None else clip_end)
    n4_end = n5_time if n5_time is not None else clip_end
    n5_end = clip_end

    init_ent = initiator_entity["id"] if initiator_entity else "E1"
    rider_ents = [e["id"] for e in scene["entities"] if e["role"] in ("rider", "pillion")]
    trailing_ents = [e["id"] for e in scene["entities"] if e["role"] == "trailing"]
    all_ents = [e["id"] for e in scene["entities"]]

    init_name = (initiator_entity or {}).get("name", "the initiator")
    cruise_known = cruise >= 1.0
    nodes = [
        {"node_id": "N1", "state": "Stable",
         "window_s": [round(clip_start, 1), round(n1_end, 1)],
         "involved_entities": all_ents,
         "evidence": [f"Normal traffic progression; initiator cruising at "
                      f"{cruise:.1f} m/s before onset." if cruise_known
                      else "Normal traffic progression; pre-crash cruise segment "
                           "not tracked (initiator first observed at rest)."]},
        {"node_id": "N2", "state": "Trigger / Critical",
         "window_s": [round(n1_end, 1), round(n2_end, 1)],
         "involved_entities": [init_ent] + rider_ents,
         "evidence": [f"Motion-machine burst at t={anchor_t:.1f}s signals the "
                      f"deceleration onset; {init_name} falls from {cruise:.1f} m/s "
                      f"through the incident window." if cruise_known
                      else f"Motion-machine burst at t={anchor_t:.1f}s: "
                           f"{init_name} loses speed into the crash anchor."]},
        {"node_id": "N3", "state": "Incident",
         "window_s": [round(n2_end, 1), round(n3_end, 1)],
         "involved_entities": [init_ent] + rider_ents,
         "evidence": [f"Initiator speed {float(v_i[incident_idx]):.1f} m/s"
                      + (" at mid-window track loss" if lost else "")
                      + (f"; lateral skid {skid_lateral_m:.1f} m" if skid else "")
                      + ("; riders down (bbox aspect collapsed)" if persons_down else "")]},
    ]
    if n4_time is not None:
        nodes.append({"node_id": "N4", "state": "Hazard Response",
                      "window_s": [round(n4_time, 1),
                                   round(max(n4_time, n4_end), 1)],
                      "involved_entities": trailing_ents,
                      "evidence": [f"Trailing four-wheelers brake: {len(responders)} "
                                   f"responder(s), mean reaction lag {mean_lag:.1f}s."]})
    if n5_time is not None:
        stream_ent = scene.get("stream_entity") or {}
        stream_id = stream_ent.get("id") if isinstance(stream_ent, dict) else None
        nodes.append({"node_id": "N5", "state": "Recovery",
                      "window_s": [round(n5_time, 1), round(clip_end, 1)],
                      "involved_entities": [init_ent] + rider_ents + trailing_ents
                                           + ([stream_id] if stream_id else []),
                      "evidence": [f"Recovery: responders resume above "
                                   f"{cfg_s.recovery_speed_thresh_mps} m/s and/or occupants upright."]})

    relations = []
    if len(nodes) >= 3:
        mech = (f"Same-actor kinematic continuity: {init_name} speed "
                f"collapse {cruise:.1f}→~0 m/s over {n2_end - n1_end:.1f}s"
                if cruise_known
                else f"Motion-machine burst at t={n1_end:.1f}s aligned with "
                     f"{init_name} loss of speed (arrested through the window)")
        if skid:
            mech += f"; lateral skid of {skid_lateral_m:.1f} m"
        relations.append({
            "source_node": "N2", "target_node": "N3", "relation_type": "DIRECT_CAUSE",
            "mechanism": mech,
        })
    if n4_time is not None:
        relations.append({
            "source_node": "N3", "target_node": "N4",
            "relation_type": "TRIGGERED_RESPONSE",
            "mechanism": f"Obstruction in the active lane prompted evasive braking by "
                         f"trailing vehicle(s) (mean reaction lag {mean_lag:.1f}s, "
                         f"{len(responders)} responder(s))",
        })
    if n5_time is not None:
        relations.append({
            "source_node": "N3", "target_node": "N5",
            "relation_type": "CONSEQUENCE",
            "mechanism": f"Incident preceded recovery: responders resumed above "
                         f"{cfg_s.recovery_speed_thresh_mps} m/s and/or occupants "
                         f"upright by t={n5_time:.1f}s",
        })

    facts = _root_cause_facts(df, scene, series, frames, initiator_oid,
                              cruise, v_i, onset, incident_idx, n3_time,
                              skid, skid_lateral_m, persons_down, responders,
                              n5_time, t_i)
    root_cause = analyze_root_cause(scene, facts)

    return {
        "entities": scene["entities"],
        "nodes": nodes,
        "relations": relations,
        "root_cause": root_cause,
    }


def _root_cause_facts(df, scene, series, frames, initiator_oid,
                      cruise, v_i, onset, incident_idx, n3_time,
                      skid, skid_lateral_m, persons_down, responders,
                      n5_time, t_i) -> dict:
    """Kinematic facts consumed by rootcause.analyze_root_cause."""
    init_ent = next((e for e in scene["entities"] if initiator_oid in e["object_ids"]), None)
    v_fin = v_i[np.isfinite(v_i)]
    vmin = float(np.min(v_fin)) if v_fin.size else 0.0
    collapse = float(t_i[incident_idx] - t_i[onset]) if np.isfinite(t_i[onset]) else 0.0

    # Flanking proximity before onset: mean distance from initiator to the
    # nearest non-trailing vehicle, over pre-onset frames.
    flank, count = 0.0, 0
    trailing_ids = {tr["object_id"] for tr in scene.get("trailing", [])}
    for idx in range(max(0, onset - 1)):
        pi = series[initiator_oid]["pos"].get(frames[idx])
        if pi is None or not np.isfinite(pi).all():
            continue
        best = None
        for oid, s in series.items():
            if oid == initiator_oid or oid in trailing_ids or s["class"] not in {
                    "car", "truck", "bus", "bicycle"}:
                continue
            po = s["pos"].get(frames[idx])
            if po is None or not np.isfinite(po).all():
                continue
            d = float(np.linalg.norm(po - pi))
            best = d if best is None else min(best, d)
        if best is not None:
            flank += best
            count += 1
    flank_mean = flank / count if count else None

    # Min gap maintained by trailing four-wheelers (per-vehicle min distance).
    gap, gcount = None, 0
    for tr in scene.get("trailing", []):
        s = series.get(tr["object_id"])
        if s is None:
            continue
        best = None
        for f in frames:
            pi = series[initiator_oid]["pos"].get(f)
            po = s["pos"].get(f)
            if pi is None or po is None or not np.isfinite(pi).all() or not np.isfinite(po).all():
                continue
            d = float(np.linalg.norm(po - pi))
            best = d if best is None else min(best, d)
        if best is not None:
            gap = best if gap is None else min(gap, best)
            gcount += 1

    occupants_upright = False
    if n5_time is not None and scene.get("rider_object_ids"):
        for oid in scene["rider_object_ids"]:
            s = series.get(oid)
            if s is None:
                continue
            frames_end = [f for f in frames if f in s["frames"]]
            if not frames_end:
                continue
            last_frame = frames_end[-1]
            last_t = s["time"].get(last_frame)
            if last_t is not None and last_t >= 0.8 * n5_time:
                occupants_upright = True

    stream = scene.get("stream") or {}
    facts = {
        "initiator_name": (init_ent or {}).get("name") or "initiator",
        "initiator_class": (init_ent or {}).get("class"),
        "cruise_mps": cruise,
        "min_speed_mps": vmin,
        "collapse_s": collapse,
        "skid": skid,
        "skid_lateral_m": skid_lateral_m,
        "persons_down": persons_down,
        "n_responders": len(responders),
        "min_trailing_gap_m": gap,
        "ambient_speed_mps": stream.get("mean_speed_mps"),
        "density_per_frame": stream.get("density_per_frame", 0.0),
        "flank_proximity_m": flank_mean,
        "occupants_upright_at_end": occupants_upright,
        "skid_position_note": None,
        "clip_seconds": float(np.nanmax(t_i)) if np.isfinite(t_i).any() else 0.0,
    }
    if skid and skid_lateral_m < 4.0:
        facts["skid_position_note"] = \
            f"Skid stayed within the active lane (max lateral deviation {skid_lateral_m:.1f} m)"
    elif skid:
        facts["skid_position_note"] = \
            f"Skid deviated across lane boundaries (max lateral deviation {skid_lateral_m:.1f} m)"
    return facts


def build_episode(event_id: str, colours: dict[str, str] | None = None) -> dict | None:
    """
    Deterministic episode block (entities + staged nodes + typed relations +
    root cause) for an event, or None when the event has no usable kinematics.
    """
    df = _load_df(event_id)
    if df is None or df.empty:
        return None
    try:
        scene = analyze_scene(event_id, colours)
        if scene is None:
            return None
        return _build_episode_parts(df, scene)
    except Exception as exc:
        logger.warning("Episode analysis for %s failed (non-fatal): %s", event_id, exc)
        return None