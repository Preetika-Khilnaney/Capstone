"""
Scene semantics — deterministic entity/role assignment for an event.

Builds the TrueLabels-*meaning* entity table (motorcycle / rider / pillion /
trailing four-wheelers / upstream traffic stream) from the event's causal CSV,
optionally enriched with SigLIP colours. No LLM, no randomness: same CSV yields
the same table. Reliability depends on the Phase 2 min-lifespan filter
(settings.interpolation.min_track_frames), which keeps ghost/fragment tracks
out of this table.

Reference: context.md §3; incident_anchor_plan.md §6
"""

import logging

import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
_FOUR_WHEELERS = {"car", "truck", "bus"}
_MOTORCYCLE = "motorcycle"
_PERSON = "person"


def _load_df(event_id: str) -> pd.DataFrame | None:
    p = settings.paths.dataset_dir / event_id / f"{event_id}_causal_data.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def forward_dir(pts: np.ndarray) -> np.ndarray:
    """Unit vector of overall travel direction from first→last finite position."""
    pts = np.asarray(pts, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return np.array([0.0, 1.0])
    d = pts[-1] - pts[0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.array([0.0, 1.0])


def sustained_drop(v: np.ndarray, min_drop: float) -> tuple[float, int]:
    """
    Largest sustained speed drop (running-max - speed) and the frame of its onset.

    Returns (max_drop, onset_frame); onset_frame is -1 when no drop clears
    ``min_drop``. Mirrors the causal engine's candidate ranking.
    """
    v = np.asarray(v, dtype=float)
    finite = np.isfinite(v)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return 0.0, -1
    vv = v[idx]
    drop = np.maximum.accumulate(vv) - vv
    hits = np.where(drop >= min_drop)[0]
    onset = int(idx[hits[0]]) if hits.size else -1
    return float(np.max(drop)), onset


def _track_dicts(t: dict) -> tuple[dict, dict, dict, dict]:
    """Unpack a track table into frame-keyed plain dicts for fast lookup."""
    pos: dict = {}
    bbox: dict = {}
    vel: dict = {}
    time: dict = {}
    for row in t["pos"].itertuples():
        f = int(row.Index)
        pos[f] = np.array([row.Pos_X_m, row.Pos_Y_m], dtype=float)
    for row in t["bbox"].itertuples():
        f = int(row.Index)
        bbox[f] = (float(row.BBox_X1), float(row.BBox_Y1),
                   float(row.BBox_X2), float(row.BBox_Y2))
    for f, v in t["vel"].items():
        vel[int(f)] = float(v)
    for f, v in t["time"].items():
        time[int(f)] = float(v)
    return pos, bbox, vel, time


def series_for(df: pd.DataFrame, oid: str) -> dict:
    """Frame-indexed kinematics for one Object_ID."""
    sub = df[df["Object_ID"] == oid]
    cls = str(sub["Class"].dropna().iloc[0]) if sub["Class"].notna().any() else "object"
    pos = sub.set_index("Frame_ID")[["Pos_X_m", "Pos_Y_m"]]
    bbox = sub.set_index("Frame_ID")[["BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"]]
    vel = sub.set_index("Frame_ID")["Velocity_mps"]
    time = sub.set_index("Frame_ID")["Timestamp"]
    p, b, v, t = _track_dicts({"pos": pos, "bbox": bbox, "vel": vel, "time": time})
    return {"class": cls, "frames": set(pos.index.astype(int)),
            "pos": p, "bbox": b, "vel": v, "time": t}


def _mean_finite_pos(s: dict) -> np.ndarray | None:
    arr = np.array([p for p in s["pos"].values() if np.isfinite(p).all()], dtype=float)
    if arr.size == 0:
        return None
    return arr.mean(axis=0)


def _peak_speed(s: dict) -> float:
    vals = [v for v in s["vel"].values() if np.isfinite(v)]
    return max(vals) if vals else 0.0


def analyze_scene(event_id: str, colours: dict[str, str] | None = None) -> dict | None:
    """
    Deterministic entity/role table for an event.

    Returns dict with ``entities`` (id/name/kind/role/object_ids/class/colour),
    ``initiator_object``, ``initiator_entity``, ``trailing``, ``stream`` and a
    ``drop_ranking`` used as initiator fallback when no motorcycle exists.
    """
    df = _load_df(event_id)
    if df is None or df.empty:
        return None
    cfg = settings.scene
    colours = colours or {}
    series = {oid: series_for(df, oid) for oid in df["Object_ID"].unique()}

    # ── motorcycles (initiator candidate) ────────────────────────────────
    bikes = {o: s for o, s in series.items() if s["class"] == _MOTORCYCLE}
    if bikes:
        # CSV time is incident-anchored (t=0 = crash-machine burst): prefer the
        # bike observed inside the crash window [-1, +5]s (the fallen/debris
        # vehicle) over merely the longest-lived bike, which on multi-bike
        # scenes can be a later, unrelated motorcycle.
        window = (-1.0, 5.0)

        def _window_rows(o: str) -> int:
            s = bikes[o]
            return sum(1 for t in s["time"].values()
                       if np.isfinite(t) and window[0] <= t <= window[1])
        bike_oid = max(bikes, key=lambda o: (_window_rows(o), len(bikes[o]["frames"])))
    else:
        bike_oid = None
    bike_pos = _mean_finite_pos(series[bike_oid]) if bike_oid else None
    bike_dir = forward_dir(np.array([series[bike_oid]["pos"].get(f, np.array([np.nan, np.nan]))
                                     for f in sorted(series[bike_oid]["frames"])])) if bike_oid else None
    # The role geometry (trailing/upstream) must follow the ROAD flow
    # direction, not the crashing actor's slide — a skidding motorcycle's
    # forward vector is the crash direction and would flip every assignment.
    road_dir = None
    flows = [s for o, s in series.items()
             if s["class"] in _FOUR_WHEELERS and len(s["frames"]) >= settings.interpolation.min_track_frames]
    if flows:
        longest = max(flows, key=lambda s: len(s["frames"]))
        road_dir = forward_dir(np.array([longest["pos"].get(f, np.array([np.nan, np.nan]))
                                         for f in sorted(longest["frames"])]))
    if road_dir is None:
        road_dir = bike_dir
    use_dir = road_dir if road_dir is not None else np.array([0.0, 1.0])

    entities: list[dict] = []
    next_e = 1

    def add(name, kind, role, object_ids, cls=None):
        nonlocal next_e
        eid = f"E{next_e}"
        next_e += 1
        col = colours.get(object_ids[0]) if object_ids else None
        entities.append({"id": eid, "name": name, "kind": kind, "role": role,
                         "object_ids": list(object_ids), "class": cls, "colour": col})
        return eid

    initiator_entity = None
    if bike_oid is not None:
        name = "Motorcycle"
        if colours.get(bike_oid):
            name = f"{colours[bike_oid].title()} Motorcycle"
        initiator_entity = add(name, "vehicle", "initiator", [bike_oid], _MOTORCYCLE)

    # ── riders / pillion (persons near the crash who came off the bike) ──
    # A detached rider is only labelled "person" after the dismount (while
    # riding, rider+bike are a single merged "motorcycle" box), so bbox
    # overlap cannot identify them. Use geometry instead: persons whose track
    # starts around the crash anchor and whose position is close to the
    # motorcycle's fall point.
    rider_ids: list[str] = []
    if bike_oid is not None:
        # CSV frame grid is anchored: t = frame/10 - 10, so t=0 = frame 100
        # (clip exports are exactly 220 frames at 10 FPS around the anchor).
        crash_frame = 100
        persons = [(o, s) for o, s in series.items() if s["class"] == _PERSON]
        near: list[tuple[float, int, str]] = []
        for o, s in persons:
            fz = sorted(s["frames"])
            if not fz:
                continue
            lo, hi = crash_frame - 10, crash_frame + 80
            if not (lo <= fz[0] <= hi):
                continue
            m = _mean_finite_pos(s)
            if m is None or bike_pos is None:
                continue
            d = float(np.linalg.norm(m - bike_pos))
            if d <= 25.0:
                near.append((d, len(fz), o))
        near.sort(key=lambda x: (x[0], -x[1]))
        rider_m = _mean_finite_pos(series[near[0][2]]) if near else None
        if rider_m is not None:
            for o, s in persons:
                if any(o == n[2] for n in near):
                    continue
                fz = sorted(s["frames"])
                if not fz:
                    continue
                m = _mean_finite_pos(s)
                if m is None:
                    continue
                if float(np.linalg.norm(m - rider_m)) <= 15.0:
                    near.append((float(np.linalg.norm(m - rider_m)), len(fz), o))
        for i, (_, _, o) in enumerate(near[:2]):
            role = "rider" if i == 0 else "pillion"
            name = "Motorcycle Rider" if role == "rider" else "Pillion Passenger"
            add(name, "person", role, [o], _PERSON)
            rider_ids.append(o)

    # ── trailing four-wheelers (behind the incident, along travel) ───────
    trailing: list[dict] = []
    if bike_oid is not None and bike_pos is not None and bike_dir is not None:
        for oid, s in series.items():
            if oid == bike_oid or s["class"] not in _FOUR_WHEELERS:
                continue
            if _peak_speed(s) > settings.causal.max_plausible_speed_mps:
                continue
            m = _mean_finite_pos(s)
            if m is None:
                continue
            behind = float((m - bike_pos) @ use_dir)
            if behind >= 0:
                continue
            name = s["class"].title()
            if colours.get(oid):
                name = f"{colours[oid].title()} {name}"
            eid = add(name, "vehicle", "trailing", [oid], s["class"])
            trailing.append({"id": eid, "object_id": oid, "class": s["class"],
                             "behind_m": behind})
        trailing.sort(key=lambda t: t["behind_m"])  # nearest behind first

    # ── upstream traffic stream (vehicles ahead of the incident) ─────────
    stream_ids: list[str] = []
    if bike_oid is not None and bike_pos is not None and bike_dir is not None:
        for oid, s in series.items():
            if oid == bike_oid or oid in [t["object_id"] for t in trailing]:
                continue
            if s["class"] not in _VEHICLE_CLASSES:
                continue
            if _peak_speed(s) > settings.causal.max_plausible_speed_mps:
                continue
            m = _mean_finite_pos(s)
            if m is None:
                continue
            if float((m - bike_pos) @ use_dir) > 0:
                stream_ids.append(oid)
    stream_stats: dict = {"object_ids": stream_ids, "mean_speed_mps": 0.0, "density_per_frame": 0.0}
    if stream_ids:
        speeds = [v for o in stream_ids for v in series[o]["vel"].values()
                  if np.isfinite(v)]
        n_frames = max(len(series[o]["frames"]) for o in stream_ids) or 1
        present = sum(len(series[o]["frames"]) for o in stream_ids) / n_frames
        stream_stats = {"object_ids": stream_ids,
                        "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
                        "density_per_frame": present}
    stream_entity = add("Traffic Flow (Upstream)", "stream", "aggregate", stream_ids, None)

    # ── drop ranking (initiator fallback / context) ──────────────────────
    min_drop = settings.causal.min_speed_drop_mps
    ranking = []
    for oid, s in series.items():
        if _peak_speed(s) > settings.causal.max_plausible_speed_mps:
            continue
        frames = sorted(s["frames"])
        v = np.array([s["vel"].get(f, np.nan) for f in frames], dtype=float)
        drop, onset = sustained_drop(v, min_drop)
        ranking.append({"object_id": oid, "class": s["class"], "drop": drop,
                        "onset_frame": onset})
    ranking.sort(key=lambda r: -r["drop"])

    return {
        "entities": entities,
        "initiator_object": bike_oid,
        "initiator_entity": initiator_entity,
        "rider_object_ids": rider_ids,
        "trailing": trailing,
        "stream": stream_stats,
        "stream_entity": stream_entity,
        "bike_dir": bike_dir.tolist() if bike_dir is not None else None,
        "bike_pos_m": bike_pos.tolist() if bike_pos is not None else None,
        "drop_ranking": ranking,
    }