"""
Track 2: Causal Engine — multi-target causal discovery over event kinematics.

Loads an event's causal CSV, ranks up to `causal.max_targets` candidate vehicles
(each with a sustained speed drop = "braked"), and for each builds a compact set of
relative-kinematic variables and runs PCMCI+ (tigramite) to find which variables
causally drive that target's speed changes, and at what time lag. tau_max and
pc_alpha adapt to each target's usable timestep count rather than being fixed.

Speed-primary: on monocular BEV, speed is the reliable signal while acceleration is
a noisy derivative, so the variables are speeds and gaps — not accelerations.

NOTE: an event clip is ~100 timesteps, which is short for causal discovery. Results
are ranked hypotheses, not proof.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.database import get_event

logger = logging.getLogger(__name__)


def _load_event_df(event_id: str) -> pd.DataFrame | None:
    """Load an event's causal CSV, from the DB path or the conventional location."""
    ev = get_event(event_id)
    csv_path: Path | None = None
    if ev and ev.get("Causal_CSV_Path"):
        p = Path(ev["Causal_CSV_Path"])
        csv_path = p if p.exists() else None
    if csv_path is None:
        cand = settings.paths.dataset_dir / event_id / f"{event_id}_causal_data.csv"
        csv_path = cand if cand.exists() else None
    if csv_path is None:
        return None
    return pd.read_csv(csv_path)


def _object_series(df: pd.DataFrame) -> dict:
    """{Object_ID: DataFrame indexed by Frame_ID with position/velocity/class}."""
    out = {}
    for oid, sub in df.groupby("Object_ID"):
        out[oid] = sub.set_index("Frame_ID")[
            ["Pos_X_m", "Pos_Y_m", "Velocity_mps", "Class"]
        ].sort_index()
    return out


def _lead_present_fraction(target_oid: str, series: dict, frames: list, lane_tol: float, frame_lookup: dict) -> float:
    """Fraction of the target's valid frames that have a same-lane vehicle ahead."""
    s = series[target_oid]
    u = _forward_dir(s)
    tp_all = s.reindex(frames)[["Pos_X_m", "Pos_Y_m"]].to_numpy(dtype=float)
    others = {oid: o for oid, o in series.items() if oid != target_oid}
    valid = with_lead = 0
    for k, f in enumerate(frames):
        tp = tp_all[k]
        if not np.isfinite(tp).all():
            continue
        valid += 1
        objects_in_frame = frame_lookup.get(f, [])
        for oid, op_x, op_y, _ in objects_in_frame:
            if oid == target_oid:
                continue
            op = np.array([op_x, op_y], dtype=float)
            if not np.isfinite(op).all():
                continue
            r = op - tp
            if float(r @ u) > 0 and abs(float(r[0] * u[1] - r[1] * u[0])) < lane_tol:
                with_lead += 1
                break
    return with_lead / valid if valid else 0.0


def _select_targets(series: dict, min_len: int, frames: list, lane_tol: float, cfg) -> list[tuple]:
    """
    Rank candidate analysis targets: vehicles whose braking can plausibly be *explained*.

    Every vehicle with a sustained speed drop clearing `cfg.min_speed_drop_mps` is a
    candidate (falls back to all candidates if none clear the floor, so a scene with
    only subtle drops still gets analyzed rather than short-circuiting to no_target).
    Followers (a lead is present ≥ cfg.follower_lead_fraction_threshold of their
    frames) are ranked first — a reactor's speed drop can have an in-scene cause,
    whereas the frontmost braker cannot — then non-followers, each group by drop
    size descending. Capped at `cfg.max_targets`.
    """
    candidates = []
    
    # Pre-build frame lookup to avoid O(N^2) pandas .loc calls
    frame_lookup = {}
    for oid, s in series.items():
        for row in s.itertuples():
            f = row.Index
            if f not in frame_lookup:
                frame_lookup[f] = []
            frame_lookup[f].append((oid, row.Pos_X_m, row.Pos_Y_m, row.Velocity_mps))

    for oid, s in series.items():
        v = s["Velocity_mps"].to_numpy(dtype=float)
        valid = np.isfinite(v)
        if valid.sum() < min_len:
            continue
        if not np.isfinite(s["Pos_X_m"].to_numpy(dtype=float)).any():
            continue  # need near-field (non-gated) positions
        vv = v[valid]
        if float(np.nanmax(vv)) > cfg.max_plausible_speed_mps:
            continue  # implausible peak speed → projection/tracking spike, not a real vehicle
        drop = float((np.maximum.accumulate(vv) - vv).max())
        lead_frac = _lead_present_fraction(oid, series, frames, lane_tol, frame_lookup)
        candidates.append((oid, drop, lead_frac))

    if not candidates:
        return []
    meaningful = [c for c in candidates if c[1] >= cfg.min_speed_drop_mps]
    pool = meaningful if meaningful else candidates
    followers = sorted((c for c in pool if c[2] >= cfg.follower_lead_fraction_threshold),
                        key=lambda c: -c[1])
    others = sorted((c for c in pool if c[2] < cfg.follower_lead_fraction_threshold),
                     key=lambda c: -c[1])
    return (followers + others)[:cfg.max_targets]


def _forward_dir(s: pd.DataFrame) -> np.ndarray:
    """Unit vector of the target's overall travel direction, from first→last valid position."""
    p = s[["Pos_X_m", "Pos_Y_m"]].to_numpy(dtype=float)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 2:
        return np.array([0.0, 1.0])
    d = p[-1] - p[0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.array([0.0, 1.0])


def _build_variables(target_oid: str, series: dict, frames: list, lane_tol: float, frame_lookup: dict) -> dict:
    """
    Build target-centric time series over `frames`:
      tgt_speed  — target's speed (the effect)
      lead_gap   — forward distance to nearest same-lane vehicle ahead
      rel_speed  — that lead vehicle's speed minus the target's (car-following
                   causation is driven by relative speed, not raw speeds)
      nn_gap     — distance to nearest vehicle (any direction)
      nn_speed   — that nearest vehicle's speed
    """
    tgt = series[target_oid].reindex(frames)
    u = _forward_dir(series[target_oid])
    tgt_pos = tgt[["Pos_X_m", "Pos_Y_m"]].to_numpy(dtype=float)
    tgt_speed = tgt["Velocity_mps"].to_numpy(dtype=float)

    n = len(frames)
    lead_gap = np.full(n, np.nan); lead_speed = np.full(n, np.nan)
    nn_gap = np.full(n, np.nan);   nn_speed = np.full(n, np.nan)

    for k, f in enumerate(frames):
        tp = tgt_pos[k]
        if not np.isfinite(tp).all():
            continue
        best_lead = (np.inf, np.nan)
        best_nn = (np.inf, np.nan)
        
        objects_in_frame = frame_lookup.get(f, [])
        for oid, op_x, op_y, osp in objects_in_frame:
            if oid == target_oid:
                continue
            
            op = np.array([op_x, op_y], dtype=float)
            if not np.isfinite(op).all():
                continue
                
            osp = float(osp) if np.isfinite(osp) else np.nan
            r = op - tp
            dist = float(np.linalg.norm(r))
            if dist < best_nn[0]:
                best_nn = (dist, osp)
            fwd = float(r @ u)                          # forward component (ahead > 0)
            lat = abs(float(r[0] * u[1] - r[1] * u[0]))  # lateral offset
            if fwd > 0 and lat < lane_tol and fwd < best_lead[0]:
                best_lead = (fwd, osp)
                
        if np.isfinite(best_lead[0]):
            lead_gap[k], lead_speed[k] = best_lead
        if np.isfinite(best_nn[0]):
            nn_gap[k], nn_speed[k] = best_nn

    rel_speed = lead_speed - tgt_speed  # replace, not add: avoids multicollinearity in ParCorr

    return {"tgt_speed": tgt_speed, "lead_gap": lead_gap, "rel_speed": rel_speed,
            "nn_gap": nn_gap, "nn_speed": nn_speed}


def _assemble(cols: dict, min_presence_frac: float) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Keep tgt_speed plus variables that are present ≥ min_presence_frac and
    non-constant; restrict to frames where tgt_speed is finite; interpolate short
    gaps. Returns (data, names, observed_mask) with mask True = observed,
    False = missing (tigramite-native; no 999.0 sentinel).
    """
    keep_rows = np.isfinite(cols["tgt_speed"])
    names, arrays, dropped = [], [], []
    for name, arr in cols.items():
        a = arr[keep_rows]
        finite = np.isfinite(a)
        if name != "tgt_speed":
            if finite.mean() < min_presence_frac:
                dropped.append((name, f"presence {finite.mean():.2f} < {min_presence_frac}"))
                continue
            vals = a[finite]
            if vals.size and np.nanstd(vals) < 1e-6:
                dropped.append((name, "constant (ParCorr unusable)"))
                continue
        names.append(name)
        arrays.append(a)

    d = pd.DataFrame(np.column_stack(arrays), columns=names)
    d = d.interpolate(limit=5, limit_direction="both")
    mask = d.notna().to_numpy(dtype=bool)
    data = d.fillna(0.0).to_numpy(dtype=float)
    for name, why in dropped:
        logger.info("Dropped variable %s: %s", name, why)
    return data, names, mask


class CausalEngine:
    """Stateless target-centric causal analysis over a single event's CSV."""

    def analyze_event(self, event_id: str) -> dict:
        df = _load_event_df(event_id)
        if df is None or df.empty:
            return {"status": "error", "message": f"No causal CSV for {event_id}"}

        cfg = settings.causal
        series = _object_series(df)
        frames = sorted(int(f) for f in df["Frame_ID"].unique())
        candidates = _select_targets(series, cfg.min_series_len, frames, cfg.lane_tolerance_m, cfg)
        if not candidates:
            return {"status": "no_target",
                    "message": "No object with enough valid near-field data to analyze."}

        # ── PCMCI+ (lazy import to keep server startup light) ────────────────
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr

        # Pre-build frame lookup to avoid O(N^2) pandas .loc calls
        frame_lookup = {}
        for oid, s in series.items():
            for row in s.itertuples():
                f = row.Index
                if f not in frame_lookup:
                    frame_lookup[f] = []
                frame_lookup[f].append((oid, row.Pos_X_m, row.Pos_Y_m, row.Velocity_mps))

        targets = []
        for target_oid, drop, lead_frac in candidates:
            cols = _build_variables(target_oid, series, frames, cfg.lane_tolerance_m, frame_lookup)
            data, names, obs_mask = _assemble(cols, cfg.min_variable_presence_frac)
            if data.shape[0] < cfg.min_series_len or len(names) < 2:
                logger.info("Skipping target %s for %s: only %d timesteps / %d variables",
                            target_oid, event_id, data.shape[0], len(names))
                continue
            logger.info("Target %s: %d timesteps × %d variables: %s | observed-mask "
                        "coverage %.2f", target_oid, data.shape[0], len(names), names,
                        float(obs_mask.mean()))

            tau_max = max(2, min(cfg.tau_max, int(data.shape[0] * cfg.tau_max_frame_frac)))
            dataframe = pp.DataFrame(data, var_names=names, mask=~obs_mask)
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
            res = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=cfg.pc_alpha)
            graph, val = res["graph"], res["val_matrix"]

            tj = names.index("tgt_speed")
            pre_links = [(i, tau) for i in range(len(names)) for tau in range(graph.shape[2])
                         if graph[i, tj, tau] == "-->"]
            logger.info("Target %s: PCMCI+ found %d candidate links (pre self-link filter)",
                        target_oid, len(pre_links))
            links = []
            for i, tau in pre_links:
                if i == tj:
                    continue  # Fix 1: all self-links dropped (autoregression is trivially true)
                links.append({"cause": names[i], "lag": int(tau),
                              "strength": round(float(val[i, tj, tau]), 3)})
            links.sort(key=lambda l: -abs(l["strength"]))

            cls = series[target_oid]["Class"].dropna()
            targets.append({
                "target_object": target_oid,
                "target_class": str(cls.iloc[0]) if not cls.empty else None,
                "target_speed_drop_mps": round(drop, 2),
                "target_lead_fraction": round(lead_frac, 2),
                "variables": names,
                "n_timesteps": int(data.shape[0]),
                "tau_max": tau_max,
                "pc_alpha_used": res.get("optimal_alpha", cfg.pc_alpha),
                "drivers_of_target_speed": links,
            })

        if not targets:
            return {"status": "insufficient",
                    "message": "No candidate target had enough usable timesteps/variables after filtering."}

        result = {
            "status": "ok",
            "event_id": event_id,
            "targets": targets,
            "episode": self._build_episode(event_id),
            "note": "PCMCI+ over a short (~100-step) clip — treat links as ranked hypotheses, not proof.",
        }
        self._persist(event_id, result)
        return result

    @staticmethod
    def _build_episode(event_id: str) -> dict | None:
        """Stage-level narrative (nodes/relations/root cause) — deterministic, additive."""
        try:
            from app.pipeline.stage import build_episode
            return build_episode(event_id)
        except Exception as exc:
            logger.warning("Episode analysis for %s failed (non-fatal): %s", event_id, exc)
            return None

    @staticmethod
    def _persist(event_id: str, result: dict) -> None:
        out = settings.paths.dataset_dir / event_id / "causal_graph.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        n_links = sum(len(t.get("drivers_of_target_speed", [])) for t in result.get("targets", []))
        logger.info("Causal graph for %s: %d target(s), %d driver link(s) total, episode=%s",
                    event_id, len(result.get("targets", [])), n_links,
                    "present" if result.get("episode") else "absent")


_engine: CausalEngine | None = None


def get_causal_engine() -> CausalEngine:
    global _engine
    if _engine is None:
        _engine = CausalEngine()
    return _engine
