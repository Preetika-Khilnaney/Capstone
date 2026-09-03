"""
causal_discovery.py — Multi-method statistical causal discovery for Track 2.

Implements and wraps:
  1. PCMCI+    (tigramite)        — primary temporal causal method
  2. DirectLiNGAM (lingam)       — directional causal evidence
  3. GES       (causal-learn)    — score-based structural evidence
  4. Causal Forest (econml)      — heterogeneous treatment effects
  5. CycleNet  — marked unavailable (not installable)

All methods produce normalised edge lists:
  {source, target, strength_raw, strength_norm, method, ...metadata}

A consensus mechanism aggregates across available methods.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MISSING_FLAG = 999.0  # tigramite missing-value sentinel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_to_01(values: List[float]) -> List[float]:
    """Min-max normalise a list of floats to [0, 1]."""
    arr = np.array(values, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-10:
        return [0.5] * len(values)
    return ((arr - mn) / (mx - mn)).tolist()


def _prepare_matrix(
    df: pd.DataFrame,
    var_names: List[str],
    min_presence: float = 0.4,
) -> Tuple[np.ndarray, List[str]]:
    """
    Select columns from df, drop near-constant and low-presence variables,
    interpolate short gaps, fill residuals with MISSING_FLAG.
    Returns (data_array [T x N], kept_var_names).
    """
    cols, arrays = [], []
    for name in var_names:
        if name not in df.columns:
            continue
        a = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        finite_frac = np.isfinite(a).mean()
        if finite_frac < min_presence:
            continue
        vals = a[np.isfinite(a)]
        if vals.size > 0 and np.nanstd(vals) < 1e-8:
            continue  # constant
        cols.append(name)
        arrays.append(a)

    if not cols:
        return np.empty((0, 0)), []

    mat = np.column_stack(arrays)
    # Interpolate short gaps
    tmp = pd.DataFrame(mat, columns=cols)
    tmp = tmp.interpolate(limit=5, limit_direction="both")
    mat = tmp.to_numpy(dtype=float)
    mat = np.where(np.isfinite(mat), mat, MISSING_FLAG)
    return mat, cols


# ─────────────────────────────────────────────────────────────────────────────
# 1. PCMCI+
# ─────────────────────────────────────────────────────────────────────────────

def run_pcmci_plus(
    df: pd.DataFrame,
    var_names: List[str],
    tau_min: int = 1,
    tau_max: int = 8,
    pc_alpha: Optional[float] = None,
    fps: float = 10.0,
) -> Dict[str, Any]:
    """
    Run PCMCI+ on the given variable columns.

    Returns a result dict:
      {
        "status": "ok" | "error" | "insufficient",
        "edges": [{source, target, lag_frames, lag_seconds, strength, p_value, method}],
        "graph": raw graph array,
        "var_names": [...],
        "tau_max": ...,
      }
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        return {"status": "unavailable", "reason": "tigramite not installed", "edges": []}

    data, kept = _prepare_matrix(df, var_names)
    if data.shape[0] < 20 or len(kept) < 2:
        return {"status": "insufficient", "edges": [], "var_names": kept}

    # Adaptive tau_max
    tau_max = max(2, min(tau_max, int(data.shape[0] * 0.12)))

    try:
        dataframe = pp.DataFrame(data, var_names=kept, missing_flag=MISSING_FLAG)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
        res = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
        graph = res["graph"]        # shape [N, N, tau_max+1]
        val_mat = res["val_matrix"] # partial correlation strengths
        p_mat = res.get("p_matrix", np.ones_like(val_mat))

        edges = []
        for i, src in enumerate(kept):
            for j, tgt in enumerate(kept):
                for tau in range(graph.shape[2]):
                    if i == j and tau == 0:
                        continue
                    if graph[i, j, tau] == "-->":
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "lag_frames": int(tau),
                            "lag_seconds": round(tau / fps, 3),
                            "strength": round(float(val_mat[i, j, tau]), 4),
                            "strength_norm": None,  # filled by consensus
                            "p_value": round(float(p_mat[i, j, tau]), 4),
                            "method": "PCMCI+",
                        })

        # Normalise strengths by absolute magnitude
        if edges:
            raw = [abs(e["strength"]) for e in edges]
            normed = _normalise_to_01(raw)
            for e, n in zip(edges, normed):
                e["strength_norm"] = round(n, 4)

        logger.info("PCMCI+: %d causal edges from %d variables (tau_max=%d)", len(edges), len(kept), tau_max)
        return {
            "status": "ok",
            "edges": edges,
            "var_names": kept,
            "tau_max": tau_max,
            "n_timesteps": data.shape[0],
        }
    except Exception as exc:
        logger.warning("PCMCI+ failed: %s", exc, exc_info=True)
        return {"status": "error", "reason": str(exc), "edges": []}


# ─────────────────────────────────────────────────────────────────────────────
# 2. DirectLiNGAM
# ─────────────────────────────────────────────────────────────────────────────

def run_lingam(
    df: pd.DataFrame,
    var_names: List[str],
) -> Dict[str, Any]:
    """
    Run DirectLiNGAM on the given variable columns.

    Returns a result dict with edges:
      [{source, target, strength, strength_norm, p_value, method}]
    Note: LiNGAM is not temporal; lag_frames is 0.
    """
    try:
        import lingam
    except ImportError:
        return {"status": "unavailable", "reason": "lingam not installed", "edges": []}

    data, kept = _prepare_matrix(df, var_names)
    if data.shape[0] < 20 or len(kept) < 2:
        return {"status": "insufficient", "edges": [], "var_names": kept}

    # Replace MISSING_FLAG with NaN before LiNGAM (it doesn't accept the flag)
    data_clean = np.where(data == MISSING_FLAG, np.nan, data)
    # Drop rows with any NaN
    mask = np.all(np.isfinite(data_clean), axis=1)
    data_clean = data_clean[mask]
    if data_clean.shape[0] < 20:
        return {"status": "insufficient", "edges": [], "var_names": kept,
                "reason": "Too few complete rows after NaN removal"}

    # Standardise
    means = data_clean.mean(axis=0)
    stds = data_clean.std(axis=0)
    stds[stds < 1e-10] = 1.0
    data_std = (data_clean - means) / stds

    try:
        model = lingam.DirectLiNGAM()
        model.fit(data_std)
        adj = model.adjacency_matrix_  # shape [N, N]; adj[i,j] = effect of kept[j] on kept[i]

        edges = []
        n = len(kept)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = float(adj[i, j])
                if abs(val) > 0.05:  # threshold weak edges
                    edges.append({
                        "source": kept[j],
                        "target": kept[i],
                        "lag_frames": 0,
                        "lag_seconds": 0.0,
                        "strength": round(val, 4),
                        "strength_norm": None,
                        "p_value": None,
                        "method": "LiNGAM",
                    })

        if edges:
            raw = [abs(e["strength"]) for e in edges]
            normed = _normalise_to_01(raw)
            for e, n_val in zip(edges, normed):
                e["strength_norm"] = round(n_val, 4)

        logger.info("LiNGAM: %d causal edges from %d variables", len(edges), len(kept))
        return {"status": "ok", "edges": edges, "var_names": kept, "adjacency_matrix": adj.tolist()}
    except Exception as exc:
        logger.warning("LiNGAM failed: %s", exc, exc_info=True)
        return {"status": "error", "reason": str(exc), "edges": []}


# ─────────────────────────────────────────────────────────────────────────────
# 3. GES (Greedy Equivalence Search)
# ─────────────────────────────────────────────────────────────────────────────

def run_ges(
    df: pd.DataFrame,
    var_names: List[str],
) -> Dict[str, Any]:
    """
    Run GES (BIC score) on the given variable columns via causal-learn.

    Returns edges:
      [{source, target, lag_frames, lag_seconds, strength, method}]
    GES is not temporal; lag=0.
    """
    try:
        from causallearn.search.ScoreBased.GES import ges
    except ImportError:
        return {"status": "unavailable", "reason": "causal-learn not installed", "edges": []}

    data, kept = _prepare_matrix(df, var_names)
    if data.shape[0] < 20 or len(kept) < 2:
        return {"status": "insufficient", "edges": [], "var_names": kept}

    data_clean = np.where(data == MISSING_FLAG, np.nan, data)
    mask = np.all(np.isfinite(data_clean), axis=1)
    data_clean = data_clean[mask]
    if data_clean.shape[0] < 20:
        return {"status": "insufficient", "edges": [], "var_names": kept,
                "reason": "Too few complete rows"}

    # Standardise
    means = data_clean.mean(axis=0)
    stds = data_clean.std(axis=0)
    stds[stds < 1e-10] = 1.0
    data_std = (data_clean - means) / stds

    try:
        record = ges(data_std, score_func="local_score_BIC", maxP=4)
        # record["G"] is a GeneralGraph; adjacency via record["G"].graph
        G = record["G"]
        adj = np.array(G.graph)  # shape [N, N]; adj[i,j]=1 means i→j or i—j

        edges = []
        n = len(kept)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if adj[i, j] == 1 and adj[j, i] == -1:
                    # Directed: kept[i] → kept[j]
                    edges.append({
                        "source": kept[i],
                        "target": kept[j],
                        "lag_frames": 0,
                        "lag_seconds": 0.0,
                        "strength": 1.0,  # GES doesn't give strength scores per edge
                        "strength_norm": 1.0,
                        "p_value": None,
                        "method": "GES",
                    })
                elif adj[i, j] == 1 and adj[j, i] == 1:
                    # Undirected: add both directions with lower strength
                    edges.append({
                        "source": kept[i],
                        "target": kept[j],
                        "lag_frames": 0,
                        "lag_seconds": 0.0,
                        "strength": 0.5,
                        "strength_norm": 0.5,
                        "p_value": None,
                        "method": "GES",
                    })

        logger.info("GES: %d edges from %d variables", len(edges), len(kept))
        return {
            "status": "ok",
            "edges": edges,
            "var_names": kept,
            "bic_score": float(record.get("score", 0.0)),
        }
    except Exception as exc:
        logger.warning("GES failed: %s", exc, exc_info=True)
        return {"status": "error", "reason": str(exc), "edges": []}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Causal Forest
# ─────────────────────────────────────────────────────────────────────────────

def run_causal_forest(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: List[str],
    treatment_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Estimate heterogeneous treatment effects using CausalForestDML.

    Parameters
    ----------
    df : DataFrame aligned on frame_id
    treatment_col : binary treatment indicator (e.g. 'V01_sudden_braking')
    outcome_col : outcome variable (e.g. 'V02_deceleration')
    covariate_cols : confound controls (e.g. ['distance', 'relative_speed'])
    treatment_threshold : value above which treatment_col → 1

    Returns
    -------
    {status, treatment, outcome, effect_mean, effect_std, method}
    """
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {"status": "unavailable", "reason": "econml not installed", "edges": []}

    if treatment_col not in df.columns or outcome_col not in df.columns:
        return {"status": "skipped", "reason": "treatment or outcome column missing", "edges": []}

    # Build clean dataset
    cols = [treatment_col, outcome_col] + [c for c in covariate_cols if c in df.columns]
    sub = df[cols].copy()
    sub = sub.replace(MISSING_FLAG, np.nan).dropna()

    if len(sub) < 30:
        return {
            "status": "skipped",
            "reason": f"Only {len(sub)} complete rows — not enough for causal forest",
            "edges": [],
        }

    T = (sub[treatment_col].to_numpy() > treatment_threshold).astype(float)
    Y = sub[outcome_col].to_numpy(dtype=float)
    cov_cols_used = [c for c in covariate_cols if c in sub.columns]
    X = sub[cov_cols_used].to_numpy(dtype=float) if cov_cols_used else np.ones((len(sub), 1))

    # Check treatment variation
    if T.mean() < 0.05 or T.mean() > 0.95:
        return {
            "status": "skipped",
            "reason": f"Treatment prevalence {T.mean():.2f} — insufficient variation",
            "edges": [],
        }

    try:
        model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            model_t=LogisticRegression(max_iter=500, random_state=42),
            discrete_treatment=True,
            n_estimators=200,
            random_state=42,
        )
        model.fit(Y, T, X=X)
        effects = model.effect(X)
        eff_mean = float(np.mean(effects))
        eff_std = float(np.std(effects))

        try:
            lb, ub = model.effect_interval(X, alpha=0.1)
            ci = (float(np.mean(lb)), float(np.mean(ub)))
        except Exception:
            ci = None

        edge = {
            "source": treatment_col,
            "target": outcome_col,
            "lag_frames": 0,
            "lag_seconds": 0.0,
            "strength": round(eff_mean, 4),
            "strength_norm": None,
            "effect_std": round(eff_std, 4),
            "confidence_interval_90": ci,
            "p_value": None,
            "method": "CausalForest",
        }

        logger.info(
            "CausalForest: treatment=%s → outcome=%s, ATE=%.4f±%.4f",
            treatment_col, outcome_col, eff_mean, eff_std,
        )
        return {
            "status": "ok",
            "treatment": treatment_col,
            "outcome": outcome_col,
            "covariates": cov_cols_used,
            "n_samples": len(sub),
            "effect_mean": round(eff_mean, 4),
            "effect_std": round(eff_std, 4),
            "edges": [edge],
        }
    except Exception as exc:
        logger.warning("CausalForest failed: %s", exc, exc_info=True)
        return {"status": "error", "reason": str(exc), "edges": []}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-method consensus
# ─────────────────────────────────────────────────────────────────────────────

def build_consensus(
    method_results: Dict[str, Dict],
    min_support_ratio: float = 0.33,
) -> List[Dict]:
    """
    Aggregate edges from all available methods into consensus edges.

    For each unique (source, target) pair, compute:
      - available_methods: methods that ran successfully
      - support_count: how many support this edge
      - support_ratio: support_count / available_methods_count
      - average_strength_norm: mean of normalised strengths
      - final_confidence: geometric blend of support_ratio and strength

    Only edges exceeding min_support_ratio are included.
    """
    available_methods = [m for m, r in method_results.items() if r.get("status") == "ok"]
    n_available = max(1, len(available_methods))

    # Collect all edges keyed by (source, target)
    edge_map: Dict[Tuple[str, str], Dict] = {}

    for method, result in method_results.items():
        if result.get("status") != "ok":
            continue
        for edge in result.get("edges", []):
            key = (edge["source"], edge["target"])
            if key not in edge_map:
                edge_map[key] = {
                    "source": edge["source"],
                    "target": edge["target"],
                    "method_support": {},
                    "lag_frames_list": [],
                    "lag_seconds_list": [],
                    "p_values": [],
                }
            rec = edge_map[key]
            norm = edge.get("strength_norm")
            rec["method_support"][method] = {
                "strength": edge.get("strength"),
                "strength_norm": norm,
                "lag_frames": edge.get("lag_frames", 0),
                "p_value": edge.get("p_value"),
            }
            if edge.get("lag_frames") is not None:
                rec["lag_frames_list"].append(edge["lag_frames"])
                rec["lag_seconds_list"].append(edge.get("lag_seconds", 0))
            if edge.get("p_value") is not None:
                rec["p_values"].append(edge["p_value"])

    consensus = []
    for key, rec in edge_map.items():
        support_count = len(rec["method_support"])
        support_ratio = support_count / n_available

        if support_ratio < min_support_ratio:
            continue

        norms = [v["strength_norm"] for v in rec["method_support"].values() if v["strength_norm"] is not None]
        avg_norm = float(np.mean(norms)) if norms else 0.5

        # Lag: median of reported lags
        lag_f = int(round(np.median(rec["lag_frames_list"]))) if rec["lag_frames_list"] else 0
        lag_s = round(float(np.median(rec["lag_seconds_list"])), 3) if rec["lag_seconds_list"] else 0.0

        # p-value: max (most conservative)
        p_val = round(max(rec["p_values"]), 4) if rec["p_values"] else None

        # Final confidence: blend support_ratio and average normalised strength
        final_confidence = round(0.6 * support_ratio + 0.4 * avg_norm, 4)

        # Relationship type
        if final_confidence >= 0.65 and support_ratio >= 0.5:
            relationship = "causes"
        elif final_confidence >= 0.40:
            relationship = "supports"
        else:
            relationship = "precedes"

        consensus.append({
            "source": rec["source"],
            "target": rec["target"],
            "relationship": relationship,
            "lag_frames": lag_f,
            "lag_seconds": lag_s,
            "p_value": p_val,
            "average_strength_norm": round(avg_norm, 4),
            "support_count": support_count,
            "available_methods": n_available,
            "support_ratio": round(support_ratio, 4),
            "method_support": rec["method_support"],
            "final_confidence": final_confidence,
        })

    consensus.sort(key=lambda e: -e["final_confidence"])
    logger.info(
        "Consensus: %d edges from %d available methods (min_support=%.2f)",
        len(consensus), n_available, min_support_ratio,
    )
    return consensus
