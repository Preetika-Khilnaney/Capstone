"""
causal_kinematics.py — Kinematic enrichment for the Causal Engine (Track 2).

Extends the raw causal CSV (Event_ID, Timestamp, Frame_ID, Object_ID, Class,
BBox_*, Pos_X_m, Pos_Y_m, Velocity_mps) to the full schema required by causal
discovery:

    timestamp, frame_id, object_id, class,
    world_x, world_y,
    vx, vy, speed,
    ax, ay, acceleration,
    heading,
    detection_confidence, tracking_confidence,
    position_source, velocity_source

Rules:
- Use the ACTUAL timestamp difference between consecutive frames for the same
  object_id (not a fixed dt=0.1).
- vx = Δx/Δt, vy = Δy/Δt, speed = √(vx²+vy²).
- ax = Δvx/Δt, ay = Δvy/Δt, acceleration = √(ax²+ay²).
- heading = atan2(vy, vx) in radians.
- NaN positions → NaN kinematics (no invention).
- Position source is 'observed' when raw; 'interpolated' when gap-filled.

Returns one clean pd.DataFrame per object_id (keyed dict), and a combined
"wide" DataFrame with all entities.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# ── column aliases from the legacy schema ────────────────────────────────────
_COL_MAP = {
    "Timestamp": "timestamp",
    "Frame_ID": "frame_id",
    "Object_ID": "object_id",
    "Class": "class",
    "Pos_X_m": "world_x",
    "Pos_Y_m": "world_y",
}


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoother; no-ops on NaN or too-short series."""
    if window < 5 or arr.size <= window or not np.all(np.isfinite(arr)):
        return arr
    if window % 2 == 0:
        window += 1
    try:
        return savgol_filter(arr, window, polyorder=2)
    except Exception:
        return arr


def _derive_kinematics(
    times: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    smooth_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given parallel arrays of timestamps and world positions (may contain NaN),
    compute vx, vy, speed, ax, ay, acceleration, heading.

    Uses central differences where possible, forward/backward at edges.
    Gaps (NaN positions) propagate NaN to kinematics.
    """
    n = len(times)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)

    # Smooth positions before differencing to reduce noise
    xs_s = _smooth(xs.copy(), smooth_window)
    ys_s = _smooth(ys.copy(), smooth_window)

    for i in range(n):
        # find nearest valid neighbours
        # backward neighbour
        bi = None
        for j in range(i - 1, -1, -1):
            if np.isfinite(xs_s[j]) and np.isfinite(ys_s[j]):
                bi = j
                break
        # forward neighbour
        fi = None
        for j in range(i + 1, n):
            if np.isfinite(xs_s[j]) and np.isfinite(ys_s[j]):
                fi = j
                break

        if not np.isfinite(xs_s[i]) or not np.isfinite(ys_s[i]):
            continue  # own position missing → can't compute velocity

        if bi is not None and fi is not None:
            dt = times[fi] - times[bi]
            if dt > 1e-6:
                vx[i] = (xs_s[fi] - xs_s[bi]) / dt
                vy[i] = (ys_s[fi] - ys_s[bi]) / dt
        elif bi is not None:
            dt = times[i] - times[bi]
            if dt > 1e-6:
                vx[i] = (xs_s[i] - xs_s[bi]) / dt
                vy[i] = (ys_s[i] - ys_s[bi]) / dt
        elif fi is not None:
            dt = times[fi] - times[i]
            if dt > 1e-6:
                vx[i] = (xs_s[fi] - xs_s[i]) / dt
                vy[i] = (ys_s[fi] - ys_s[i]) / dt

    speed = np.where(
        np.isfinite(vx) & np.isfinite(vy),
        np.sqrt(vx**2 + vy**2),
        np.nan,
    )

    # Acceleration
    ax = np.full(n, np.nan)
    ay = np.full(n, np.nan)
    for i in range(n):
        bi = None
        for j in range(i - 1, -1, -1):
            if np.isfinite(vx[j]) and np.isfinite(vy[j]):
                bi = j
                break
        fi = None
        for j in range(i + 1, n):
            if np.isfinite(vx[j]) and np.isfinite(vy[j]):
                fi = j
                break

        if not np.isfinite(vx[i]) or not np.isfinite(vy[i]):
            continue

        if bi is not None and fi is not None:
            dt = times[fi] - times[bi]
            if dt > 1e-6:
                ax[i] = (vx[fi] - vx[bi]) / dt
                ay[i] = (vy[fi] - vy[bi]) / dt
        elif bi is not None:
            dt = times[i] - times[bi]
            if dt > 1e-6:
                ax[i] = (vx[i] - vx[bi]) / dt
                ay[i] = (vy[i] - vy[bi]) / dt
        elif fi is not None:
            dt = times[fi] - times[i]
            if dt > 1e-6:
                ax[i] = (vx[fi] - vx[i]) / dt
                ay[i] = (vy[fi] - vy[i]) / dt

    acceleration = np.where(
        np.isfinite(ax) & np.isfinite(ay),
        np.sqrt(ax**2 + ay**2),
        np.nan,
    )
    heading = np.where(
        np.isfinite(vx) & np.isfinite(vy),
        np.arctan2(vy, vx),
        np.nan,
    )
    return vx, vy, speed, ax, ay, acceleration, heading


def enrich_kinematics(
    df: pd.DataFrame,
    smooth_window: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Enrich a raw causal CSV DataFrame to the full kinematic schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw event CSV with at minimum:
        Timestamp / Frame_ID / Object_ID / Class / Pos_X_m / Pos_Y_m
    smooth_window : int
        Savitzky-Golay smoothing window applied to positions before
        differencing. Set to 0 to disable.

    Returns
    -------
    dict : {object_id: enriched_DataFrame}, sorted by first timestamp.
    """
    # Normalise column names
    df = df.rename(columns=_COL_MAP)
    # Keep legacy column if present
    if "Velocity_mps" in df.columns and "speed_legacy" not in df.columns:
        df = df.rename(columns={"Velocity_mps": "speed_legacy"})

    required = {"timestamp", "frame_id", "object_id", "class", "world_x", "world_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Numeric coercion
    for col in ("timestamp", "frame_id", "world_x", "world_y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Position-source tag (pre-enrichment; post-interpolation rows will be 'interpolated')
    if "position_source" not in df.columns:
        df["position_source"] = "observed"

    results: Dict[str, pd.DataFrame] = {}

    for oid, sub in df.groupby("object_id", sort=False):
        sub = sub.sort_values("frame_id").copy()
        sub = sub.reset_index(drop=True)

        times = sub["timestamp"].to_numpy(dtype=float)
        xs = sub["world_x"].to_numpy(dtype=float)
        ys = sub["world_y"].to_numpy(dtype=float)

        # Clamp implausible positions (inherit from projection config if available)
        try:
            from app.config import settings
            cfg = settings.projection
            xs = np.where((xs < cfg.max_abs_lateral_m * -1) | (xs > cfg.max_abs_lateral_m), np.nan, xs)
            ys = np.where((ys < cfg.min_depth_m) | (ys > cfg.max_depth_m), np.nan, ys)
        except Exception:
            pass

        vx, vy, speed, ax, ay, acceleration, heading = _derive_kinematics(
            times, xs, ys, smooth_window=smooth_window if smooth_window >= 5 else 0
        )

        out = sub[["timestamp", "frame_id", "object_id", "class", "world_x", "world_y", "position_source"]].copy()
        out["vx"] = vx
        out["vy"] = vy
        out["speed"] = speed
        out["ax"] = ax
        out["ay"] = ay
        out["acceleration"] = acceleration
        out["heading"] = heading

        # Confidence columns (populate from source if available, else defaults)
        if "detection_confidence" in sub.columns:
            out["detection_confidence"] = sub["detection_confidence"].values
        else:
            out["detection_confidence"] = np.nan

        if "tracking_confidence" in sub.columns:
            out["tracking_confidence"] = sub["tracking_confidence"].values
        else:
            out["tracking_confidence"] = np.nan

        if "velocity_source" in sub.columns:
            out["velocity_source"] = sub["velocity_source"].values
        else:
            out["velocity_source"] = "derived"

        results[str(oid)] = out

    logger.info(
        "KinematicsEnricher: %d objects enriched from %d raw rows",
        len(results), len(df),
    )
    return results


def compute_pairwise_features(
    entity_series: Dict[str, pd.DataFrame],
    max_distance_m: float = 50.0,
    interaction_distance_m: float = 20.0,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    For every pair of entities that share overlapping timestamps AND come within
    `interaction_distance_m` of each other at least once, compute a time series
    of pairwise interaction features aligned on the shared timestamp grid.

    Features:
        distance, distance_rate (closing speed positive = approaching),
        relative_x, relative_y, relative_vx, relative_vy,
        relative_speed, closing_speed,
        relative_acceleration, heading_difference,
        overlap_indicator (bool approx based on distance < 2m)

    Returns
    -------
    dict : {(id_a, id_b): DataFrame}  — only relevant pairs included.
    """
    pair_results: Dict[Tuple[str, str], pd.DataFrame] = {}
    ids = sorted(entity_series.keys())

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            sa = entity_series[id_a].set_index("frame_id")
            sb = entity_series[id_b].set_index("frame_id")

            common_frames = sa.index.intersection(sb.index)
            if len(common_frames) < 5:
                continue

            ra = sa.loc[common_frames]
            rb = sb.loc[common_frames]

            ax_arr = ra["world_x"].to_numpy(dtype=float)
            ay_arr = ra["world_y"].to_numpy(dtype=float)
            bx_arr = rb["world_x"].to_numpy(dtype=float)
            by_arr = rb["world_y"].to_numpy(dtype=float)

            dx = bx_arr - ax_arr
            dy = by_arr - ay_arr
            dist = np.sqrt(dx**2 + dy**2)

            # Only keep pairs that come close enough
            min_dist = float(np.nanmin(dist)) if np.any(np.isfinite(dist)) else np.inf
            if min_dist > interaction_distance_m:
                continue

            avx = ra["vx"].to_numpy(dtype=float)
            avy = ra["vy"].to_numpy(dtype=float)
            bvx = rb["vx"].to_numpy(dtype=float)
            bvy = rb["vy"].to_numpy(dtype=float)

            rel_vx = bvx - avx
            rel_vy = bvy - avy
            rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)

            # Closing speed: component of relative velocity along separation vector (positive = closing)
            dist_safe = np.where(dist > 1e-6, dist, np.nan)
            unit_dx = dx / dist_safe
            unit_dy = dy / dist_safe
            closing_speed = -(rel_vx * unit_dx + rel_vy * unit_dy)  # positive = approaching

            # Distance rate (numerical derivative of distance)
            timestamps = ra["timestamp"].to_numpy(dtype=float)
            dist_rate = np.full(len(dist), np.nan)
            for k in range(1, len(dist) - 1):
                dt = timestamps[k + 1] - timestamps[k - 1]
                if dt > 1e-6 and np.isfinite(dist[k - 1]) and np.isfinite(dist[k + 1]):
                    dist_rate[k] = (dist[k + 1] - dist[k - 1]) / dt
            # edges
            if len(dist) > 1:
                dt0 = timestamps[1] - timestamps[0]
                if dt0 > 1e-6 and np.isfinite(dist[0]) and np.isfinite(dist[1]):
                    dist_rate[0] = (dist[1] - dist[0]) / dt0
                dt_last = timestamps[-1] - timestamps[-2]
                if dt_last > 1e-6 and np.isfinite(dist[-2]) and np.isfinite(dist[-1]):
                    dist_rate[-1] = (dist[-1] - dist[-2]) / dt_last

            # Relative acceleration
            aax = ra["ax"].to_numpy(dtype=float)
            aay = ra["ay"].to_numpy(dtype=float)
            bax = rb["ax"].to_numpy(dtype=float)
            bay = rb["ay"].to_numpy(dtype=float)
            rel_acc = np.sqrt((bax - aax)**2 + (bay - aay)**2)

            # Heading difference (wrapped to [-π, π])
            a_head = ra["heading"].to_numpy(dtype=float)
            b_head = rb["heading"].to_numpy(dtype=float)
            head_diff = np.arctan2(np.sin(b_head - a_head), np.cos(b_head - a_head))

            overlap = (dist < 2.0).astype(float)
            # NaN where dist itself is NaN
            overlap = np.where(np.isfinite(dist), overlap, np.nan)

            pair_df = pd.DataFrame({
                "frame_id": common_frames.to_numpy(),
                "timestamp": timestamps,
                "distance": dist,
                "distance_rate": dist_rate,
                "relative_x": dx,
                "relative_y": dy,
                "relative_vx": rel_vx,
                "relative_vy": rel_vy,
                "relative_speed": rel_speed,
                "closing_speed": closing_speed,
                "relative_acceleration": rel_acc,
                "heading_difference": head_diff,
                "overlap_indicator": overlap,
            })

            pair_results[(id_a, id_b)] = pair_df
            logger.debug("Pair %s↔%s: %d frames, min_dist=%.1fm", id_a, id_b, len(pair_df), min_dist)

    logger.info(
        "Pairwise features: %d relevant pairs from %d entities",
        len(pair_results), len(ids),
    )
    return pair_results
