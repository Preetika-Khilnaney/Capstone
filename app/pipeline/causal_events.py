"""
causal_events.py — Temporal event extraction and event-level causal reasoning.

Converts kinematic time series into discrete temporal events (SUDDEN_BRAKING,
CLOSING_DISTANCE, CONTACT, etc.) and builds an event-level causal graph that
answers WHAT HAPPENED → WHAT HAPPENED NEXT → WHY.

Also implements EventOnsetLocalizer: finds the earliest physically supported
transition frame for each event (e.g. actual contact vs. later confirmation).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Event type registry
# ─────────────────────────────────────────────────────────────────────────────

EVENT_TYPES = {
    "SUDDEN_BRAKING",
    "SUDDEN_ACCELERATION",
    "SUDDEN_STOP",
    "SUDDEN_DIRECTION_CHANGE",
    "SWERVE",
    "LANE_DEVIATION",
    "APPROACHING",
    "CLOSING_DISTANCE",
    "CONTACT",
    "NEAR_COLLISION",
    "COLLISION",
    "FALL",
    "TRAJECTORY_DEVIATION",
    "ABNORMAL_MOTION",
}


@dataclass
class TemporalEvent:
    event_id: str
    object_ids: List[str]
    event_type: str
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    features: Dict = field(default_factory=dict)
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (all tuneable)
# ─────────────────────────────────────────────────────────────────────────────

BRAKING_ACCEL_THRESH = -1.5         # m/s²   (negative = deceleration)
ACCEL_THRESH = 1.5                  # m/s²
STOP_SPEED_THRESH = 0.3             # m/s    (consider stopped)
DIRECTION_CHANGE_DEG = 20.0         # degrees per second
SWERVE_LAT_ACCEL_THRESH = 1.2      # m/s²
CLOSING_SPEED_THRESH = 1.0          # m/s    (positive = approaching)
CONTACT_DIST_THRESH = 3.0           # metres (vehicles within this → contact)
NEAR_COLL_DIST_THRESH = 6.0         # metres
TRAJ_DEV_HEADING_CHANGE_DEG = 30.0 # degrees (sudden heading change)
FALL_SPEED_DROP_FRAC = 0.7          # speed drops to 30% of max in short window


def _rolling_mean(arr: np.ndarray, w: int = 3) -> np.ndarray:
    """Simple causal rolling mean (no look-ahead)."""
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        sl = arr[max(0, i - w + 1):i + 1]
        finite = sl[np.isfinite(sl)]
        if len(finite) > 0:
            out[i] = finite.mean()
    return out


def _deg(rad: float) -> float:
    return math.degrees(rad)


# ─────────────────────────────────────────────────────────────────────────────
# Entity-level event extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_entity_events(
    oid: str,
    df: pd.DataFrame,
    min_duration_frames: int = 2,
) -> List[TemporalEvent]:
    """
    Extract temporal events for a single entity from its enriched kinematic DataFrame.
    """
    events: List[TemporalEvent] = []
    df = df.sort_values("frame_id").reset_index(drop=True)

    frames = df["frame_id"].to_numpy(dtype=int)
    timestamps = df["timestamp"].to_numpy(dtype=float)
    speed = df["speed"].to_numpy(dtype=float)
    accel = df["acceleration"].to_numpy(dtype=float)
    ax_arr = df["ax"].to_numpy(dtype=float)
    ay_arr = df["ay"].to_numpy(dtype=float)
    heading = df["heading"].to_numpy(dtype=float)
    vx = df["vx"].to_numpy(dtype=float)
    vy = df["vy"].to_numpy(dtype=float)

    n = len(frames)
    eid_counter = [0]

    def _new_id() -> str:
        eid_counter[0] += 1
        return f"EVT_{oid}_{eid_counter[0]:03d}"

    # ── SUDDEN_BRAKING / SUDDEN_STOP ──────────────────────────────────────
    in_event = False
    ev_start = 0
    for i in range(1, n):
        if not np.isfinite(ax_arr[i]) or not np.isfinite(ay_arr[i]):
            continue
        # Deceleration in direction of motion
        if np.isfinite(vx[i]) and np.isfinite(vy[i]) and np.isfinite(speed[i]) and speed[i] > 0.1:
            # Longitudinal acceleration (component along velocity vector)
            spd = speed[i]
            lon_acc = (ax_arr[i] * vx[i] + ay_arr[i] * vy[i]) / spd
        else:
            lon_acc = -float(accel[i]) if np.isfinite(accel[i]) else 0.0

        braking = lon_acc < BRAKING_ACCEL_THRESH

        if braking and not in_event:
            in_event = True
            ev_start = i
        elif not braking and in_event:
            dur = frames[i - 1] - frames[ev_start]
            if dur >= min_duration_frames:
                # Distinguish STOP from BRAKING
                final_speed = speed[i - 1] if np.isfinite(speed[i - 1]) else None
                ev_type = "SUDDEN_STOP" if (final_speed is not None and final_speed < STOP_SPEED_THRESH) else "SUDDEN_BRAKING"
                events.append(TemporalEvent(
                    event_id=_new_id(),
                    object_ids=[oid],
                    event_type=ev_type,
                    start_frame=int(frames[ev_start]),
                    end_frame=int(frames[i - 1]),
                    start_timestamp=float(timestamps[ev_start]),
                    end_timestamp=float(timestamps[i - 1]),
                    features={"lon_acc_mean": round(lon_acc, 3)},
                    confidence=0.85,
                ))
            in_event = False

    # ── SUDDEN_ACCELERATION ───────────────────────────────────────────────
    in_event = False
    for i in range(1, n):
        if not np.isfinite(accel[i]):
            continue
        acc = float(accel[i])
        if np.isfinite(vx[i]) and np.isfinite(vy[i]) and np.isfinite(speed[i]) and speed[i] > 0.1:
            lon_acc = (ax_arr[i] * vx[i] + ay_arr[i] * vy[i]) / speed[i]
        else:
            lon_acc = acc

        accelerating = lon_acc > ACCEL_THRESH
        if accelerating and not in_event:
            in_event = True
            ev_start = i
        elif not accelerating and in_event:
            dur = frames[i - 1] - frames[ev_start]
            if dur >= min_duration_frames:
                events.append(TemporalEvent(
                    event_id=_new_id(),
                    object_ids=[oid],
                    event_type="SUDDEN_ACCELERATION",
                    start_frame=int(frames[ev_start]),
                    end_frame=int(frames[i - 1]),
                    start_timestamp=float(timestamps[ev_start]),
                    end_timestamp=float(timestamps[i - 1]),
                    features={"lon_acc_mean": round(lon_acc, 3)},
                    confidence=0.80,
                ))
            in_event = False

    # ── TRAJECTORY_DEVIATION / SWERVE ─────────────────────────────────────
    prev_heading = None
    in_event = False
    for i in range(n):
        if not np.isfinite(heading[i]):
            prev_heading = None
            continue
        if prev_heading is not None:
            dt = timestamps[i] - timestamps[i - 1]
            if dt > 1e-6:
                dh = heading[i] - prev_heading
                # Wrap
                dh = math.atan2(math.sin(dh), math.cos(dh))
                dh_deg_s = abs(math.degrees(dh)) / dt
                is_dev = dh_deg_s > DIRECTION_CHANGE_DEG

                if is_dev and not in_event:
                    in_event = True
                    ev_start = i
                    lat_acc = abs(float(ay_arr[i])) if np.isfinite(ay_arr[i]) else 0.0
                elif not is_dev and in_event:
                    dur = frames[i - 1] - frames[ev_start]
                    if dur >= min_duration_frames:
                        ev_type = "SWERVE" if lat_acc > SWERVE_LAT_ACCEL_THRESH else "TRAJECTORY_DEVIATION"
                        events.append(TemporalEvent(
                            event_id=_new_id(),
                            object_ids=[oid],
                            event_type=ev_type,
                            start_frame=int(frames[ev_start]),
                            end_frame=int(frames[i - 1]),
                            start_timestamp=float(timestamps[ev_start]),
                            end_timestamp=float(timestamps[i - 1]),
                            features={"heading_rate_deg_s": round(dh_deg_s, 2)},
                            confidence=0.75,
                        ))
                    in_event = False

        prev_heading = float(heading[i])

    # ── FALL (rapid speed drop + trajectory deviation) ─────────────────────
    if n > 10:
        max_speed = np.nanmax(speed) if np.any(np.isfinite(speed)) else 0.0
        if max_speed > 1.0:
            for i in range(5, n):
                if not np.isfinite(speed[i]):
                    continue
                window_max = np.nanmax(speed[max(0, i - 10):i])
                if window_max > 0.5 and speed[i] < window_max * (1 - FALL_SPEED_DROP_FRAC):
                    events.append(TemporalEvent(
                        event_id=_new_id(),
                        object_ids=[oid],
                        event_type="FALL",
                        start_frame=int(frames[i]),
                        end_frame=int(frames[min(i + 3, n - 1)]),
                        start_timestamp=float(timestamps[i]),
                        end_timestamp=float(timestamps[min(i + 3, n - 1)]),
                        features={"speed_before": round(window_max, 3), "speed_after": round(float(speed[i]), 3)},
                        confidence=0.70,
                    ))
                    break  # one fall per entity

    logger.debug("Entity %s: extracted %d events", oid, len(events))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise event extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pairwise_events(
    id_a: str,
    id_b: str,
    pair_df: pd.DataFrame,
    min_duration_frames: int = 2,
) -> List[TemporalEvent]:
    """
    Extract pairwise interaction events (APPROACHING, CLOSING_DISTANCE, CONTACT, etc.)
    from the pairwise feature DataFrame.
    """
    events: List[TemporalEvent] = []
    pair_df = pair_df.sort_values("frame_id").reset_index(drop=True)

    frames = pair_df["frame_id"].to_numpy(dtype=int)
    timestamps = pair_df["timestamp"].to_numpy(dtype=float)
    dist = pair_df["distance"].to_numpy(dtype=float)
    closing = pair_df["closing_speed"].to_numpy(dtype=float)
    dist_rate = pair_df["distance_rate"].to_numpy(dtype=float)

    pair_label = f"{id_a}_{id_b}"
    eid_counter = [0]

    def _new_id() -> str:
        eid_counter[0] += 1
        return f"EVT_PAIR_{pair_label}_{eid_counter[0]:03d}"

    n = len(frames)

    # ── APPROACHING / CLOSING_DISTANCE ────────────────────────────────────
    in_event = False
    for i in range(n):
        approaching = (
            np.isfinite(closing[i]) and closing[i] > CLOSING_SPEED_THRESH
            and np.isfinite(dist[i]) and dist[i] > CONTACT_DIST_THRESH
        )
        if approaching and not in_event:
            in_event = True
            ev_start = i
        elif not approaching and in_event:
            dur = frames[i - 1] - frames[ev_start]
            if dur >= min_duration_frames:
                d_start = dist[ev_start] if np.isfinite(dist[ev_start]) else None
                d_end = dist[i - 1] if np.isfinite(dist[i - 1]) else None
                ev_type = "CLOSING_DISTANCE" if (d_start and d_end and d_start - d_end > 2.0) else "APPROACHING"
                events.append(TemporalEvent(
                    event_id=_new_id(),
                    object_ids=[id_a, id_b],
                    event_type=ev_type,
                    start_frame=int(frames[ev_start]),
                    end_frame=int(frames[i - 1]),
                    start_timestamp=float(timestamps[ev_start]),
                    end_timestamp=float(timestamps[i - 1]),
                    features={
                        "distance_start": round(d_start, 2) if d_start else None,
                        "distance_end": round(d_end, 2) if d_end else None,
                        "peak_closing_speed": round(float(np.nanmax(closing[ev_start:i])), 3),
                    },
                    confidence=0.80,
                ))
            in_event = False

    # ── CONTACT / NEAR_COLLISION / COLLISION ──────────────────────────────
    contact_frames = np.where(np.isfinite(dist) & (dist <= CONTACT_DIST_THRESH))[0]
    near_frames = np.where(np.isfinite(dist) & (dist <= NEAR_COLL_DIST_THRESH) & (dist > CONTACT_DIST_THRESH))[0]

    if len(contact_frames) >= min_duration_frames:
        # Check for sudden relative velocity change (signature of physical impact)
        rel_speed = pair_df["relative_speed"].to_numpy(dtype=float)
        rel_acc = pair_df["relative_acceleration"].to_numpy(dtype=float)
        cf_start = contact_frames[0]
        cf_end = contact_frames[-1]

        # Look for sharp relative velocity change during/just after contact
        post_window = rel_speed[cf_start:min(cf_end + 10, n)]
        has_impact_signature = (
            np.any(np.isfinite(rel_acc[cf_start:min(cf_end + 5, n)]) & (rel_acc[cf_start:min(cf_end + 5, n)] > 1.5))
        )

        ev_type = "COLLISION" if has_impact_signature else "CONTACT"
        events.append(TemporalEvent(
            event_id=_new_id(),
            object_ids=[id_a, id_b],
            event_type=ev_type,
            start_frame=int(frames[cf_start]),
            end_frame=int(frames[cf_end]),
            start_timestamp=float(timestamps[cf_start]),
            end_timestamp=float(timestamps[cf_end]),
            features={
                "min_distance": round(float(np.nanmin(dist[cf_start:cf_end + 1])), 3),
                "has_impact_signature": bool(has_impact_signature),
            },
            confidence=0.90 if has_impact_signature else 0.80,
        ))
    elif len(near_frames) >= min_duration_frames:
        nf_start, nf_end = near_frames[0], near_frames[-1]
        events.append(TemporalEvent(
            event_id=_new_id(),
            object_ids=[id_a, id_b],
            event_type="NEAR_COLLISION",
            start_frame=int(frames[nf_start]),
            end_frame=int(frames[nf_end]),
            start_timestamp=float(timestamps[nf_start]),
            end_timestamp=float(timestamps[nf_end]),
            features={"min_distance": round(float(np.nanmin(dist[nf_start:nf_end + 1])), 3)},
            confidence=0.75,
        ))

    logger.debug("Pair %s↔%s: extracted %d events", id_a, id_b, len(events))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Event causality
# ─────────────────────────────────────────────────────────────────────────────

def _granger_check(
    cause_series: np.ndarray,
    effect_series: np.ndarray,
    timestamps: np.ndarray,
    max_lag: int = 8,
) -> Tuple[bool, float]:
    """
    Lightweight Granger-causality approximation:
    compare prediction error of AR(p) model for effect_series with and without
    cause_series as exogenous. Returns (is_causal, improvement_ratio).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    if len(cause_series) < 30:
        return False, 0.0

    # Build lagged features
    lag = min(max_lag, len(effect_series) // 5)
    if lag < 1:
        return False, 0.0

    n = len(effect_series)
    X_base, X_aug, Y = [], [], []
    for i in range(lag, n):
        if not np.isfinite(effect_series[i]):
            continue
        base_row = [effect_series[i - k] if np.isfinite(effect_series[i - k]) else 0.0 for k in range(1, lag + 1)]
        cause_row = [cause_series[i - k] if np.isfinite(cause_series[i - k]) else 0.0 for k in range(1, lag + 1)]
        X_base.append(base_row)
        X_aug.append(base_row + cause_row)
        Y.append(effect_series[i])

    if len(Y) < 20:
        return False, 0.0

    X_base = np.array(X_base)
    X_aug = np.array(X_aug)
    Y = np.array(Y)

    try:
        m_base = Ridge(alpha=1.0)
        m_aug = Ridge(alpha=1.0)
        score_base = -cross_val_score(m_base, X_base, Y, cv=3, scoring="neg_mean_squared_error").mean()
        score_aug = -cross_val_score(m_aug, X_aug, Y, cv=3, scoring="neg_mean_squared_error").mean()
        improvement = (score_base - score_aug) / (score_base + 1e-10)
        return improvement > 0.05, float(improvement)
    except Exception:
        return False, 0.0


def build_event_causal_chain(
    all_events: List[TemporalEvent],
    entity_series: Dict[str, pd.DataFrame],
    pair_series: Dict[Tuple[str, str], pd.DataFrame],
    fps: float = 10.0,
    max_causal_gap_s: float = 5.0,
) -> List[Dict]:
    """
    For every pair of temporally ordered events (A before B), assess whether
    A causally supports B:

      1. A ends before B starts (temporal precedence)
      2. Kinematic/interaction evidence in the gap supports linkage
      3. Optionally Granger-style predictive check

    Returns a list of causal edge dicts (event-level):
      {source_event_id, target_event_id, relationship, confidence, lag_seconds, evidence}
    """
    edges = []
    evts = sorted(all_events, key=lambda e: e.start_timestamp)

    for i, evt_a in enumerate(evts):
        for evt_b in evts[i + 1:]:
            # Temporal precedence check
            if evt_b.start_timestamp <= evt_a.start_timestamp:
                continue
            gap_s = evt_b.start_timestamp - evt_a.end_timestamp
            if gap_s < 0 or gap_s > max_causal_gap_s:
                continue

            # Collect supporting evidence
            evidence = []
            conf = 0.3  # base

            # 1. Involves overlapping entities → stronger prior
            shared_objects = set(evt_a.object_ids) & set(evt_b.object_ids)
            if shared_objects:
                conf += 0.2
                evidence.append(f"shared_entities:{','.join(sorted(shared_objects))}")

            # 2. Event type logic
            CAUSAL_PAIRS = {
                ("CLOSING_DISTANCE", "CONTACT"): 0.3,
                ("CLOSING_DISTANCE", "COLLISION"): 0.3,
                ("APPROACHING", "CLOSING_DISTANCE"): 0.2,
                ("APPROACHING", "CONTACT"): 0.25,
                ("SUDDEN_BRAKING", "SUDDEN_BRAKING"): 0.15,
                ("SUDDEN_BRAKING", "COLLISION"): 0.2,
                ("CONTACT", "FALL"): 0.3,
                ("CONTACT", "TRAJECTORY_DEVIATION"): 0.25,
                ("COLLISION", "FALL"): 0.35,
                ("COLLISION", "TRAJECTORY_DEVIATION"): 0.25,
                ("CLOSING_DISTANCE", "SUDDEN_BRAKING"): 0.15,
            }
            pair_bonus = CAUSAL_PAIRS.get((evt_a.event_type, evt_b.event_type), 0.0)
            conf += pair_bonus
            if pair_bonus > 0:
                evidence.append(f"event_type_pattern:{evt_a.event_type}→{evt_b.event_type}")

            # 3. Granger check on shared entities' speed
            granger_found = False
            for oid_a in evt_a.object_ids:
                for oid_b in evt_b.object_ids:
                    if oid_a == oid_b or oid_a not in entity_series or oid_b not in entity_series:
                        continue
                    sa = entity_series[oid_a]
                    sb = entity_series[oid_b]
                    # Restrict to window around the events
                    t_start = max(evt_a.start_timestamp - 1.0, sa["timestamp"].min())
                    t_end = min(evt_b.end_timestamp + 1.0, sb["timestamp"].max())
                    wa = sa[(sa["timestamp"] >= t_start) & (sa["timestamp"] <= t_end)]["speed"].to_numpy(dtype=float)
                    wb = sb[(sb["timestamp"] >= t_start) & (sb["timestamp"] <= t_end)]["speed"].to_numpy(dtype=float)
                    min_len = min(len(wa), len(wb))
                    if min_len >= 20:
                        is_c, improvement = _granger_check(wa[:min_len], wb[:min_len], np.arange(min_len) / fps)
                        if is_c:
                            conf += 0.15
                            evidence.append(f"granger:{oid_a}→{oid_b} (improvement={improvement:.3f})")
                            granger_found = True
                            break
                if granger_found:
                    break

            # 4. Check pairwise interaction features
            for (pa, pb), pf in pair_series.items():
                obj_set = {pa, pb}
                if obj_set == set(evt_a.object_ids) or obj_set == set(evt_b.object_ids):
                    # During the gap, is distance decreasing or closing speed positive?
                    window = pf[(pf["timestamp"] >= evt_a.start_timestamp) &
                                (pf["timestamp"] <= evt_b.end_timestamp)]
                    if len(window) > 2:
                        closing = window["closing_speed"].dropna()
                        if len(closing) > 0 and closing.mean() > 0.5:
                            conf += 0.1
                            evidence.append(f"closing_speed_positive:{pa}↔{pb}")

            conf = min(conf, 1.0)
            if conf < 0.35:
                continue

            relationship = "causes" if conf >= 0.65 else ("supports" if conf >= 0.45 else "precedes")

            edges.append({
                "source_event_id": evt_a.event_id,
                "target_event_id": evt_b.event_id,
                "source_type": evt_a.event_type,
                "target_type": evt_b.event_type,
                "source_objects": evt_a.object_ids,
                "target_objects": evt_b.object_ids,
                "relationship": relationship,
                "confidence": round(conf, 4),
                "lag_seconds": round(gap_s, 3),
                "evidence": evidence,
            })

    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Event Onset Localizer
# ─────────────────────────────────────────────────────────────────────────────

class EventOnsetLocalizer:
    """
    Finds the earliest physically-supported transition frame for each detected
    event, distinguishing:

        event_onset    : earliest frame with physical evidence of the transition
        confirmation   : frame at which evidence is strong enough to confirm

    For a collision: onset = first frame of contact (distance < threshold),
    confirmation = when subsequent trajectory deviation / fall is clear.
    """

    def localize(
        self,
        all_events: List[TemporalEvent],
        entity_series: Dict[str, pd.DataFrame],
        pair_series: Dict[Tuple[str, str], pd.DataFrame],
        fps: float = 10.0,
    ) -> Dict[str, Dict]:
        """
        Returns {event_id: {onset_frame, onset_timestamp, confirmation_frame,
                            confirmation_timestamp, onset_reason}}
        """
        results = {}
        for evt in all_events:
            onset, confirm = self._localize_event(evt, all_events, entity_series, pair_series, fps)
            results[evt.event_id] = onset | {"confirmation": confirm}
        return results

    def _localize_event(
        self,
        evt: TemporalEvent,
        all_events: List[TemporalEvent],
        entity_series: Dict[str, pd.DataFrame],
        pair_series: Dict[Tuple[str, str], pd.DataFrame],
        fps: float,
    ) -> Tuple[Dict, Dict]:
        onset = {
            "onset_frame": evt.start_frame,
            "onset_timestamp": evt.start_timestamp,
            "onset_reason": "event_start",
        }
        confirm = {
            "confirmation_frame": evt.end_frame,
            "confirmation_timestamp": evt.end_timestamp,
        }

        # For CONTACT/COLLISION: search backward for first frame where distance
        # fell below threshold — this is the true physical onset
        if evt.event_type in ("CONTACT", "COLLISION", "NEAR_COLLISION"):
            best_frame = evt.start_frame
            best_ts = evt.start_timestamp
            reason = "contact_threshold_crossed"

            for (pa, pb), pf in pair_series.items():
                if not (set([pa, pb]) & set(evt.object_ids)):
                    continue
                pf_sorted = pf.sort_values("frame_id")
                dist_arr = pf_sorted["distance"].to_numpy(dtype=float)
                frame_arr = pf_sorted["frame_id"].to_numpy(dtype=int)
                ts_arr = pf_sorted["timestamp"].to_numpy(dtype=float)

                # Find FIRST frame where distance crossed the threshold (search backward from confirmation)
                thresh = CONTACT_DIST_THRESH if evt.event_type in ("CONTACT", "COLLISION") else NEAR_COLL_DIST_THRESH
                for k in range(len(frame_arr)):
                    if np.isfinite(dist_arr[k]) and dist_arr[k] <= thresh:
                        if ts_arr[k] < best_ts:
                            best_frame = int(frame_arr[k])
                            best_ts = float(ts_arr[k])
                            reason = f"first_dist_below_{thresh}m"
                        break

                # Also search back further: look for first monotonic approach frame
                # (closing_speed becomes positive and distance starts decreasing)
                closing = pf_sorted["closing_speed"].to_numpy(dtype=float)
                dist_rate = pf_sorted["distance_rate"].to_numpy(dtype=float)
                for k in range(len(frame_arr)):
                    if ts_arr[k] >= best_ts:
                        break
                    if (np.isfinite(closing[k]) and closing[k] > CLOSING_SPEED_THRESH and
                            np.isfinite(dist_rate[k]) and dist_rate[k] < -0.5):
                        # This is when vehicles started closing — precursor onset
                        onset["precursor_frame"] = int(frame_arr[k])
                        onset["precursor_timestamp"] = float(ts_arr[k])
                        break

            onset["onset_frame"] = best_frame
            onset["onset_timestamp"] = best_ts
            onset["onset_reason"] = reason

            # Confirmation: look for trajectory deviation or fall by affected objects after contact
            confirm_frame = evt.end_frame
            confirm_ts = evt.end_timestamp
            for follow_evt in sorted(all_events, key=lambda e: e.start_timestamp):
                if follow_evt.event_id == evt.event_id:
                    continue
                if follow_evt.event_type in ("FALL", "TRAJECTORY_DEVIATION", "SUDDEN_STOP"):
                    if set(follow_evt.object_ids) & set(evt.object_ids):
                        if follow_evt.start_timestamp > best_ts:
                            if follow_evt.end_timestamp > confirm_ts:
                                confirm_frame = follow_evt.end_frame
                                confirm_ts = follow_evt.end_timestamp
            confirm["confirmation_frame"] = confirm_frame
            confirm["confirmation_timestamp"] = confirm_ts

        # For SUDDEN_BRAKING: find earliest frame of sustained deceleration
        elif evt.event_type in ("SUDDEN_BRAKING", "SUDDEN_STOP"):
            for oid in evt.object_ids:
                if oid not in entity_series:
                    continue
                se = entity_series[oid]
                se = se[se["frame_id"] <= evt.start_frame + 5].sort_values("frame_id")
                ax_a = se["ax"].to_numpy(dtype=float)
                ay_a = se["ay"].to_numpy(dtype=float)
                vx_a = se["vx"].to_numpy(dtype=float)
                vy_a = se["vy"].to_numpy(dtype=float)
                spd_a = se["speed"].to_numpy(dtype=float)
                ts_a = se["timestamp"].to_numpy(dtype=float)
                fr_a = se["frame_id"].to_numpy(dtype=int)

                for k in range(len(fr_a) - 1, -1, -1):
                    if not (np.isfinite(ax_a[k]) and np.isfinite(spd_a[k]) and spd_a[k] > 0.1):
                        continue
                    lon_acc = (ax_a[k] * vx_a[k] + ay_a[k] * vy_a[k]) / spd_a[k] if spd_a[k] > 0.1 else 0.0
                    if lon_acc >= BRAKING_ACCEL_THRESH:
                        # Frame after this was first deceleration
                        actual_start = k + 1
                        if actual_start < len(fr_a):
                            onset["onset_frame"] = int(fr_a[actual_start])
                            onset["onset_timestamp"] = float(ts_a[actual_start])
                            onset["onset_reason"] = "first_deceleration_frame"
                        break

        return onset, confirm


# ─────────────────────────────────────────────────────────────────────────────
# Explanation generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(
    all_events: List[TemporalEvent],
    event_edges: List[Dict],
    onset_map: Dict[str, Dict],
    entity_series: Dict[str, pd.DataFrame],
) -> str:
    """
    Generate a human-readable explanation of the causal chain, traceable to graph evidence.
    Does not invent facts — only uses events and edges that exist.
    """
    if not all_events:
        return "Insufficient kinematic data to reconstruct an event sequence."

    # Sort events by onset time
    sorted_events = sorted(all_events, key=lambda e: onset_map.get(e.event_id, {}).get("onset_timestamp", e.start_timestamp))
    edges_by_target = {}
    for edge in event_edges:
        tid = edge["target_event_id"]
        if tid not in edges_by_target:
            edges_by_target[tid] = []
        edges_by_target[tid].append(edge)

    # Build entity class map
    entity_class = {}
    for oid, sdf in entity_series.items():
        cls_col = sdf["class"] if "class" in sdf.columns else None
        if cls_col is not None and not cls_col.dropna().empty:
            entity_class[oid] = str(cls_col.dropna().iloc[0])

    sentences = []

    for evt in sorted_events:
        onset_info = onset_map.get(evt.event_id, {})
        t_onset = onset_info.get("onset_timestamp", evt.start_timestamp)
        t_confirm = onset_info.get("confirmation", {}).get("confirmation_timestamp", evt.end_timestamp)

        obj_labels = []
        for oid in evt.object_ids:
            cls = entity_class.get(oid, "vehicle")
            obj_labels.append(f"{cls} {oid}")
        obj_str = " and ".join(obj_labels)

        ev_type_human = evt.event_type.replace("_", " ").title()

        if evt.event_type == "APPROACHING":
            sentences.append(
                f"{obj_str} were approaching each other (from t={t_onset:.1f}s)."
            )
        elif evt.event_type == "CLOSING_DISTANCE":
            d_start = evt.features.get("distance_start")
            d_end = evt.features.get("distance_end")
            d_str = ""
            if d_start and d_end:
                d_str = f" (gap reduced from {d_start:.1f}m to {d_end:.1f}m)"
            sentences.append(
                f"The distance between {obj_str} was closing rapidly{d_str} from t={t_onset:.1f}s."
            )
        elif evt.event_type in ("CONTACT", "COLLISION"):
            impact = "Physical contact" if evt.event_type == "CONTACT" else "A collision"
            sig = "with" if evt.features.get("has_impact_signature") else "without"
            sentences.append(
                f"{impact} between {obj_str} was detected at approximately t={t_onset:.1f}s "
                f"({sig} a sharp impact signature). "
                f"Sufficient evidence accumulated by t={t_confirm:.1f}s."
            )
        elif evt.event_type == "SUDDEN_BRAKING":
            sentences.append(
                f"{obj_str} underwent sudden braking at t={t_onset:.1f}s."
            )
        elif evt.event_type == "FALL":
            sentences.append(
                f"{obj_str} experienced a rapid speed reduction consistent with a fall at t={t_onset:.1f}s."
            )
        elif evt.event_type == "TRAJECTORY_DEVIATION":
            sentences.append(
                f"{obj_str} deviated from their trajectory at t={t_onset:.1f}s."
            )
        else:
            sentences.append(
                f"{ev_type_human} involving {obj_str} was detected at t={t_onset:.1f}s."
            )

        # Append causal predecessor info
        predecessors = edges_by_target.get(evt.event_id, [])
        for pred_edge in predecessors:
            if pred_edge["relationship"] in ("causes", "supports"):
                src_evt = next((e for e in all_events if e.event_id == pred_edge["source_event_id"]), None)
                if src_evt:
                    rel = "was caused by" if pred_edge["relationship"] == "causes" else "may have been influenced by"
                    sentences[-1] += (
                        f" This {rel} the preceding {src_evt.event_type.lower().replace('_', ' ')} "
                        f"at t={src_evt.start_timestamp:.1f}s "
                        f"(confidence={pred_edge['confidence']:.2f})."
                    )

    return " ".join(sentences) if sentences else "No clear causal sequence could be established."
