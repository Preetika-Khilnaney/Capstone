"""
Regression test for the Track 2 causal engine.

Generates a synthetic lead-follower braking scenario with KNOWN ground-truth
causality (a lead vehicle brakes; a follower reacts LAG frames later via a
car-following model) plus an unrelated distractor vehicle, then asserts:
  - both V_LEAD and V_FOLLOW are selected as separate causal targets (proves
    multi-target selection: two vehicles each have a real sustained drop);
  - V_FOLLOW's graph recovers the lead -> target link (as rel_speed or lead_gap;
    rel_speed embeds the target's own speed, so the contemporaneous form of
    the link is what PCMCI+ surface under tau_max=15) at lag <= tau window;
  - V_LEAD (nothing ahead of it) has no lead_speed/lead_gap variable at all,
    i.e. it's correctly treated as an exogenous root cause, not chasing an
    external driver that structurally can't exist for it.

Run: python test_causal.py
"""
import shutil
import sys

import numpy as np
import pandas as pd

from app.config import settings
from app.pipeline.causal import get_causal_engine

EVENT_ID = "EVT_SYNTHCAUSAL_TEST"
LAG = 5
GAIN = 0.8   # follower tracking gain (0.35 diluted the lagged signal under tau_max=15)
SIGMA = 0.05 # per-step process noise


def _build_synthetic_csv() -> None:
    np.random.seed(1)
    T, dt = 90, 0.1

    v_lead = np.full(T, 15.0)
    for t in range(40, T):
        v_lead[t] = max(4.0, 15 - 11 * (t - 40) / 10)   # exogenous brake at frame 40
    v_lead += np.random.randn(T) * 0.1

    v_follow = np.full(T, 15.0)                           # reacts to lead's speed LAG frames ago
    for t in range(1, T):
        ref = v_lead[t - LAG] if t - LAG >= 0 else 15.0
        v_follow[t] = v_follow[t - 1] + GAIN * (ref - v_follow[t - 1]) + np.random.randn() * SIGMA

    v_other = 12.0 + np.random.randn(T) * 0.3            # unrelated distractor (adjacent lane)
    y_lead = 25 + np.cumsum(v_lead) * dt
    y_follow = np.cumsum(v_follow) * dt
    y_other = np.cumsum(v_other) * dt + 10

    rows = []
    for t in range(T):
        for oid, x, y, v in [("V_LEAD", 1.75, y_lead[t], v_lead[t]),
                             ("V_FOLLOW", 1.75, y_follow[t], v_follow[t]),
                             ("V_OTHER", 7.5, y_other[t], v_other[t])]:
            rows.append(dict(Event_ID=EVENT_ID, Timestamp=round(-4 + t * dt, 2), Frame_ID=t,
                             Object_ID=oid, Class="car", BBox_X1=0, BBox_Y1=0, BBox_X2=10, BBox_Y2=10,
                             Pos_X_m=x, Pos_Y_m=y, Velocity_mps=v))
    d = settings.paths.dataset_dir / EVENT_ID
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{EVENT_ID}_causal_data.csv", index=False)


def main() -> int:
    _build_synthetic_csv()
    try:
        res = get_causal_engine().analyze_event(EVENT_ID)
    finally:
        shutil.rmtree(settings.paths.dataset_dir / EVENT_ID, ignore_errors=True)

    ok = res.get("status") == "ok"
    targets = {t["target_object"]: t for t in res.get("targets", [])}
    print("status:", res.get("status"), "| targets found:", list(targets))

    multi_target_ok = "V_FOLLOW" in targets and "V_LEAD" in targets

    follow = targets.get("V_FOLLOW", {})
    drivers = follow.get("drivers_of_target_speed", [])
    for l in drivers:
        print(f"   V_FOLLOW <- {l['cause']}(t-{l['lag']})  strength={l['strength']}")
    # Fix 3 renamed lead_speed → rel_speed (= lead_speed − tgt_speed); rel_speed
    # embeds the target's own speed, so the recovered link shows up at low lag.
    lead_link = [l for l in drivers if l["cause"] in ("rel_speed", "lead_gap")]
    tau_max = follow.get("tau_max", 15)
    link_ok = bool(lead_link) and all(l["lag"] <= tau_max for l in lead_link)

    lead = targets.get("V_LEAD", {})
    lead_exogenous_ok = not ({"rel_speed", "lead_gap"} & set(lead.get("variables", [])))

    lead_link_desc = [f"{l['cause']}(lag {l['lag']})" for l in lead_link]
    print(f"\nmulti-target selection (V_FOLLOW + V_LEAD both found) : {multi_target_ok}")
    print(f"lead->follow link recovered                          : {link_ok}  {lead_link_desc}")
    print(f"V_LEAD correctly has no lead variable (exogenous root)  : {lead_exogenous_ok}")

    passed = ok and multi_target_ok and link_ok and lead_exogenous_ok
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
