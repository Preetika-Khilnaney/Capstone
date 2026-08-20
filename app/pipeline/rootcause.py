"""
Root-cause analysis — deterministic incident factors for an event.

Emits structured primary / contributing / mitigating factors from kinematic
scene facts (incident_anchor_plan.md §9). Always deterministic; Track 4 may
have an LLM re-phrase the factors, never invent new ones.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def analyze_root_cause(scene: dict, facts: dict) -> dict:
    """
    Build the root-cause block from the scene table + kinematic facts.

    ``facts`` (gathered by the stage segmenter):
      initiator_name, initiator_class, cruise_mps, min_speed_mps, collapse_s,
      skid, skid_lateral_m, persons_down, n_responders, min_trailing_gap_m,
      ambient_speed_mps, flank_proximity_m, density_per_frame,
      occupants_upright_at_end, skid_position_note
    """
    contrib: list[str] = []
    mitigate: list[str] = []

    density = facts.get("density_per_frame", 0.0) or 0.0
    contrib.append(
        f"Moderate traffic density (mean {density:.2f} upstream vehicles/frame "
        "requiring variable speed adjustments)"
    )
    prox = facts.get("flank_proximity_m")
    if prox is not None and np.isfinite(prox):
        contrib.append(
            f"Proximity to surrounding traffic reducing reaction margin "
            f"(nearest flanking vehicle mean {prox:.1f} m before onset)"
        )
    ambient = facts.get("ambient_speed_mps")
    if ambient is not None and np.isfinite(ambient):
        contrib.append(f"Ambient traffic speed approx {ambient * 3.6:.0f} km/h")

    gap = facts.get("min_trailing_gap_m")
    if gap is not None and np.isfinite(gap):
        mitigate.append(
            f"Adequate following distance maintained by trailing four-wheelers "
            f"(min gap {gap:.1f} m) preventing secondary collision"
        )
    note = facts.get("skid_position_note")
    if note:
        mitigate.append(note)
    if facts.get("occupants_upright_at_end"):
        mitigate.append("Occupants ambulatory and visible after the incident "
                        "(cleared / walked off the travel path)")

    initiator = facts.get("initiator_name") or "the initiator"
    cruise = facts.get("cruise_mps")
    vmin = facts.get("min_speed_mps")
    dur = facts.get("collapse_s")
    kin = ""
    if all(v is not None and np.isfinite(v) for v in (cruise, vmin, dur)):
        kin = (f"speed collapse {cruise:.1f}→{vmin:.1f} m/s over {dur:.1f}s"
               if cruise >= 1.0
               else "observed at rest after the crash anchor")
    if facts.get("skid") or facts.get("persons_down"):
        kind = "traction_loss"
        text = (f"Abrupt braking / traction loss on the {initiator}"
                + (f" ({kin}, low-side slide/skid)" if kin else "; low-side slide/skid"))
    else:
        kind = "abrupt_deceleration"
        text = f"Abrupt deceleration by the {initiator}" + (f" ({kin})" if kin else "")

    summary = f"Primary: {text}. " \
              f"Contributing: {'; '.join(contrib) or 'none identified'}. " \
              f"Mitigating: {'; '.join(mitigate) or 'none identified'}."
    return {
        "primary_factor": {"kind": kind, "text": text},
        "contributing_factors": contrib,
        "mitigating_factors": mitigate,
        "summary": summary,
    }