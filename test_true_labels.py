"""
Meaning-based eval harness: does the pipeline's episode block match testVideo2TrueLabels.json?

Compares ONLY the stable meaning of the gold labels (entity roles, stage order and
boundaries, typed relations, root-cause primary factor) against the deterministic
episode produced from a real run's kinematics — no LLM, no SigLIP, no API server.

Usage:
    python test_true_labels.py                # latest registry event whose source is testVideo2
    python test_true_labels.py EVT_XXXXXXXX   # explicit event id

Gold timestamps are absolute video seconds; the episode windows are clip-relative
(t=0 = incident anchor). Boundary comparison aligns the produced N3 (Incident)
window start to the gold N3 start and applies STAGE_BOUNDARY_TOL_S tolerance.

Run: python test_true_labels.py
"""
import json
import re
import shutil
import sys

from app.config import settings
from app.pipeline.stage import build_episode

GOLD_PATH = settings.paths.base_dir / "testVideo2TrueLabels.json"
STAGE_BOUNDARY_TOL_S = float(__import__("os").environ.get("STAGE_BOUNDARY_TOL_S", "1.5"))

STAGE_ORDER = ["Stable", "Trigger / Critical", "Incident", "Hazard Response", "Recovery"]  # gold order
PRODUCED_KEYWORDS = {
    "Stable": ["Stable"],
    "Trigger / Critical": ["Trigger"],
    "Incident": ["Incident"],
    "Hazard Response": ["Hazard"],
    "Recovery": ["Recovery"],
}
FOUR_WHEELERS = {"car", "truck", "bus"}
ROOT_CAUSE_KEYWORDS = ("abrupt", "decelerat", "traction", "brak")


def _latest_testvideo2_event() -> str | None:
    """Most recent registry event whose source video mentions testVideo2."""
    import sqlite3
    conn = sqlite3.connect(str(settings.paths.db_path), timeout=10)
    try:
        row = conn.execute(
            "SELECT Event_ID FROM Master_Event_Log "
            "WHERE Source_Video_Path LIKE '%testVideo2%' OR Raw_Video_Path LIKE '%testVideo2%' "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _gold_seconds(ts: str) -> float:
    m, s = re.match(r"(\d+):(\d+)", ts).groups()
    return float(m) * 60 + float(s)


def _stage_start(episode: dict, node_id: str) -> float | None:
    for n in episode.get("nodes", []):
        if n.get("node_id") == node_id:
            w = n.get("window_s")
            return float(w[0]) if w else None
    return None


def main() -> int:
    event_id = sys.argv[1] if len(sys.argv) > 1 else _latest_testvideo2_event()
    if not event_id:
        print("No event id given and no testVideo2 event in the registry.")
        return 1

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    episode = build_episode(event_id)
    if episode is None:
        print(f"RESULT: FAIL — no episode produced for {event_id} (missing kinematics?)")
        return 1

    nodes_by = {n["node_id"]: n for n in episode.get("nodes", [])}
    gold_starts = {n["node_id"]: _gold_seconds(n["timestamp"].split("-")[0].strip())
                   for n in gold["nodes"]}

    passed = True
    fails: list[str] = []

    # 1. entity roles present
    ents = episode.get("entities", [])
    roles = [e.get("role") for e in ents]
    has_moto = any(e.get("class") == "motorcycle" or e.get("role") == "initiator" for e in ents)
    has_rider = any(r in roles for r in ("rider", "pillion")) and \
        any(e.get("class") == "person" for e in ents)
    trailing = [e for e in ents if e.get("role") == "trailing"
                and e.get("class") in FOUR_WHEELERS]
    print(f"motorcycle present                    : {has_moto}")
    print(f"rider/pillion present                 : {has_rider}")
    print(f"trailing 4-wheelers ({len(trailing)}): "
          f"{[e.get('name') or e.get('id') for e in trailing]}")
    for cond, msg in ((has_moto, "no motorcycle/initiator entity"),
                      (has_rider, "no rider/pillion entity"),
                      (len(trailing) >= 2, "fewer than 2 trailing 4-wheelers")):
        if not cond:
            fails.append(msg)
            passed = False

    # 2. stage ordering + boundaries
    produced_order = [n.get("state") for n in episode.get("nodes", [])]
    ok_order = all(any(kw in (s or "") for kw in kws) for s, kws in
                   zip(produced_order, PRODUCED_KEYWORDS.values()))
    idx = 0
    subseq = True
    for s in produced_order:
        for j in range(idx, len(STAGE_ORDER)):
            if any(kw in (s or "") for kw in PRODUCED_KEYWORDS[STAGE_ORDER[j]]):
                idx = j + 1
                break
        else:
            subseq = False
    ok_order = subseq and len(produced_order) >= 3
    if not ok_order:
        fails.append(f"stage order not a subsequence of gold order: {produced_order}")
        passed = False
    print(f"stage order (subsequence check)       : {ok_order}  {produced_order}")

    align = _stage_start(episode, "N3")
    boundary_ok = True
    if align is None:
        boundary_ok = False
        fails.append("no N3 (Incident) node — cannot align boundaries")
        passed = False
    else:
        offset = gold_starts["N3"] - align
        for nid, gs in gold_starts.items():
            pstart = _stage_start(episode, nid)
            if pstart is None:
                continue
            dev = abs(pstart + offset - gs)
            status = "ok" if dev <= STAGE_BOUNDARY_TOL_S else "FAIL"
            if status == "FAIL":
                boundary_ok = False
                passed = False
                fails.append(f"{nid} boundary off by {dev:.1f}s (gold {gs:.0f}s, produced {pstart + offset:.1f}s)")
            print(f"  {nid} boundary vs gold   : {status}  (dev {dev:.1f}s, tol {STAGE_BOUNDARY_TOL_S}s)")
    if boundary_ok:
        print("stage boundaries within tolerance      : True")

    # 3. typed relations
    rels = {(r.get("source_node"), r.get("relation_type")) for r in episode.get("relations", [])}
    want = {("N2", "DIRECT_CAUSE"), ("N3", "TRIGGERED_RESPONSE"), ("N3", "CONSEQUENCE")}
    missing = want - rels
    ok_rels = not missing
    if not ok_rels:
        fails.append(f"missing relations: {sorted(missing)}")
        passed = False
    print(f"typed relations (3/3)                 : {ok_rels}  {sorted(r.get('relation_type') for r in episode.get('relations', []))}")

    # 4. root-cause primary alignment
    rc = episode.get("root_cause", {})
    pf = rc.get("primary_factor", {})
    text = f"{pf.get('kind', '')} {pf.get('text', '')}".lower()
    ok_rc = any(kw in text for kw in ROOT_CAUSE_KEYWORDS)
    if not ok_rc:
        fails.append(f"root cause '{text}' does not mention abrupt deceleration/traction loss")
        passed = False
    print(f"root-cause primary alignment           : {ok_rc}  ({pf.get('kind')}: {pf.get('text')})")

    print()
    for f in fails:
        print("  -", f)
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())