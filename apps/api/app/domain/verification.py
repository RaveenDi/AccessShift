"""Compare recaptured (independently perceived) poses to the plan, then recompute the route."""
from __future__ import annotations

import math

from .connectivity import check_route
from .footprints import Footprint
from .search import Plan


def _iou(a, b) -> float:
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 0.0


def verify(before: list[Footprint], plan: Plan, observed: list[Footprint], board_w: float, board_h: float,
           start, end, clearance_cm: float, tol_cm: float = 6.0, min_iou: float = 0.7) -> dict:
    """Identity is by unique id/label; duplicates or missing objects are reported, not guessed."""
    expected = {f.id: f for f in plan.layout}
    moved_ids = {m.id for m in plan.moves}
    before_by = {f.id: f for f in before}
    obs = {}
    dup = []
    for f in observed:
        if f.id in obs:
            dup.append(f.id)
        obs[f.id] = f
    missing = [i for i in expected if i not in obs]
    extra = [i for i in obs if i not in expected]
    objects = []
    problems = []
    for fid, exp in expected.items():
        if fid not in obs:
            continue
        o = obs[fid]
        d_exp = math.dist(o.center, exp.center)
        iou = _iou(o.polygon, exp.polygon)
        d_old = math.dist(o.center, before_by[fid].center)
        ok = d_exp <= tol_cm and iou >= min_iou
        if fid in moved_ids:
            kind = "moved_as_planned" if ok else ("not_moved" if d_old <= tol_cm else "moved_to_wrong_place")
        else:
            kind = "unchanged" if ok else "unexpected_move"
        if kind not in ("moved_as_planned", "unchanged"):
            problems.append(f"{fid}: {kind} ({d_exp:.1f} cm from expected)")
        objects.append({"id": fid, "kind": kind, "deviation_cm": round(d_exp, 1), "iou": round(iou, 3),
                        "ok": kind in ("moved_as_planned", "unchanged"),
                        "observed_center_cm": [round(v, 1) for v in o.center],
                        "expected_center_cm": [round(v, 1) for v in exp.center]})
    route = check_route(observed, board_w, board_h, start, end, clearance_cm) if not (missing or dup) else None
    if missing or dup or extra:
        verdict = "needs_remeasurement"
        problems.append(f"identity problem: missing={missing} duplicate={dup} unexpected={extra}")
    elif route.status != "open":
        verdict = "route_still_blocked" if route.status == "blocked" else "route_borderline"
    elif problems:
        verdict = "route_open_but_deviates_from_plan"
    else:
        verdict = "verified_open"
    return {"verdict": verdict, "pass": verdict == "verified_open", "problems": problems,
            "objects": objects, "tolerance_cm": tol_cm, "min_iou": min_iou,
            "route": route.to_dict() if route else None,
            "scope": "calibrated tabletop model only; not accessibility compliance"}
