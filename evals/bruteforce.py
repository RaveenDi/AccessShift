"""Independent brute-force check of 'how many moves are needed', used to grade the agent's plan size.

It does not call plan_moves. It enumerates single-object placements on a coarse grid through the public
check_route. Exact for 0 and 1 moves on that grid; for anything else it can only say 'more than one move (or
impossible) on this grid', returned as None. Coarse on purpose: it is a cross-check, not the planner.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

from app.domain.connectivity import check_route  # noqa: E402


def min_moves_up_to_one(fps, board_w, board_h, start, end, clearance_cm, locked_ids=(), step_cm=40.0):
    """0 if already open, 1 if some single unlocked object placement opens it, else None."""
    if check_route(fps, board_w, board_h, start, end, clearance_cm).status == "open":
        return 0
    board = box(0, 0, board_w, board_h)
    for f in fps:
        if f.locked or f.id in locked_ids:
            continue
        for x in np.arange(step_cm / 2, board_w, step_cm):
            for y in np.arange(step_cm / 2, board_h, step_cm):
                c = f.moved_to(float(x), float(y))
                if not board.contains(c.polygon):
                    continue
                if any(o.id != f.id and o.polygon.intersection(c.polygon).area > 0.05 for o in fps):
                    continue
                trial = [c if o.id == f.id else o for o in fps]
                if check_route(trial, board_w, board_h, start, end, clearance_cm).status == "open":
                    return 1
    return None
