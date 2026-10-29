"""Bounded rearrangement search over a declared finite candidate set.

Candidates: the object's centre on a grid of `grid_step_cm`, same orientation, inside the board,
not overlapping any object that stays put. Locked objects never move. Zero, then one, then two
moves are enumerated; within a move count the smallest total centre displacement wins.
Optimality is claimed only for the enumerated set and only when the enumeration finished.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import box

from .connectivity import Grid, blocked_mask, boundary_mask, check_route, connected, layout_margin, raster_poly
from .footprints import Footprint


@dataclass
class Move:
    id: str
    label: str
    from_center: tuple[float, float]
    to_center: tuple[float, float]

    @property
    def displacement_cm(self) -> float:
        return math.dist(self.from_center, self.to_center)

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label,
                "from_cm": [round(v, 1) for v in self.from_center],
                "to_cm": [round(v, 1) for v in self.to_center],
                "displacement_cm": round(self.displacement_cm, 1)}


@dataclass
class Plan:
    status: str  # already_open | plan_found | plan_found_search_incomplete | no_solution_in_set | search_incomplete
    moves: list[Move] = field(default_factory=list)
    layout: list[Footprint] = field(default_factory=list)  # expected layout after the moves
    complete: bool = True
    evaluated: dict = field(default_factory=dict)
    candidate_set: dict = field(default_factory=dict)
    route_before: dict | None = None
    route_after: dict | None = None
    input_hash: str = ""
    message: str = ""

    @property
    def total_displacement_cm(self) -> float:
        return sum(m.displacement_cm for m in self.moves)

    def to_dict(self, cal=None) -> dict:
        return {"status": self.status, "message": self.message, "n_moves": len(self.moves),
                "total_displacement_cm": round(self.total_displacement_cm, 1),
                "moves": [m.to_dict() for m in self.moves], "search_complete": self.complete,
                "minimal_within_candidate_set": self.complete and self.status == "plan_found",
                "evaluated": self.evaluated, "candidate_set": self.candidate_set,
                "route_before": self.route_before, "route_after": self.route_after,
                "layout_after": [f.to_dict(cal) for f in self.layout], "input_hash": self.input_hash}


def input_hash(footprints, start, end, clearance_cm, extra=None) -> str:
    payload = {"fp": [(f.id, f.locked, [round(c, 2) for xy in f.polygon.exterior.coords for c in xy])
                      for f in footprints], "s": list(start), "e": list(end), "c": clearance_cm, "x": extra}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _overlap(a, b) -> bool:
    ab, bb = a.bounds, b.bounds
    if ab[2] <= bb[0] or bb[2] <= ab[0] or ab[3] <= bb[1] or bb[3] <= ab[1]:
        return False
    return a.intersection(b).area > 0.05


def plan_moves(footprints: list[Footprint], board_w: float, board_h: float, start, end, clearance_cm: float,
               max_moves: int = 2, grid_step_cm: float = 10.0, time_budget_s: float = 8.0,
               cell: float = 5.0) -> Plan:
    t0 = time.monotonic()
    ids = [f.id for f in footprints]
    assert len(set(ids)) == len(ids), "footprint ids must be unique"
    ihash = input_hash(footprints, start, end, clearance_cm, [max_moves, grid_step_cm])
    before = check_route(footprints, board_w, board_h, start, end, clearance_cm, cell=cell)
    movable = [f for f in footprints if not f.locked]
    cset = {"grid_step_cm": grid_step_cm, "max_moves": max_moves, "orientation": "unchanged",
            "movable_ids": [f.id for f in movable], "locked_ids": [f.id for f in footprints if f.locked],
            "bounds": "object fully inside board", "time_budget_s": time_budget_s}
    plan = Plan("no_solution_in_set", route_before=before.to_dict(), candidate_set=cset, input_hash=ihash)
    if before.status == "open":
        plan.status, plan.layout, plan.route_after = "already_open", list(footprints), before.to_dict()
        plan.message = "Route already open at this clearance; no move needed."
        return plan

    grid = Grid(board_w, board_h, cell)
    m = layout_margin(footprints, cell)
    rc = clearance_cm / 2.0 + m
    s, e = grid.cell_of(start), grid.cell_of(end)
    bound = boundary_mask(grid, rc)
    base = {f.id: raster_poly(f.polygon, rc, grid) for f in footprints}
    by_id = {f.id: f for f in footprints}
    board = box(0, 0, board_w, board_h)
    movable_ids = {f.id for f in movable}
    deadline = t0 + time_budget_s

    # raw candidates per movable object: inside the board, not overlapping any object that cannot move
    xs = np.arange(grid_step_cm, board_w, grid_step_cm)
    ys = np.arange(grid_step_cm, board_h, grid_step_cm)
    raw: dict[str, list] = {}
    for f in movable:
        others = [o.polygon for o in footprints if o.id not in movable_ids]
        lst = []
        for x in xs:
            for y in ys:
                c = f.moved_to(float(x), float(y))
                if not board.contains(c.polygon):
                    continue
                if math.dist(f.center, (x, y)) < 1.0:
                    continue
                if any(_overlap(c.polygon, o) for o in others):
                    continue
                lst.append((float(x), float(y), c.polygon))
        raw[f.id] = lst
    mask_cache: dict[tuple, np.ndarray] = {}

    def cmask(fid, x, y, poly):
        k = (fid, x, y)
        if k not in mask_cache:
            mask_cache[k] = raster_poly(poly, rc, grid)
        return mask_cache[k]

    timed_out = False
    complete_levels = 0
    for k in range(1, max_moves + 1):
        best = None  # (cost, moves tuple)
        n_eval = 0
        level_complete = True
        for combo in itertools.combinations([f.id for f in movable], k):
            stay = [o for o in movable_ids if o not in combo]
            fixed = bound.copy()
            for f in footprints:
                if f.id not in combo:
                    fixed |= base[f.id]
            cand_lists = []
            for fid in combo:
                lst = [c for c in raw[fid] if not any(_overlap(c[2], by_id[o].polygon) for o in stay)]
                cand_lists.append(lst)
            for choice in itertools.product(*cand_lists):
                n_eval += 1
                if (n_eval & 255) == 0 and time.monotonic() > deadline:
                    timed_out, level_complete = True, False
                    break
                if k == 2 and _overlap(choice[0][2], choice[1][2]):
                    continue
                blk = fixed
                for fid, c in zip(combo, choice):
                    blk = blk | cmask(fid, c[0], c[1], c[2])
                if not connected(blk, s, e):
                    continue
                cost = sum(math.dist(by_id[fid].center, (c[0], c[1])) for fid, c in zip(combo, choice))
                key = (round(cost, 6), combo, tuple((c[0], c[1]) for c in choice))
                if best is None or key < best[0]:
                    best = (key, combo, choice)
            if timed_out:
                break
        plan.evaluated[f"{k}_move"] = n_eval
        if best is not None:
            _, combo, choice = best
            new = {f.id: f for f in footprints}
            moves = []
            for fid, c in zip(combo, choice):
                new[fid] = by_id[fid].moved_to(c[0], c[1])
                moves.append(Move(fid, by_id[fid].label, by_id[fid].center, (c[0], c[1])))
            plan.layout = [new[i] for i in ids]
            plan.moves = moves
            after = check_route(plan.layout, board_w, board_h, start, end, clearance_cm, cell=cell)
            plan.route_after = after.to_dict()
            plan.complete = level_complete and complete_levels == k - 1
            plan.status = "plan_found" if plan.complete else "plan_found_search_incomplete"
            plan.message = (f"{k} move(s), minimal total displacement within the declared set."
                            if plan.complete else
                            "Plan found but the search hit its time budget: not claimed minimal.")
            plan.evaluated["seconds"] = round(time.monotonic() - t0, 2)
            return plan
        if not level_complete:
            break
        complete_levels = k
    plan.evaluated["seconds"] = round(time.monotonic() - t0, 2)
    plan.layout = list(footprints)
    if timed_out:
        plan.status, plan.complete = "search_incomplete", False
        plan.message = "Search incomplete: time budget reached before the candidate set was exhausted."
    else:
        plan.status = "no_solution_in_set"
        plan.message = (f"No solution within this candidate set (<= {max_moves} moves, grid "
                        f"{grid_step_cm} cm, locks respected). This does not prove none exists.")
    return plan
