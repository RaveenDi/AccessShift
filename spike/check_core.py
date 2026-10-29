"""Finite-cell planning probe, not measured camera geometry or accessibility proof."""
from collections import deque
from copy import deepcopy
import json

WIDTH, HEIGHT = 12, 10
START, END = (1, 5), (10, 5)


def cells(obj):
    x, y, w, h = obj["rect"]
    if any(type(v) is not int for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Expected positive integer cell dimensions")
    return {(i, j) for i in range(x, x + w) for j in range(y, y + h)}


def occupied(layout):
    result = set()
    for obj in layout.values():
        footprint = cells(obj)
        if any(not (0 <= x < WIDTH and 0 <= y < HEIGHT) for x, y in footprint):
            raise ValueError("Invalid board placement")
        if result & footprint:
            raise ValueError("Overlapping footprints")
        result |= footprint
    return result


def route(layout):
    blocked = occupied(layout)
    if START in blocked or END in blocked:
        return None
    queue, previous = deque([START]), {START: None}
    while queue:
        point = queue.popleft()
        if point == END:
            path = []
            while point is not None:
                path.append(point)
                point = previous[point]
            return list(reversed(path))
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxt = (point[0] + dx, point[1] + dy)
            if 0 <= nxt[0] < WIDTH and 0 <= nxt[1] < HEIGHT and nxt not in blocked and nxt not in previous:
                previous[nxt] = point
                queue.append(nxt)
    return None


def propose(layout, allowed):
    """Exhaust zero and one move only, within the explicit candidate list."""
    if route(layout):
        return {"layout": deepcopy(layout), "moves": [], "cost": (0, 0)}
    best = None
    for identifier, poses in allowed.items():
        if layout[identifier]["locked"]:
            continue
        for x, y in poses:
            trial = deepcopy(layout)
            ox, oy, w, h = trial[identifier]["rect"]
            trial[identifier]["rect"] = [x, y, w, h]
            try:
                witness = route(trial)
            except ValueError:
                continue
            if not witness:
                continue
            cost = (1, abs(x - ox) + abs(y - oy))
            if best is None or cost < best["cost"]:
                best = {"layout": trial, "moves": [identifier], "cost": cost}
    return best


def main():
    scene = {
        "wall_top": {"rect": [5, 0, 1, 5], "locked": True},
        "wall_bottom": {"rect": [5, 6, 1, 4], "locked": True},
        "plug": {"rect": [5, 5, 1, 1], "locked": False},
        "desk": {"rect": [1, 0, 2, 1], "locked": True},
        "stall": {"rect": [10, 9, 2, 1], "locked": True},
        "crate": {"rect": [7, 8, 1, 1], "locked": True},
    }
    allowed = {"plug": [(5, 5), (8, 8), (7, 8), (1, 5), (15, 1)]}
    plan = propose(scene, allowed)
    locked = deepcopy(scene)
    locked["plug"]["locked"] = True
    wrong = deepcopy(plan["layout"])
    wrong["plug"]["rect"] = [8, 7, 1, 1]
    checks = {
        "full_barrier_blocks_route": route(scene) is None,
        "one_move_opens_route": plan["moves"] == ["plug"] and route(plan["layout"]) is not None,
        "zero_moves_preferred_when_open": propose(plan["layout"], allowed)["cost"] == (0, 0),
        "locked_objects_preserved": all(scene[k] == plan["layout"][k] for k in scene if scene[k]["locked"]),
        "locked_blocker_no_solution_in_declared_set": propose(locked, allowed) is None,
        "invalid_candidates_not_selected": plan["layout"]["plug"]["rect"][:2] == [8, 8],
        "wrong_after_state_detected_even_if_route_open": wrong != plan["layout"] and route(wrong) is not None,
        "input_scene_not_mutated": scene["plug"]["rect"] == [5, 5, 1, 1],
    }
    print(json.dumps({"kind": "synthetic_discrete_grid_only", "checks": checks,
                      "passed": sum(checks.values()), "total": len(checks),
                      "candidate_move_limit": 1, "model_tested": False,
                      "physical_measurement_tested": False}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
