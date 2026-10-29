import math

import pytest

from app.domain.calibration import CalibrationError, calibrate
from app.domain.connectivity import check_route
from app.domain.footprints import rect_footprint as R
from app.domain.search import plan_moves

W, H = 500.0, 350.0
S, E = (60, 175), (440, 180)
IMG = (1200, 900)
GOOD = [(100, 100), (1100, 100), (1100, 800), (100, 800)]  # 2 px/cm
CHK = [(300, 760), (700, 760)]  # 200 cm


def scene(arm=(230, 145), extra=()):
    fps = [R("sofa", "sofa", 160, 20, 180, 90, True), R("shelf", "shelf", 160, 240, 180, 50, True),
           R("arm", "arm", arm[0], arm[1], 70, 70)]
    return fps + list(extra)


def test_calibration_accepts_good_and_maps_cm():
    cal = calibrate(GOOD, W, H, CHK, 200, IMG)
    assert cal.check_error_cm < 0.01
    assert cal.to_cm([(600, 450)])[0].tolist() == pytest.approx([250, 175])


@pytest.mark.parametrize("corners,chk,L", [
    ([(100, 100), (600, 100), (1100, 100), (100, 800)], CHK, 200),  # collinear/degenerate
    ([(100, 100), (1100, 800), (1100, 100), (100, 800)], CHK, 200),  # bow-tie ordering
    ([(100, 100), (1100, 120), (1000, 800), (900, 780)], CHK, 200),  # ill-conditioned quad
    (GOOD, CHK, 150),  # independent length disagrees
    ([(100, 100), (160, 100), (160, 140), (100, 140)], CHK, 200),  # board tiny in image
])
def test_calibration_rejects(corners, chk, L):
    with pytest.raises(CalibrationError):
        calibrate(corners, W, H, chk, L, IMG)


def test_connectivity_blocked_open_borderline():
    assert check_route(scene(), W, H, S, E, 90).status == "blocked"
    r = check_route(scene(arm=(40, 40)), W, H, S, E, 90)
    assert r.status == "open" and r.witness_cm and r.witness_clearance_cm >= 45
    # corridor between sofa and shelf is 130 cm: 120 wide fails, 128 is only borderline
    assert check_route(scene(arm=(40, 40)), W, H, S, E, 120).status != "open"
    assert check_route(scene(arm=(40, 40)), W, H, S, E, 126, margin_cm=4).status in ("borderline", "blocked")
    assert check_route(scene(arm=(40, 40)), W, H, S, E, 90, margin_cm=0.5).status == "open"


def _brute(fps, step, locked_ok=True):
    """Independent enumeration through the public route check; returns min displacement of 1-move fixes."""
    best = None
    for f in fps:
        if f.locked:
            continue
        for x in range(step, int(W), step):
            for y in range(step, int(H), step):
                c = f.moved_to(x, y)
                if not (0 <= c.polygon.bounds[0] and c.polygon.bounds[2] <= W and c.polygon.bounds[1] >= 0
                        and c.polygon.bounds[3] <= H):
                    continue
                trial = [c if o.id == f.id else o for o in fps]
                if any(o.id != f.id and o.polygon.intersection(c.polygon).area > 0.05 for o in fps):
                    continue
                if check_route(trial, W, H, S, E, 90).status == "open":
                    d = math.dist(f.center, (x, y))
                    best = d if best is None else min(best, d)
    return best


def test_search_optimal_on_declared_set_and_locks_kept():
    fps = scene(extra=[R("plant", "plant", 50, 290, 40, 40)])
    plan = plan_moves(fps, W, H, S, E, 90, max_moves=2, grid_step_cm=25)
    assert plan.status == "plan_found" and plan.complete and len(plan.moves) == 1
    assert plan.moves[0].displacement_cm == pytest.approx(_brute(fps, 25), abs=1e-6)
    for a, b in zip(fps, plan.layout):
        if a.locked:
            assert a.polygon.equals(b.polygon)
    assert check_route(plan.layout, W, H, S, E, 90).status == "open"
    assert fps[2].center == (265.0, 180.0)  # input not mutated


def test_zero_moves_when_open():
    p = plan_moves(scene(arm=(40, 40)), W, H, S, E, 90)
    assert p.status == "already_open" and not p.moves


def test_two_moves_when_one_is_not_enough():
    fps = scene(extra=[R("arm2", "arm2", 180, 145, 50, 70)])  # second blocker fills the corridor
    p = plan_moves(fps, W, H, S, E, 90, max_moves=2, grid_step_cm=50, time_budget_s=20)
    assert p.status == "plan_found" and len(p.moves) == 2
    assert p.evaluated["1_move"] > 0
    p1 = plan_moves(fps, W, H, S, E, 90, max_moves=1, grid_step_cm=50)
    assert p1.status == "no_solution_in_set"


def test_no_solution_when_blocker_locked_and_incomplete_when_out_of_time():
    fps = [f.locked_as(True) for f in scene()]
    p = plan_moves(fps, W, H, S, E, 90)
    assert p.status == "no_solution_in_set" and "does not prove" in p.message
    fps2 = scene(extra=[R("arm2", "arm2", 180, 145, 50, 70)])
    p2 = plan_moves(fps2, W, H, S, E, 90, max_moves=2, grid_step_cm=5, time_budget_s=0.05)
    assert p2.status in ("search_incomplete", "plan_found_search_incomplete") and not p2.complete
