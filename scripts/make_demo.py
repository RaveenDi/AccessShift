"""Generate the synthetic demo photos into apps/api/demo_data and the synthetic eval manifest.

  python scripts/make_demo.py

writes   apps/api/demo_data/*.png                (the four UI demo photos)
         apps/api/demo_data/eval/*.png           (photos for the eval manifest, gitignored)
         evals/manifests/demo.synthetic.jsonl    (ground truth written from the scene definitions below, BEFORE any run)

SYNTHETIC: coloured rectangles. It tests plumbing and geometry, not Meta models or real photos.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

import cv2  # noqa: E402

from app import demo  # noqa: E402
from app.demo import ANCHORS, BOARD_H, BOARD_W, CHECK_LEN, GT, TEXT, ensure_demo, render  # noqa: E402
from app.domain.connectivity import check_route  # noqa: E402
from app.domain.footprints import rect_footprint  # noqa: E402
from app.domain.search import plan_moves  # noqa: E402

CLEAR = 90.0
LOCKS = ("sofa", "shelf")  # plus the endpoint object (desk), locked implicitly
TEXT_LOCKED = ("Keep a 90 cm wheelchair route from the door to the desk. "
               "Lock the sofa, the shelf and the armchair. Use at most 1 move.")
MAX_MOVES, GRID = 2, 10.0


def fps(layout, locked):
    return [rect_footprint(k, k.rsplit("_", 1)[0] if k not in demo.PALETTE else k, *g, locked=k in locked,
                           margin_cm=2.0) for k, g in layout.items()]


def locks_for(extra=()):
    return set(LOCKS) | {"desk"} | set(extra)


def gt_status(layout, locked):
    f = fps(layout, locked)
    s, e = ANCHORS["door"], ANCHORS["desk"]
    return (check_route(f, BOARD_W, BOARD_H, s, e, CLEAR, margin_cm=0.0).status == "open",
            check_route(f, BOARD_W, BOARD_H, s, e, CLEAR).status)


def expected_min_moves(layout, locked, max_moves=MAX_MOVES):
    p = plan_moves(fps(layout, locked), BOARD_W, BOARD_H, ANCHORS["door"], ANCHORS["desk"], CLEAR, max_moves, GRID, 30)
    if p.status == "already_open":
        return 0, p
    return (len(p.moves) if p.status == "plan_found" else None), p


def apply_moves(layout, moves):
    out = dict(layout)
    for m in moves:
        w, h = layout[m.id][2:]
        out[m.id] = (m.to_center[0] - w / 2, m.to_center[1] - h / 2, w, h)
    return out


def recapture_layouts(layout, locked):
    """The four things a person might do after being given the plan: follow it, nudge it, do something else, nothing."""
    _, plan = expected_min_moves(layout, locked)
    assert plan.moves, plan.message
    mv = plan.moves[0]
    x, y, w, h = layout[mv.id]
    nudged = dict(layout)
    nudged[mv.id] = (x + 25, y, w, h)
    if "plant" in nudged and mv.id != "plant":  # an unrelated object also moved
        px, py, pw, ph = nudged["plant"]
        nudged["plant"] = (px, py - 50, pw, ph)
    assert not gt_status(nudged, locked)[0]
    alt = None
    for cx in range(40, 480, 20):
        for cy in range(40, 320, 20):
            trial = dict(layout)
            trial[mv.id] = (cx - w / 2, cy - h / 2, w, h)
            if (math.dist((cx, cy), mv.to_center) < 70 or cx - w / 2 < 0 or cy - h / 2 < 0 or cx + w / 2 > BOARD_W
                    or cy + h / 2 > BOARD_H):
                continue
            f = fps(trial, locked)
            if any(a.polygon.intersection(b.polygon).area > 0.05 for i, a in enumerate(f) for b in f[i + 1:]):
                continue
            if check_route(f, BOARD_W, BOARD_H, ANCHORS["door"], ANCHORS["desk"], CLEAR).status == "open":
                d = math.dist((cx, cy), (x + w / 2, y + h / 2))
                if alt is None or d < alt[0]:
                    alt = (d, trial)
    assert alt, "no alternative open layout"
    return [("correct", apply_moves(layout, plan.moves), "correct"), ("nudged", nudged, "wrong"),
            ("elsewhere", alt[1], "wrong"), ("unchanged", layout, "wrong")]


def build_case(cid, split, layout, text, locked, traps, out_dir, seed, recap=False, expect_review=False,
               expect_status=None, notes="", gt_open=None, max_moves=MAX_MOVES):
    ok, status = gt_status(layout, locked)
    ok = ok if gt_open is None else gt_open  # a 2 cm physical margin is below the 5 cm raster: state it by construction
    if expect_status:
        assert status == expect_status, (cid, status, expect_status)
    emm, _ = expected_min_moves(layout, locked, max_moves)
    rel = out_dir.relative_to(ROOT).as_posix()
    cv2.imwrite(str(out_dir / f"{cid}.png"), render(layout, seed=seed))
    recs = []
    if recap:
        for i, (name, lay, expected) in enumerate(recapture_layouts(layout, locked)):
            cv2.imwrite(str(out_dir / f"{cid}-{name}.png"), render(lay, seed=seed + 1 + i))
            recs.append({"photo": f"{rel}/{cid}-{name}.png", "expected": expected})
    gt = {"footprints_mm": [{"id": k, "label": k.rsplit("_", 1)[0] if k not in demo.PALETTE else k,
                              "x": g[0] * 10, "y": g[1] * 10, "w": g[2] * 10, "h": g[3] * 10} for k, g in layout.items()],
          "constraint": {"clearance_mm": CLEAR * 10, "start": "door", "end": "desk", "locked_ids": sorted(locked),
                         "max_moves": max_moves},
          "route_open": ok, "route_expected_status": status, "expected_min_moves": emm, "recapture": recs}
    if expect_review:
        gt["expect_review"] = True
    if notes:
        gt["notes"] = notes
    return {"id": cid, "split": split, "photo": f"{rel}/{cid}.png", "board_mm": [BOARD_W * 10, BOARD_H * 10],
            "check_len_mm": CHECK_LEN * 10, "anchors_mm": {k: [v[0] * 10, v[1] * 10] for k, v in ANCHORS.items()},
            "constraint_text": text, "traps": traps, "ground_truth": gt}


def build_eval_manifest(root: Path = ROOT) -> Path:
    out = root / "apps" / "api" / "demo_data" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    blocked = dict(GT)
    open_layout = demo.scenes()["correct"]
    shifted = dict(GT)
    shifted.update(armchair=(215, 150, 70, 70), plant=(60, 60, 40, 40), box=(380, 240, 50, 50))
    borderline = dict(GT)
    borderline.update(armchair=(40, 40, 70, 70), plant=(230, 110, 40, 38))  # leaves a 92 cm gap for a 90 cm route
    merged = dict(GT)
    merged.pop("box")
    merged.update(box=(400, 250, 50, 50), box_2=(450, 270, 50, 40))  # two same-colour boxes touching in an L
    cases = [
        build_case("demo-blocked", "dev", blocked, TEXT, locks_for(), ["blocked_route", "deliberately_wrong_move"],
                   out, 100, recap=True, expect_status="blocked"),
        build_case("demo-open", "dev", open_layout, TEXT, locks_for(), ["already_open"], out, 200,
                   expect_status="open"),
        build_case("demo-locked", "dev", blocked, TEXT_LOCKED, locks_for(["armchair"]), ["locked_object_blocks"],
                   out, 300, expect_status="blocked", max_moves=1),
        build_case("demo-shifted", "heldout", shifted, TEXT, locks_for(), ["blocked_route", "deliberately_wrong_move"],
                   out, 400, recap=True, expect_status="blocked"),
        build_case("demo-borderline", "heldout", borderline, TEXT, locks_for(), ["borderline_clearance"], out, 500,
                   expect_status="borderline", gt_open=True),
        build_case("demo-merged", "heldout", merged, TEXT, locks_for(), ["merged_pieces"], out, 600,
                   expect_review=True, notes="GT-CANARY-merged-L-shape"),
    ]
    mp = root / "evals" / "manifests"
    mp.mkdir(parents=True, exist_ok=True)
    path = mp / "demo.synthetic.jsonl"
    path.write_text("# SYNTHETIC demo manifest generated by scripts/make_demo.py (coloured rectangles). Plumbing only.\n"
                    + "\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    demo_dir = ROOT / "apps" / "api" / "demo_data"
    for p in demo_dir.glob("*.png"):
        p.unlink()
    print(ensure_demo(demo_dir))
    print("manifest:", build_eval_manifest())
