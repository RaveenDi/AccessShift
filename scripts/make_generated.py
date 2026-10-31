"""Build eval photos from one AI-generated overhead tabletop photo (Higgsfield, GPT Image 2.5, 2026-10-31).

  apps/api/.venv/Scripts/python scripts/make_generated.py [--measure]

reads    evals/photos/generated/gen-base.png   (the generated base photo; layout as generated)
writes   evals/photos/generated/gen-*.png           (layouts made by moving the base photo's blocks in code)
         evals/manifests/generated.jsonl            (ground truth from the scene definitions below, BEFORE any run)

SYNTHETIC: a generated image, not a photo of a physical board. Footprints are exact by construction (blocks are
moved as pixel patches in a rectified frame), so this tests perception on a realistic-looking image and the full
plan/recapture/verify loop. It does NOT test camera calibration on a real board or physical measurement error.
The board represents a 6.0 m x 4.5 m room (units: cm, origin top-left), like the demo's represented-room units.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.domain.connectivity import check_route  # noqa: E402
from app.domain.footprints import rect_footprint  # noqa: E402
from app.domain.search import plan_moves  # noqa: E402

OUT = ROOT / "evals" / "photos" / "generated"
BASE = OUT / "gen-base.png"
BOARD_W, BOARD_H, CHECK_LEN = 600.0, 450.0, 200.0
RULER = (350.0, 36.0)  # left end (cm); runs CHECK_LEN to the right
ANCHORS = {"door": [60.0, 222.0], "desk": [475.0, 207.0]}
CLEAR, MAX_MOVES, GRID = 90.0, 2, 10.0
LOCKS = ("sofa", "shelf", "desk")
TEXT = "Keep a 90 cm wheelchair route from the door to the desk. Lock the sofa and the shelf."
TEXT_LOCKED = "Keep a 90 cm wheelchair route from the door to the desk. Lock the sofa, the shelf and the box. Use at most 1 move."
PX, PAD = 3.5, 70  # rectified pixels per cm; wood margin in pixels
REF = {"sofa": (40, 90, 200), "desk": (130, 75, 45), "shelf": (35, 130, 75),
       "armchair": (205, 40, 45), "plant": (230, 190, 30), "box": (240, 120, 30)}


def board_corners(img):
    """Corners of the light-grey board area (TL, TR, BR, BL) in base-photo pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey = ((hsv[..., 1] < 40) & (hsv[..., 2] > 150)).astype(np.uint8)
    grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grey)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnt, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnt, key=cv2.contourArea)).reshape(-1, 2).astype(np.float64)
    s, d = hull.sum(1), hull[:, 0] - hull[:, 1]
    return np.float32([hull[np.argmin(s)], hull[np.argmax(d)], hull[np.argmax(s)], hull[np.argmin(d)]])


def rectify(img):
    W, H = int(BOARD_W * PX) + 2 * PAD, int(BOARD_H * PX) + 2 * PAD
    dst = np.float32([[PAD, PAD], [W - PAD, PAD], [W - PAD, H - PAD], [PAD, H - PAD]])
    M = cv2.getPerspectiveTransform(board_corners(img), dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def blocks(rect):
    """{label: (x, y, w, h) cm, mask} for the six coloured blocks (side faces included, shadows excluded)."""
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] > 90) & (hsv[..., 2] > 50)).astype(np.uint8)
    m[:PAD], m[-PAD:], m[:, :PAD], m[:, -PAD:] = 0, 0, 0, 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = {}
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 3000:
            continue
        cnt, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mk = cv2.drawContours(np.zeros_like(m), cnt, -1, 1, cv2.FILLED).astype(bool)  # include the white sticker
        rgb = rect[lab == i][:, ::-1].mean(0)
        label = min(REF, key=lambda k: np.linalg.norm(rgb - np.array(REF[k])))
        assert label not in out, f"two blocks look like {label}"
        x, y, w, h = st[i, :4]
        out[label] = (((x - PAD) / PX, (y - PAD) / PX, w / PX, h / PX), mk)
    assert set(out) == set(REF), sorted(out)
    return out


def fps(layout, locked):
    return [rect_footprint(k, k, *g, locked=k in locked, margin_cm=2.0) for k, g in layout.items()]


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
    """Follow the plan, nudge it (plus an unrelated object), open the route some other way, or do nothing."""
    _, plan = expected_min_moves(layout, locked)
    assert plan.moves, plan.message
    mv = plan.moves[0]
    x, y, w, h = layout[mv.id]
    nudged = dict(layout)
    nudged[mv.id] = (x + 25, y, w, h)
    if mv.id != "plant":
        px, py, pw, ph = nudged["plant"]
        nudged["plant"] = (px, py - 50, pw, ph)
    assert not gt_status(nudged, locked)[0]
    alt = None
    for cx in range(40, int(BOARD_W) - 20, 20):
        for cy in range(40, int(BOARD_H) - 20, 20):
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


class Scene:
    """Empty board (blocks painted out) plus one movable patch per block, all in the rectified frame."""

    def __init__(self, rect, found):
        self.rect, self.found = rect, found
        holes = np.zeros(rect.shape[:2], np.uint8)
        for _, mk in found.values():
            holes |= mk.astype(np.uint8)
        holes = cv2.dilate(holes, np.ones((75, 75), np.uint8))  # side faces, soft shadows, anti-aliased edges
        # Fill vacated spots with the board's own lighting (normalised blur of clean board pixels) plus fine grain.
        hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
        clean = ((hsv[..., 1] < 40) & (hsv[..., 2] > 150) & (holes == 0)).astype(np.float32)
        f = rect.astype(np.float32)
        bg, have = np.zeros_like(f), np.zeros(f.shape[:2], bool)
        for sigma in (15, 40, 100, 250):  # finest blur that still sees enough clean board wins
            den = cv2.GaussianBlur(clean, (0, 0), sigma)
            est = cv2.GaussianBlur(f * clean[..., None], (0, 0), sigma) / np.maximum(den, 1e-6)[..., None]
            take = (den > 0.25) & ~have if sigma < 250 else ~have
            bg[take], have = est[take], have | take
        bg = cv2.GaussianBlur(bg, (0, 0), 8)
        fill = bg + np.random.default_rng(0).normal(0, 1.5, bg.shape)
        soft = cv2.GaussianBlur(holes.astype(np.float32), (0, 0), 12)[..., None]
        self.board = np.clip(f * (1 - soft) + fill * soft, 0, 255).astype(np.uint8)
        p0, p1 = (int(PAD + RULER[0] * PX), int(PAD + RULER[1] * PX)), (int(PAD + (RULER[0] + CHECK_LEN) * PX),
                                                                         int(PAD + RULER[1] * PX))
        cv2.line(self.board, p0, p1, (28, 28, 28), 9)
        for p in (p0, p1):
            cv2.line(self.board, (p[0], p[1] - 14), (p[0], p[1] + 14), (28, 28, 28), 5)
        cv2.putText(self.board, "CHECK 2 m", (p0[0] + 250, p0[1] + 52), cv2.FONT_HERSHEY_DUPLEX, 1.1, (28, 28, 28), 2,
                    cv2.LINE_AA)
        self.alpha = {}
        for k, (_, mk) in found.items():
            a = cv2.dilate(mk.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(np.float32)
            self.alpha[k] = cv2.GaussianBlur(a, (0, 0), 5)[..., None]

    def render(self, layout, seed):
        im = self.board.astype(np.float32)
        for k, (x, y, w, h) in layout.items():
            (x0, y0, _, _), _ = self.found[k]
            dx, dy = int(round((x - x0) * PX)), int(round((y - y0) * PX))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            size = (im.shape[1], im.shape[0])
            src = cv2.warpAffine(self.rect, M, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            a = cv2.warpAffine(self.alpha[k], M, size, flags=cv2.INTER_LINEAR)[..., None]
            im = im * (1 - a) + src.astype(np.float32) * a
        im = np.clip(im, 0, 255).astype(np.uint8)
        rng = np.random.default_rng(seed)  # each "photo" is taken from a slightly different camera position
        h, w = im.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-0.012, 0.012, (4, 2)).astype(np.float32) * np.float32([w, h])
        return cv2.warpPerspective(im, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT)


def snapped(found):
    """Positions in whole millimetres so the manifest and the rendered pixels agree."""
    return {k: tuple(round(v, 1) for v in g) for k, (g, _) in found.items()}


def build_case(scene, cid, split, layout, text, locked, traps, seed, recap=False, expect_status=None, max_moves=MAX_MOVES):
    ok, status = gt_status(layout, locked)
    if expect_status:
        assert status == expect_status, (cid, status, expect_status)
    emm, _ = expected_min_moves(layout, locked, max_moves)
    rel = OUT.relative_to(ROOT).as_posix()
    cv2.imwrite(str(OUT / f"{cid}.png"), scene.render(layout, seed))
    recs = []
    if recap:
        for i, (name, lay, expected) in enumerate(recapture_layouts(layout, locked)):
            cv2.imwrite(str(OUT / f"{cid}-{name}.png"), scene.render(lay, seed + 1 + i))
            recs.append({"photo": f"{rel}/{cid}-{name}.png", "expected": expected})
    gt = {"footprints_mm": [{"id": k, "label": k, "x": round(g[0] * 10), "y": round(g[1] * 10), "w": round(g[2] * 10),
                             "h": round(g[3] * 10)} for k, g in layout.items()],
          "constraint": {"clearance_mm": CLEAR * 10, "start": "door", "end": "desk", "locked_ids": sorted(locked),
                         "max_moves": max_moves},
          "route_open": ok, "route_expected_status": status, "expected_min_moves": emm, "recapture": recs}
    return {"id": cid, "split": split, "photo": f"{rel}/{cid}.png", "board_mm": [BOARD_W * 10, BOARD_H * 10],
            "check_len_mm": CHECK_LEN * 10, "anchors_mm": {k: [v[0] * 10, v[1] * 10] for k, v in ANCHORS.items()},
            "constraint_text": text, "traps": traps, "ground_truth": gt}


def main():
    rect = rectify(cv2.imread(str(BASE)))
    found = blocks(rect)
    base = snapped(found)
    if "--measure" in sys.argv:
        for k, g in base.items():
            print(k, g)
        print("base status:", gt_status(base, set(LOCKS)))
        return
    scene = Scene(rect, found)
    a, b, s = base["armchair"], base["box"], base["sofa"]
    # Blocked: armchair raised, box pushed in under it -> sofa/armchair/box/shelf form a wall with gaps under 90 cm.
    blocked = dict(base, armchair=(a[0], s[1] + s[3] + 50, a[2], a[3]))
    ay = blocked["armchair"][1] + a[3]
    blocked["box"] = (a[0] + (a[2] - b[2]) / 2, ay + 45, b[2], b[3])
    # Held-out: the wall moves left and uses the plant instead of the box.
    p = base["plant"]
    # Open: armchair parked out of the way, bottom right.
    open_layout = dict(base, armchair=(380.0, 330.0, a[2], a[3]))
    shifted = dict(base, armchair=(150.0, s[1] + s[3] + 50, a[2], a[3]))
    shifted["plant"] = (150.0 + (a[2] - p[2]) / 2, shifted["armchair"][1] + a[3] + 44, p[2], p[3])
    cases = [
        build_case(scene, "gen-open", "dev", open_layout, TEXT, set(LOCKS), ["already_open"], 200, expect_status="open"),
        build_case(scene, "gen-blocked", "dev", blocked, TEXT, set(LOCKS), ["blocked_route", "deliberately_wrong_move"],
                   100, recap=True, expect_status="blocked"),
        build_case(scene, "gen-locked", "dev", blocked, TEXT_LOCKED, set(LOCKS) | {"box"}, ["locked_object_blocks"], 300,
                   expect_status="blocked", max_moves=1),
        build_case(scene, "gen-shifted", "heldout", shifted, TEXT, set(LOCKS),
                   ["blocked_route", "deliberately_wrong_move"], 400, recap=True, expect_status="blocked"),
        # The layout exactly as generated: ~105 cm gaps around the armchair, inside measurement uncertainty.
        build_case(scene, "gen-borderline", "heldout", base, TEXT, set(LOCKS), ["borderline_clearance"], 500,
                   expect_status="borderline"),
    ]
    path = ROOT / "evals" / "manifests" / "generated.jsonl"
    path.write_text("# SYNTHETIC: blocks moved in code inside one AI-generated photo (Higgsfield GPT Image 2.5, "
                    "2026-10-31). Not a physical board.\n" + "\n".join(json.dumps(c) for c in cases) + "\n",
                    encoding="utf-8")
    for c in cases:
        g = c["ground_truth"]
        print(c["id"], c["split"], g["route_expected_status"], "min moves:", g["expected_min_moves"],
              "recaptures:", len(g["recapture"]))
    print("manifest:", path)


if __name__ == "__main__":
    main()
