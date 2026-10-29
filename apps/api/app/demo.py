"""Synthetic demo scenes: coloured rectangles on a 500x350 cm (represented) board, drawn with Pillow,
then keystone-warped and noised so calibration and segmentation do real work.
SYNTHETIC: tests plumbing and geometry only; says nothing about real photos or Meta models."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .domain.connectivity import check_route
from .domain.footprints import rect_footprint
from .domain.search import plan_moves
from .providers.mock import MARKER_RGB, PALETTE, RULER_RGB

BOARD_W, BOARD_H, CHECK_LEN = 500.0, 350.0, 200.0
ANCHORS = {"door": [60.0, 175.0], "desk": [440.0, 180.0]}
TEXT = ("Keep a 90 cm wheelchair route from the door to the desk. "
        "Lock the sofa and the shelf.")
# id: (x, y, w, h) in cm; id == palette label
GT = {"sofa": (160, 20, 180, 90), "desk": (400, 30, 100, 60), "shelf": (160, 240, 180, 50),
      "armchair": (230, 145, 70, 70), "plant": (50, 290, 40, 40), "box": (400, 250, 50, 50)}
RULER = (100.0, 330.0)  # left end (cm); runs CHECK_LEN to the right
PX, ORIG, CANVAS = 2.0, (100, 100), (1200, 900)
_SRC = np.float32([[0, 0], [1200, 0], [1200, 900], [0, 900]])
_DST = np.float32([[40, 15], [1160, 45], [1185, 870], [12, 885]])


def _p(x, y):
    return (ORIG[0] + PX * x, ORIG[1] + PX * y)


def render(layout: dict, seed: int = 7) -> np.ndarray:
    im = Image.new("RGB", CANVAS, (90, 80, 70))
    d = ImageDraw.Draw(im)
    d.rectangle([*_p(0, 0), *_p(BOARD_W, BOARD_H)], fill=(225, 215, 190))
    d.rectangle([*_p(RULER[0], RULER[1] - 3), *_p(RULER[0] + CHECK_LEN, RULER[1] + 3)], fill=RULER_RGB)
    for fid, (x, y, w, h) in layout.items():
        d.rectangle([*_p(x, y), *_p(x + w, y + h)], fill=PALETTE[fid if fid in PALETTE else fid.rsplit('_', 1)[0]])
    for cx, cy in [(0, 0), (BOARD_W, 0), (BOARD_W, BOARD_H), (0, BOARD_H)]:
        px, py = _p(cx, cy)
        d.rectangle([px - 15, py - 15, px + 15, py + 15], fill=MARKER_RGB)
    arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    H = cv2.getPerspectiveTransform(_SRC, _DST)
    arr = cv2.warpPerspective(arr, H, CANVAS, flags=cv2.INTER_LINEAR, borderValue=(70, 80, 90))
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    rng = np.random.default_rng(seed)
    return np.clip(arr + rng.normal(0, 3, arr.shape), 0, 255).astype(np.uint8)


def _fps(layout, locked=("sofa", "shelf", "desk")):
    return [rect_footprint(i, i, *g, locked=i in locked, margin_cm=2.0) for i, g in layout.items()]


def scenes() -> dict[str, dict]:
    """Ground-truth layouts for the four photos. 'correct' follows the planner's first move;
    this stands in for a person physically moving the object as instructed."""
    blocked = dict(GT)
    plan = plan_moves(_fps(blocked), BOARD_W, BOARD_H, ANCHORS["door"], ANCHORS["desk"], 90.0, 2, 10.0)
    assert plan.moves, plan.message
    mv = plan.moves[0]
    w, h = GT[mv.id][2:]
    correct = dict(GT)
    correct[mv.id] = (mv.to_center[0] - w / 2, mv.to_center[1] - h / 2, w, h)
    wrong = dict(GT)
    wrong["armchair"] = (255, 145, 70, 70)  # nudged 25 cm, still in the corridor
    wrong["plant"] = (50, 240, 40, 40)  # unrelated object moved
    # wrong_open: armchair placed somewhere else that also opens the route (deviates from plan)
    alt = None
    base = _fps(blocked)
    for x in range(40, 480, 20):
        for y in range(40, 320, 20):
            trial = dict(GT)
            trial["armchair"] = (x - 35, y - 35, 70, 70)
            if math.dist((x, y), mv.to_center) < 70 or not (0 <= x - 35 and x + 35 <= BOARD_W and y + 35 <= BOARD_H):
                continue
            fps = _fps(trial)
            from .search_helpers import no_overlap  # noqa: E402
            if not no_overlap(fps):
                continue
            if check_route(fps, BOARD_W, BOARD_H, ANCHORS["door"], ANCHORS["desk"], 90.0).status == "open":
                d = math.dist((x, y), (265, 180))
                if alt is None or d < alt[0]:
                    alt = (d, trial)
    assert alt, "no alternative open layout"
    return {"blocked": blocked, "correct": correct, "wrong": wrong, "wrong_open": alt[1]}


def ensure_demo(out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ["blocked", "correct", "wrong", "wrong_open"]
    if not all((out / f"{n}.png").exists() for n in names):
        for n, lay in scenes().items():
            cv2.imwrite(str(out / f"{n}.png"), render(lay, seed=7 + names.index(n)))
    return {"images": names, "board_w_cm": BOARD_W, "board_h_cm": BOARD_H, "check_len_cm": CHECK_LEN,
            "anchors": ANCHORS, "constraint_text": TEXT, "synthetic": True}
