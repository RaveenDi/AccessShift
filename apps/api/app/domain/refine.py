"""Deterministic refinement of the board corners and the check-ruler end points from segmentation masks.

Spark reads both on a coarse 0-1000 scale (about 1-2% of the image off), too imprecise for the homography and for the
independent length check. SAM segments the board and the ruler from box prompts built around Spark's values; the precise
points are then read from the mask pixels, and Spark's values are only seeds and a sanity reference. A refinement that
fails a check returns the Spark seed unchanged and says why (source "seed"): nothing is invented, and the independent
length check in `calibrate` stays independent of the fit (the ruler is never used to place the board).

Pure geometry on masks (OpenCV + numpy); no provider, no I/O.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

SEED_TOL = 0.05          # a refined point may sit this far from its seed: fraction of the image width (x) / height (y)
RULER_LEN_TOL = 0.25     # refined ruler length vs the seed's: at most 25% off
MIN_RULER_ASPECT = 3.0   # the ruler mask rectangle: long side / short side
MIN_RULER_PX = 20        # fewer mask pixels than this cannot define a line
MIN_BOARD_FILL = 0.85    # board mask area / area of its quadrilateral: at least this ...
MAX_BOARD_FILL = 1.15    # ... and at most this (a disc or blob is not a board, even if 4 of its points fit the seed)
MIN_RECT_FILL = 0.93     # the lower bound when only a minAreaRect (no true quadrilateral) fits
TICK_FACTOR = 1.6        # an end tick is this much thicker than the ruler line ...
TICK_MAX_FRAC = 0.04     # ... and at most this fraction of the ruler length wide


@dataclass(frozen=True)
class Refined:
    points: list    # board: TL, TR, BR, BL; check ruler: the two end points, in the seed's order. Pixels [x, y].
    source: str     # "refined" (from the mask) | "seed" (Spark's value, unchanged)
    detail: str     # what was done, or why the seed was kept


class _Reject(Exception):
    """A refinement failed a sanity check; the message is the reason the seed is kept."""


def _seed_array(seed: Sequence, n: int) -> np.ndarray:
    a = np.asarray(seed, dtype=np.float64)
    if a.shape != (n, 2) or not np.isfinite(a).all():
        raise ValueError(f"seed must be {n} finite [x, y] points")
    return a


def _largest_component(mask, image_size: tuple[int, int]) -> np.ndarray:
    """uint8 0/1 image holding the largest 8-connected piece of the mask."""
    if mask is None:
        raise _Reject("SAM returned no mask")
    w, h = image_size
    m = np.asarray(mask)
    if m.shape != (h, w):
        raise _Reject(f"mask is {m.shape[1]}x{m.shape[0]}, the image {w}x{h}")
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if n < 2:
        raise _Reject("mask is empty")
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def _within_seed(refined: np.ndarray, seed: np.ndarray, image_size: tuple[int, int], what: str) -> float:
    """Largest distance of a refined point from its seed, as a fraction of the image (per axis). Rejects above SEED_TOL."""
    w, h = image_size
    d = np.abs(refined - seed) / np.array([w, h], np.float64)
    worst = float(d.max())
    if worst > SEED_TOL:
        i = int(np.argmax(d.max(axis=1)))
        raise _Reject(f"{what} {i} is {d[i, 0] * 100:.1f}% / {d[i, 1] * 100:.1f}% of the image width / height from the "
                      f"Spark seed (limit {SEED_TOL * 100:.0f}%)")
    return worst


# ---- board ---------------------------------------------------------------------------------------------------------
def _quad(contour: np.ndarray) -> tuple[np.ndarray, str]:
    """4 vertices for a contour: cv2.approxPolyDP of its convex hull at the smallest tolerance that leaves exactly 4
    vertices; the minAreaRect corners when no tolerance does (the hull goes from 5 or more vertices to fewer than 4)."""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for frac in np.arange(0.002, 0.2, 0.002):
        approx = cv2.approxPolyDP(hull, float(frac) * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64), "approxPolyDP"
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64), "minAreaRect"


def _order(q: np.ndarray) -> np.ndarray:
    """Clockwise on screen (image y points down), starting at the top-left-most vertex: TL, TR, BR, BL."""
    c = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    return np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)


def _convex(q: np.ndarray) -> bool:
    """True when the ordered 4 vertices turn the same way at every corner (convex, not self-intersecting)."""
    cross = []
    for i in range(4):
        a, b = q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]
        cross.append(float(a[0] * b[1] - a[1] * b[0]))
    return all(c > 0 for c in cross) or all(c < 0 for c in cross)


def _pts(a: np.ndarray) -> list:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in a]


def refine_board_corners(mask, seed: Sequence, image_size: tuple[int, int]) -> Refined:
    """The 4 corners (TL, TR, BR, BL) of the board mask in pixels, with `seed` (Spark's corners in that order) kept when
    the mask is missing, is not a quadrilateral that fills its corners, or a corner is more than 5% of the image from
    its seed. `mask` is a bool/0-1 (h, w) array or None; `image_size` is (w, h)."""
    s = _seed_array(seed, 4)
    try:
        comp = _largest_component(mask, image_size)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)  # holes (objects on the board) are not part of an external contour
        q, method = _quad(cnt)
        q = _order(q)
        if not _convex(q):
            raise _Reject("corner quadrilateral is not convex")
        quad_area = abs(float(cv2.contourArea(q.astype(np.float32))))
        if quad_area < 1.0:
            raise _Reject("mask is degenerate (no area)")
        fill = float(cv2.contourArea(cnt)) / quad_area
        need = MIN_BOARD_FILL if method == "approxPolyDP" else MIN_RECT_FILL
        if not need <= fill <= MAX_BOARD_FILL:
            raise _Reject(f"mask fills {fill * 100:.0f}% of its {method} quadrilateral "
                          f"(needs {need * 100:.0f}-{MAX_BOARD_FILL * 100:.0f}%)")
        shift = _within_seed(q, s, image_size, "corner")
    except _Reject as e:
        return Refined([list(map(float, p)) for p in seed], "seed", str(e))
    return Refined(_pts(q), "refined", f"{method} of the board mask; largest shift from the Spark seed "
                                       f"{shift * 100:.1f}% of the image")


# ---- check ruler ----------------------------------------------------------------------------------------------------
def _tick_ends(t: np.ndarray, lo: float, hi: float) -> tuple[float, float, int]:
    """The two ends of the ruler along its long axis, from the pixel positions `t` on that axis (extent lo..hi).
    A tick (a short bar across the line, much thicker than the line) at an end replaces that end by the tick's centre;
    without one the end is the extent of the mask. -> (end_a, end_b, number of ends refined by a tick)."""
    idx = np.rint(t - lo).astype(int)  # slice i is centred on lo + i
    n = int(idx.max()) + 1
    if n < 12:
        return lo, hi, 0
    prof = np.bincount(idx, minlength=n).astype(np.float64)  # mask pixels per 1-px slice across the line
    base = float(np.median(prof[n // 4: n - n // 4]))  # the line's own thickness, away from the ends
    thr = max(TICK_FACTOR * base, base + 3.0)
    max_run = max(3, int(TICK_MAX_FRAC * n))
    ends, used = [lo, hi], 0
    for side in (0, 1):
        seq = prof if side == 0 else prof[::-1]
        k = 0
        while k < 2 and seq[k] <= thr:  # a ragged end may start a slice or two late
            k += 1
        if seq[k] <= thr:
            continue
        j = k
        while j < n and seq[j] > thr:
            j += 1
        if j - k > max_run:  # too wide to be a tick
            continue
        centre = (k + j - 1) / 2.0  # slice units from this end
        ends[side] = lo + centre if side == 0 else lo + (n - 1) - centre
        used += 1
    return ends[0], ends[1], used


def refine_check_points(mask, seed: Sequence, image_size: tuple[int, int]) -> Refined:
    """The two end points of the check ruler from its mask: the ends of the long axis of the mask's minAreaRect, moved to
    the centre of an end tick when the ruler has ones. `seed` is Spark's two end points; the points come back in the
    seed's order. The seed is kept when the mask is missing or is not a thin line, the length differs from the seed's by
    more than 25%, or an end point is more than 5% of the image from its seed end."""
    s = _seed_array(seed, 2)
    try:
        comp = _largest_component(mask, image_size)
        ys, xs = np.nonzero(comp)
        if len(xs) < MIN_RULER_PX:
            raise _Reject(f"mask has only {len(xs)} pixels")
        pts = np.column_stack([xs, ys]).astype(np.float64)
        box = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32))).astype(np.float64)
        e0, e1 = box[1] - box[0], box[2] - box[1]
        long_e, short_e = (e0, e1) if np.linalg.norm(e0) >= np.linalg.norm(e1) else (e1, e0)
        length, width = float(np.linalg.norm(long_e)), max(float(np.linalg.norm(short_e)), 1.0)
        if length / width < MIN_RULER_ASPECT:
            raise _Reject(f"mask is not a thin line (long/short side {length / width:.1f}, needs {MIN_RULER_ASPECT:g})")
        u = long_e / length
        nrm = np.array([-u[1], u[0]])
        c = box.mean(axis=0)
        rel = pts - c
        t, v = rel @ u, rel @ nrm
        ta, tb, ticks = _tick_ends(t, float(t.min()), float(t.max()))
        v0 = float(np.median(v))  # across the line: its centre, whatever else is in the mask
        a, b = c + u * ta + nrm * v0, c + u * tb + nrm * v0
        if np.linalg.norm(a - s[0]) + np.linalg.norm(b - s[1]) > np.linalg.norm(a - s[1]) + np.linalg.norm(b - s[0]):
            a, b = b, a
        seed_len = float(np.linalg.norm(s[1] - s[0]))
        if seed_len <= 0:
            raise _Reject("the Spark seed has zero length")
        ratio = float(np.linalg.norm(b - a)) / seed_len
        if abs(ratio - 1) > RULER_LEN_TOL:
            raise _Reject(f"mask length is {ratio:.2f}x the Spark seed's (limit +-{RULER_LEN_TOL * 100:.0f}%)")
        refined = np.array([a, b])
        shift = _within_seed(refined, s, image_size, "end point")
    except _Reject as e:
        return Refined([list(map(float, p)) for p in seed], "seed", str(e))
    how = {0: "mask extent", 1: "one end tick, one mask end", 2: "end ticks"}[ticks]
    return Refined(_pts(refined), "refined", f"ends of the ruler mask's long axis ({how}); length {ratio:.2f}x the Spark "
                                             f"seed's, largest shift {shift * 100:.1f}% of the image")
