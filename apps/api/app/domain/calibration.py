"""Planar calibration: image pixels -> board centimetres (represented-room units).

Everything downstream works in one frame: centimetres on the board, origin top-left.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class CalibrationError(ValueError):
    """Calibration rejected as unreliable. The pipeline must stop, not guess."""


@dataclass(frozen=True)
class Calibration:
    homography: np.ndarray  # 3x3, pixels -> cm
    board_w_cm: float
    board_h_cm: float
    check_error_cm: float  # |measured - declared| of the independent check length
    check_error_pct: float
    scale_ratio: float  # max/min local cm-per-pixel over the board (perspective severity)
    image_size: tuple[int, int]  # (w, h)

    def to_cm(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.homography).reshape(-1, 2)

    def to_px(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, np.linalg.inv(self.homography)).reshape(-1, 2)

    def cm_per_px_at(self, px) -> float:
        x, y = px
        a = self.to_cm([(x, y), (x + 1, y), (x, y + 1)])
        return float((np.linalg.norm(a[1] - a[0]) + np.linalg.norm(a[2] - a[0])) / 2)

    def summary(self) -> dict:
        return {"board_cm": [self.board_w_cm, self.board_h_cm],
                "check_error_cm": round(self.check_error_cm, 3),
                "check_error_pct": round(self.check_error_pct, 3),
                "scale_ratio": round(self.scale_ratio, 3)}


def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _angles(q: np.ndarray) -> list[float]:
    out = []
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        out.append(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
    return out


def calibrate(corners_px, board_w_cm: float, board_h_cm: float, check_px, check_len_cm: float,
              image_size: tuple[int, int], check_tol_pct: float = 3.0,
              max_scale_ratio: float = 2.0) -> Calibration:
    """corners_px ordered TL, TR, BR, BL. check_px = two pixel points of a known, unused length."""
    q = np.asarray(corners_px, dtype=np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise CalibrationError("need exactly 4 finite board corners")
    if board_w_cm <= 0 or board_h_cm <= 0 or check_len_cm <= 0:
        raise CalibrationError("board dimensions and check length must be positive")
    cross = [_cross2(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        raise CalibrationError("corners are collinear, self-intersecting or mis-ordered")
    area = 0.5 * abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4)))
    if area < 0.10 * image_size[0] * image_size[1]:
        raise CalibrationError("board covers under 10% of the image; resolution too low")
    ang = _angles(q)
    if min(ang) < 40 or max(ang) > 140:
        raise CalibrationError(f"board corner angles {min(ang):.0f}-{max(ang):.0f} deg: view too oblique")
    dst = np.array([[0, 0], [board_w_cm, 0], [board_w_cm, board_h_cm], [0, board_h_cm]], np.float64)
    H = cv2.getPerspectiveTransform(q.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise CalibrationError("homography is singular")
    cal0 = Calibration(H, board_w_cm, board_h_cm, 0, 0, 1.0, image_size)
    cx, cy = q.mean(axis=0)
    scales = [cal0.cm_per_px_at(p) for p in list(q) + [(cx, cy)]]
    ratio = max(scales) / min(scales)
    if ratio > max_scale_ratio:
        raise CalibrationError(f"perspective too strong (scale ratio {ratio:.2f} > {max_scale_ratio})")
    c = np.asarray(check_px, dtype=np.float64).reshape(2, 2)
    cc = cal0.to_cm(c)
    measured = float(np.linalg.norm(cc[0] - cc[1]))
    err = abs(measured - check_len_cm)
    pct = 100 * err / check_len_cm
    if pct > check_tol_pct:
        raise CalibrationError(
            f"independent check length measures {measured:.1f} cm vs declared {check_len_cm:.1f} cm "
            f"({pct:.1f}% > {check_tol_pct}%)")
    return Calibration(H, board_w_cm, board_h_cm, err, pct, ratio, image_size)
