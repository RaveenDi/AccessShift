"""Pixel mask -> metric footprint (oriented rectangle) with an uncertainty margin."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from .calibration import Calibration


@dataclass(frozen=True)
class Footprint:
    id: str
    label: str
    polygon: Polygon  # board cm
    margin_cm: float  # measurement uncertainty added to clearance
    locked: bool = False
    confidence: float = 1.0
    fit_ratio: float = 1.0  # mask area / fitted-rectangle area (1.0 = perfect rectangle)
    needs_review: bool = False
    source: str = ""

    @property
    def center(self) -> tuple[float, float]:
        c = self.polygon.centroid
        return (c.x, c.y)

    def moved_to(self, cx: float, cy: float) -> "Footprint":
        x, y = self.center
        return replace(self, polygon=affinity.translate(self.polygon, cx - x, cy - y))

    def locked_as(self, locked: bool) -> "Footprint":
        return replace(self, locked=locked)

    def to_dict(self, cal: Calibration | None = None) -> dict:
        ring = list(self.polygon.exterior.coords)[:-1]
        d = {"id": self.id, "label": self.label, "locked": self.locked,
             "center_cm": [round(v, 2) for v in self.center],
             "polygon_cm": [[round(x, 2), round(y, 2)] for x, y in ring],
             "margin_cm": round(self.margin_cm, 2), "confidence": round(self.confidence, 3),
             "fit_ratio": round(self.fit_ratio, 3), "needs_review": self.needs_review,
             "source": self.source}
        if cal is not None:
            d["polygon_px"] = [[round(float(x), 1), round(float(y), 1)] for x, y in cal.to_px(ring)]
        return d


def footprint_from_mask(mask: np.ndarray, cal: Calibration, fid: str, label: str, locked: bool = False,
                        confidence: float = 1.0, source: str = "", min_area_px: int = 150) -> Footprint:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError(f"empty mask for {fid}")
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_area_px:
        raise ValueError(f"mask for {fid} too small ({area:.0f}px)")
    rect = cv2.minAreaRect(cnt)
    box_cm = cal.to_cm(cv2.boxPoints(rect))
    poly = Polygon(box_cm)
    if not poly.is_valid or poly.area <= 0:
        raise ValueError(f"degenerate footprint for {fid}")
    rect_area_px = rect[1][0] * rect[1][1]
    fit = float(area / rect_area_px) if rect_area_px > 0 else 0.0
    cx, cy = rect[0]
    margin = 1.5 * cal.cm_per_px_at((cx, cy)) + cal.check_error_cm * 0.5
    return Footprint(fid, label, poly, float(margin), locked, confidence, fit,
                     fit < 0.85 or confidence < 0.5, source)


def rect_footprint(fid: str, label: str, x: float, y: float, w: float, h: float, locked=False,
                   margin_cm: float = 1.0) -> Footprint:
    """Axis-aligned rectangle helper (tests, ground truth, manual-layout baseline)."""
    return Footprint(fid, label, Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]), margin_cm, locked,
                     source="manual")
