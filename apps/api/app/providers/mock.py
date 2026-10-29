"""Mock providers. Perception is a real OpenCV colour-segmentation path on the synthetic demo image
(plumbing test only; says nothing about SAM or Spark quality). Labels come from a fixed colour table."""
from __future__ import annotations

import re

import cv2
import numpy as np

from .base import ConstraintSpec, DetectedObject, PerceptionResult, ProviderError

# RGB palette shared with scripts/make_demo.py. Mock "labeller" = nearest palette colour.
PALETTE = {
    "sofa": (60, 90, 200), "desk": (140, 90, 40), "shelf": (30, 120, 60),
    "armchair": (230, 120, 30), "plant": (150, 200, 60), "box": (160, 60, 160),
}
MARKER_RGB = (255, 0, 255)
RULER_RGB = (20, 20, 20)
TOL = 40.0


def _near(img_rgb: np.ndarray, rgb, tol=TOL) -> np.ndarray:
    d = np.linalg.norm(img_rgb.astype(np.float32) - np.array(rgb, np.float32), axis=2)
    return (d < tol).astype(np.uint8)


def _clean(m: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class MockPerception:
    name, model = "mock", "mock-colour-segmentation (OpenCV)"

    def perceive(self, image_bgr: np.ndarray) -> PerceptionResult:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        notes = []
        mk = _clean(_near(rgb, MARKER_RGB, 60))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mk)
        blobs = [cents[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 100]
        if len(blobs) != 4:
            raise ProviderError(f"expected 4 board corner markers, found {len(blobs)}")
        c = np.array(blobs)
        tl = c[np.argmin(c.sum(1))]
        br = c[np.argmax(c.sum(1))]
        tr = c[np.argmax(c[:, 0] - c[:, 1])]
        bl = c[np.argmin(c[:, 0] - c[:, 1])]
        corners = [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]
        rl = _clean(_near(rgb, RULER_RGB, 45))
        cnts, _ = cv2.findContours(rl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise ProviderError("check ruler not found")
        (cx, cy), (w, h), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        box = cv2.boxPoints(((cx, cy), (w, h), ang))
        # long-axis end points = midpoints of the two short edges
        e = [(box[i], box[(i + 1) % 4]) for i in range(4)]
        e.sort(key=lambda p: np.linalg.norm(p[0] - p[1]))
        m1, m2 = (e[0][0] + e[0][1]) / 2, (e[1][0] + e[1][1]) / 2
        objs = []
        for label, col in PALETTE.items():
            m = _clean(_near(rgb, col))
            k, lab, st, _ = cv2.connectedComponentsWithStats(m)
            for i in range(1, k):
                if st[i, cv2.CC_STAT_AREA] >= 300:
                    objs.append(DetectedObject(label, lab == i, 0.9))
        notes.append("labels from fixed colour table (mock); masks from colour thresholds")
        return PerceptionResult(corners, [m1.tolist(), m2.tolist()], objs, notes)


_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|centimet\w*|mm|millimet\w*|m|metres?|meters?)\b", re.I)
_LOCK = re.compile(r"\b(lock|locked|keep|fixed|fix|leave|don'?t move|do not move|stay)\b", re.I)


class MockConstraint:
    name, model = "mock", "mock-rule-parser"

    def interpret(self, text: str, known_labels: list[str]) -> ConstraintSpec:
        amb = []
        clearance = None
        for sent in re.split(r"[.;\n]", text):
            m = _NUM.search(sent)
            if m and re.search(r"route|path|clear|wide|width|wheelchair", sent, re.I):
                v, u = float(m.group(1)), m.group(2).lower()
                clearance = v / 10 if u.startswith("mm") or u.startswith("millimet") else (
                    v if u.startswith("c") else v * 100)
                break
        if clearance is None:
            if re.search(r"wheelchair", text, re.I):
                clearance = 90.0
                amb.append("no width stated; assumed 90 cm for wheelchair")
            else:
                raise ProviderError("constraint text has no route width")
        m = re.search(r"from\s+(?:the\s+)?([\w ]+?)\s+to\s+(?:the\s+)?([\w ]+?)(?:[.,;]|$|\s+(?:and|while|lock|keep))",
                      text, re.I)
        if not m:
            raise ProviderError("constraint text has no 'from X to Y' route")
        locked = []
        for sent in re.split(r"[.;\n]", text):
            if _LOCK.search(sent) and not re.search(r"\bfrom\b.*\bto\b", sent, re.I):
                locked += [lbl for lbl in known_labels if re.search(rf"\b{re.escape(lbl)}s?\b", sent, re.I)]
        mm = re.search(r"(?:at most|max(?:imum)?|up to)\s+(\d)\s+moves?", text, re.I)
        return ConstraintSpec(start_name=m.group(1).strip().lower(), end_name=m.group(2).strip().lower(),
                              clearance_cm=clearance, locked_labels=sorted(set(locked)),
                              max_moves=int(mm.group(1)) if mm else 2, ambiguities=amb)
