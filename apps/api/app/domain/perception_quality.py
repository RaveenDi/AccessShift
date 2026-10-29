"""Deterministic sanity checks on segmentation masks (SAM can return nothing, or merge pieces).

A flagged mask is never silently used: the footprint is marked needs_review and the person must accept it.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_PIECE_PX = 150
PIECE_FRACTION = 0.15  # a second connected piece this big (relative to the largest) means merged pieces
OVERLAP_FRACTION = 0.3  # two different objects sharing this much of the smaller mask means a merged/duplicate mask


def _pieces(mask: np.ndarray) -> list[int]:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    areas = sorted((int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True)
    if not areas:
        return []
    return [a for a in areas if a >= max(MIN_PIECE_PX, PIECE_FRACTION * areas[0])]


def mask_issues(items: list[tuple[str, str, np.ndarray]]) -> list[dict]:
    """items: (footprint id, label, bool mask). Returns acceptable-with-review issues."""
    out = []
    for fid, label, mask in items:
        pc = _pieces(mask)
        if len(pc) > 1:
            out.append({"kind": "merged_masks", "ids": [fid], "label": label, "blocking": True, "acceptable": True,
                        "detail": f"mask for {fid} has {len(pc)} separate pieces (possible merged objects)"})
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = np.asarray(items[i][2]) > 0, np.asarray(items[j][2]) > 0
            inter = int(np.logical_and(a, b).sum())
            small = min(int(a.sum()), int(b.sum()))
            if small and inter / small > OVERLAP_FRACTION:
                out.append({"kind": "overlapping_masks", "ids": [items[i][0], items[j][0]], "label": "",
                            "blocking": True, "acceptable": True,
                            "detail": f"masks for {items[i][0]} and {items[j][0]} overlap by "
                                      f"{inter / small:.0%} of the smaller one"})
    return out


def blocking_text(issues: list[dict]) -> str:
    return "; ".join(i["detail"] for i in issues)
