"""Route existence for a circular clearance envelope on a raster of the board.

Obstacles are expanded by clearance/2 + margin; the permitted boundary is shrunk by the same
amount; a centre path must exist between start and end. Tabletop model only: orientation and
turning behaviour of a real wheelchair are NOT modelled.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

from .footprints import Footprint

CELL_CM = 5.0


@dataclass(frozen=True)
class Grid:
    w_cm: float
    h_cm: float
    cell: float = CELL_CM

    @property
    def shape(self) -> tuple[int, int]:
        return (int(math.ceil(self.h_cm / self.cell)), int(math.ceil(self.w_cm / self.cell)))

    def cell_of(self, p) -> tuple[int, int]:
        return (int(p[1] // self.cell), int(p[0] // self.cell))  # (row, col)

    def center_of(self, rc) -> tuple[float, float]:
        return ((rc[1] + 0.5) * self.cell, (rc[0] + 0.5) * self.cell)


def raster_poly(poly: Polygon, radius: float, grid: Grid) -> np.ndarray:
    """Cells whose centre lies within `radius` of the polygon (cv2 fill, ~1 cell accuracy)."""
    ny, nx = grid.shape
    out = np.zeros((ny, nx), np.uint8)
    shape = poly.buffer(radius, quad_segs=8) if radius > 0 else poly
    polys = [shape] if shape.geom_type == "Polygon" else list(shape.geoms)
    for g in polys:
        pts = np.array(g.exterior.coords) / grid.cell - 0.5
        cv2.fillPoly(out, [np.round(pts * 16).astype(np.int32)], 1, shift=4)
    return out.astype(bool)


def boundary_mask(grid: Grid, radius: float) -> np.ndarray:
    ny, nx = grid.shape
    xs = (np.arange(nx) + 0.5) * grid.cell
    ys = (np.arange(ny) + 0.5) * grid.cell
    bx = (xs < radius) | (xs > grid.w_cm - radius)
    by = (ys < radius) | (ys > grid.h_cm - radius)
    return by[:, None] | bx[None, :]


def connected(blocked: np.ndarray, s: tuple[int, int], e: tuple[int, int]) -> bool:
    ny, nx = blocked.shape
    for r, c in (s, e):
        if not (0 <= r < ny and 0 <= c < nx) or blocked[r, c]:
            return False
    _, lab = cv2.connectedComponents((~blocked).astype(np.uint8), connectivity=4)
    return bool(lab[s] == lab[e] and lab[s] != 0)


def bfs_path(blocked: np.ndarray, s, e):
    ny, nx = blocked.shape
    prev = {s: None}
    q = deque([s])
    while q:
        p = q.popleft()
        if p == e:
            path = []
            while p is not None:
                path.append(p)
                p = prev[p]
            return path[::-1]
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (p[0] + d[0], p[1] + d[1])
            if 0 <= n[0] < ny and 0 <= n[1] < nx and not blocked[n] and n not in prev:
                prev[n] = p
                q.append(n)
    return None


def blocked_mask(polys: list[Polygon], grid: Grid, radius: float) -> np.ndarray:
    m = boundary_mask(grid, radius)
    for p in polys:
        m = m | raster_poly(p, radius, grid)
    return m


def layout_margin(footprints: list[Footprint], cell: float = CELL_CM) -> float:
    """Uncertainty used to separate open / borderline / blocked: footprint margin plus raster slack."""
    return max((f.margin_cm for f in footprints), default=0.0) + 0.75 * cell


@dataclass
class RouteResult:
    status: str  # open | borderline | blocked
    witness_cm: list[list[float]] | None
    witness_clearance_cm: float | None  # continuous min distance from witness to obstacles
    clearance_cm: float
    margin_cm: float
    note: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status, "witness_cm": self.witness_cm,
                "witness_min_gap_cm": self.witness_clearance_cm, "clearance_cm": self.clearance_cm,
                "margin_cm": round(self.margin_cm, 2), "note": self.note}


def check_route(footprints: list[Footprint], board_w: float, board_h: float, start, end,
                clearance_cm: float, margin_cm: float | None = None, cell: float = CELL_CM) -> RouteResult:
    """open: route exists even at radius+margin. blocked: no route even at radius-margin.
    borderline: route only exists under optimistic assumptions (never accepted as clear)."""
    grid = Grid(board_w, board_h, cell)
    m = layout_margin(footprints, cell) if margin_cm is None else margin_cm
    r = clearance_cm / 2.0
    polys = [f.polygon for f in footprints]
    s, e = grid.cell_of(start), grid.cell_of(end)
    cons = blocked_mask(polys, grid, r + m)
    opt = blocked_mask(polys, grid, max(r - m, 0.0))
    if connected(cons, s, e):
        path = bfs_path(cons, s, e)
        pts = [list(grid.center_of(c)) for c in path]
        pts[0], pts[-1] = list(map(float, start)), list(map(float, end))
        line = LineString(pts)
        gap = min((line.distance(p) for p in polys), default=float("inf"))
        note = ""
        if gap < r:
            note = "witness raster/continuous cross-check: gap below nominal radius"
        return RouteResult("open", [[round(x, 1), round(y, 1)] for x, y in pts], round(float(gap), 2),
                           clearance_cm, m, note)
    if connected(opt, s, e):
        path = bfs_path(opt, s, e)
        pts = [[round(x, 1), round(y, 1)] for x, y in (grid.center_of(c) for c in path)]
        return RouteResult("borderline", pts, None, clearance_cm, m,
                           "route exists only if measurement uncertainty is ignored; remeasure or loosen")
    return RouteResult("blocked", None, None, clearance_cm, m)
