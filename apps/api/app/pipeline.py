"""Run orchestration + agent trace.

Perceiver -> Calibrator (deterministic) -> Constraint Interpreter -> Planner (deterministic search)
-> Human confirm -> Verifier (recapture). States: measuring -> needs_confirmation -> planning ->
awaiting_recapture -> verifying -> ready | needs_remeasurement | failed.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np

from .domain.calibration import Calibration, CalibrationError, calibrate
from .domain.connectivity import check_route
from .domain.footprints import Footprint, footprint_from_mask
from .domain.search import Plan, plan_moves
from .domain.verification import verify
from .providers import get_providers
from .providers.base import ConstraintSpec, ProviderError

SCOPE = "Calibrated tabletop model only. Not accessibility compliance or a real-world measurement."


class PipelineError(Exception):
    def __init__(self, msg: str, code: int = 422):
        super().__init__(msg)
        self.code = code


@dataclass
class Run:
    id: str
    status: str = "measuring"
    board_w: float = 0
    board_h: float = 0
    check_len: float = 0
    text: str = ""
    anchors: dict = field(default_factory=dict)
    photos: dict = field(default_factory=dict)
    calibration: Calibration | None = None
    footprints: list[Footprint] = field(default_factory=list)
    spec: ConstraintSpec | None = None
    start: tuple | None = None
    end: tuple | None = None
    route_initial: dict | None = None
    plan: Plan | None = None
    cal2: Calibration | None = None
    observed: list[Footprint] = field(default_factory=list)
    verification: dict | None = None
    trace: list[dict] = field(default_factory=list)
    error: str | None = None
    perception_notes: list[str] = field(default_factory=list)


RUNS: dict[str, Run] = {}


class _Step:
    def __init__(self):
        self.output = ""

    def out(self, s: str):
        self.output = s


@contextmanager
def step(run: Run, agent: str, role: str, model: str, inp: str):
    rec = {"n": len(run.trace) + 1, "agent": agent, "role": role, "model": model, "input_summary": inp,
           "output_summary": "", "duration_ms": 0, "status": "ok"}
    run.trace.append(rec)
    s, t0 = _Step(), time.perf_counter()
    try:
        yield s
    except Exception as e:
        rec["status"] = "error"
        s.output = f"{type(e).__name__}: {e}"
        raise
    finally:
        rec["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["output_summary"] = s.output


def _decode(b: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise PipelineError("photo is not a decodable image")
    if max(img.shape[:2]) > 6000:
        raise PipelineError("photo too large")
    return img


def _perceive_and_calibrate(run: Run, photo: bytes, tag: str, perception):
    img = _decode(photo)
    h, w = img.shape[:2]
    with step(run, "Perceiver", f"segment objects and find board markers ({tag})", perception.model,
              f"{w}x{h} image") as s:
        res = perception.perceive(img)
        s.out(f"{len(res.objects)} objects ({', '.join(sorted({o.label for o in res.objects}))}), 4 corners, check ref")
    with step(run, "Calibrator", f"homography + independent length check ({tag})", "none (deterministic)",
              f"board {run.board_w}x{run.board_h} cm, check {run.check_len} cm") as s:
        cal = calibrate(res.board_corners_px, run.board_w, run.board_h, res.check_points_px, run.check_len, (w, h))
        s.out(f"check error {cal.check_error_pct:.2f}%, scale ratio {cal.scale_ratio:.2f}")
    fps, seen = [], {}
    for o in res.objects:
        n = seen.get(o.label, 0) + 1
        seen[o.label] = n
        fid = o.label if n == 1 else f"{o.label}_{n}"
        fps.append(footprint_from_mask(o.mask, cal, fid, o.label, confidence=o.confidence,
                                       source=perception.model))
    return cal, fps, res.notes


def start_run(photo: bytes, board_w: float, board_h: float, check_len: float, text: str, anchors: dict) -> Run:
    perception, constraint = get_providers()
    run = Run(id=uuid.uuid4().hex[:12], board_w=board_w, board_h=board_h, check_len=check_len, text=text,
              anchors={k.lower(): tuple(v) for k, v in anchors.items()}, photos={"initial": photo})
    RUNS[run.id] = run
    try:
        run.calibration, run.footprints, run.perception_notes = _perceive_and_calibrate(
            run, photo, "initial", perception)
        labels = sorted({f.label for f in run.footprints})
        with step(run, "Constraint Interpreter", "plain-language constraint -> validated schema",
                  constraint.model, f"{len(run.text)} chars, {len(labels)} known labels") as s:
            spec = constraint.interpret(run.text, labels)
            run.spec = spec
            for nm in (spec.start_name, spec.end_name):
                if nm not in run.anchors:
                    raise PipelineError(f"route endpoint '{nm}' has no anchor point on the board; mark it")
            run.start, run.end = run.anchors[spec.start_name], run.anchors[spec.end_name]
            lock = set(spec.locked_labels) | {spec.end_name, spec.start_name}
            run.footprints = [f.locked_as(f.label in lock) for f in run.footprints]
            s.out(f"{spec.clearance_cm:g} cm route {spec.start_name}->{spec.end_name}, locked "
                  f"{[f.id for f in run.footprints if f.locked]}, max_moves {spec.max_moves}")
        with step(run, "Route Checker", "initial connectivity (deterministic geometry)", "none (deterministic)",
                  f"{len(run.footprints)} footprints") as s:
            r = check_route(run.footprints, board_w, board_h, run.start, run.end, spec.clearance_cm)
            run.route_initial = r.to_dict()
            s.out(f"route {r.status}")
        run.status = "needs_confirmation"
    except (PipelineError, ProviderError, CalibrationError, ValueError) as e:
        run.status, run.error = "failed", str(e)
        raise PipelineError(str(e)) from e
    return run


def confirm(run: Run, locked_ids: list[str] | None = None, accept_review_ids: list[str] | None = None,
            grid_step_cm: float = 10.0, time_budget_s: float = 8.0) -> Run:
    if run.status not in ("needs_confirmation", "awaiting_recapture", "failed") or not run.footprints:
        raise PipelineError(f"cannot confirm in state {run.status}", 409)
    if run.calibration is None or run.spec is None:
        raise PipelineError("run has no calibration/constraints", 409)
    with step(run, "Human", "confirm footprints and locks", "person", f"{len(run.footprints)} footprints") as s:
        if locked_ids is not None:
            bad = [i for i in locked_ids if i not in {f.id for f in run.footprints}]
            if bad:
                raise PipelineError(f"unknown ids {bad}")
            run.footprints = [f.locked_as(f.id in locked_ids) for f in run.footprints]
        pending = [f.id for f in run.footprints if f.needs_review and f.id not in (accept_review_ids or [])]
        if pending:
            raise PipelineError(f"footprints need review/acceptance: {pending}", 409)
        s.out(f"locks {[f.id for f in run.footprints if f.locked]}")
    run.status = "planning"
    with step(run, "Planner", "bounded finite-grid search, 0/1/2 moves", "none (deterministic)",
              f"grid {grid_step_cm} cm, budget {time_budget_s}s") as s:
        run.plan = plan_moves(run.footprints, run.board_w, run.board_h, run.start, run.end,
                              run.spec.clearance_cm, run.spec.max_moves, grid_step_cm, time_budget_s)
        s.out(f"{run.plan.status}: {[m.id for m in run.plan.moves]} {run.plan.total_displacement_cm:.1f} cm")
    run.status = "ready" if run.plan.status == "already_open" else (
        "awaiting_recapture" if run.plan.moves else "failed")
    if run.status == "failed":
        run.error = run.plan.message
    return run


def recapture(run: Run, photo: bytes) -> Run:
    if run.status != "awaiting_recapture" or run.plan is None:
        raise PipelineError(f"cannot recapture in state {run.status}", 409)
    perception, _ = get_providers()
    run.photos["recapture"] = photo
    run.status = "verifying"
    try:
        run.cal2, run.observed, _ = _perceive_and_calibrate(run, photo, "recapture", perception)
        with step(run, "Verifier", "recaptured poses vs expected, recompute route", "none (deterministic)",
                  f"{len(run.observed)} observed, {len(run.plan.moves)} planned moves") as s:
            run.verification = verify(run.footprints, run.plan, run.observed, run.board_w, run.board_h,
                                      run.start, run.end, run.spec.clearance_cm)
            s.out(run.verification["verdict"])
        run.status = "ready" if run.verification["pass"] else "needs_remeasurement"
    except (ProviderError, CalibrationError, ValueError, PipelineError) as e:
        run.status, run.error = "awaiting_recapture", str(e)
        raise PipelineError(str(e)) from e
    return run


def run_view(run: Run) -> dict:
    cal = run.calibration
    return {"id": run.id, "status": run.status, "error": run.error, "scope": SCOPE,
            "calibration": cal.summary() if cal else None,
            "constraint": run.spec.model_dump() if run.spec else None,
            "start_cm": run.start, "end_cm": run.end,
            "footprints": [f.to_dict(cal) for f in run.footprints],
            "route_initial": _route_px(run.route_initial, cal),
            "plan": _plan_view(run),
            "observed": [f.to_dict(run.cal2) for f in run.observed],
            "verification": _verif_view(run), "perception_notes": run.perception_notes}


def _route_px(route, cal):
    if not route or not route.get("witness_cm") or cal is None:
        return route
    r = dict(route)
    r["witness_px"] = cal.to_px(route["witness_cm"]).round(1).tolist()
    return r


def _plan_view(run):
    if not run.plan:
        return None
    d = run.plan.to_dict(run.calibration)
    d["route_after"] = _route_px(d["route_after"], run.calibration)
    return d


def _verif_view(run):
    if not run.verification:
        return None
    v = dict(run.verification)
    v["route"] = _route_px(v["route"], run.cal2)
    return v
