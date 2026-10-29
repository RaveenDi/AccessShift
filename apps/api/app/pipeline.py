"""Run orchestration + agent trace.

Perceiver -> Calibrator (deterministic) -> Planner Agent (bounded tool loop: interpret, check, search, finish)
-> Human confirm -> (person moves objects) -> Perceiver -> Calibrator -> Verifier (deterministic)
-> Recovery turn (agent, once) when the verifier finds a wrong move.
States: measuring -> needs_input | needs_confirmation -> awaiting_recapture -> verifying -> ready |
needs_remeasurement | failed. Models never set route status: only the deterministic tools and Verifier do.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, replace

import cv2
import numpy as np

from .agent import loop as agent_loop
from .agent.loop import AgentState
from .agent.prompts import SYSTEM_PLAN, SYSTEM_RECOVERY
from .agent.tools import ToolContext
from .config import agent_caps
from .domain.calibration import Calibration, CalibrationError, calibrate
from .domain.footprints import Footprint, footprint_from_mask
from .domain.perception_quality import blocking_text, mask_issues
from .domain.search import Plan, plan_moves
from .domain.verification import verify
from .providers import get_providers, provider_mode
from .providers.base import ProviderError
from .telemetry import step, totals

SCOPE = "Calibrated tabletop model only. Not accessibility compliance or a real-world measurement."
RECOVERABLE = ("route_still_blocked", "route_borderline")
MAX_RECOVERIES = 1


class PipelineError(Exception):
    def __init__(self, msg: str, code: int = 422):
        super().__init__(msg)
        self.code = code


@dataclass
class Constraint:
    start_name: str
    end_name: str
    clearance_cm: float
    max_moves: int
    locked_ids: list = field(default_factory=list)
    source: str = "agent"  # agent | human


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
    constraint: Constraint | None = None
    start: tuple | None = None
    end: tuple | None = None
    route_initial: dict | None = None
    plan: Plan | None = None
    plan_source: str = ""  # agent | human_replan | manual
    plan_confirmed: bool = False
    cal2: Calibration | None = None
    observed: list[Footprint] = field(default_factory=list)
    verification: dict | None = None
    trace: list[dict] = field(default_factory=list)
    error: str | None = None
    perception_notes: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    deferred_reason: str = ""
    agent: AgentState | None = None
    ctx: ToolContext | None = None
    human_locks: set | None = None
    recoveries_used: int = 0
    recovery: dict | None = None
    history: list[dict] = field(default_factory=list)


RUNS: dict[str, Run] = {}


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
    with step(run.trace, "Perceiver", f"segment objects and find board markers ({tag})", perception.model,
              f"{w}x{h} image", kind="perception") as s:
        res = perception.perceive(img)
        s.out(f"{len(res.objects)} objects ({', '.join(sorted({o.label for o in res.objects}))}), 4 corners, "
              f"check ref, {len(res.issues)} provider issue(s)")
    with step(run.trace, "Calibrator", f"homography + independent length check ({tag})", "none (deterministic)",
              f"board {run.board_w}x{run.board_h} cm, check {run.check_len} cm") as s:
        cal = calibrate(res.board_corners_px, run.board_w, run.board_h, res.check_points_px, run.check_len, (w, h))
        s.out(f"check error {cal.check_error_pct:.2f}%, scale ratio {cal.scale_ratio:.2f}")
    fps, seen, masks = [], {}, []
    for o in res.objects:
        n = seen.get(o.label, 0) + 1
        seen[o.label] = n
        fid = o.label if n == 1 else f"{o.label}_{n}"
        fps.append(footprint_from_mask(o.mask, cal, fid, o.label, confidence=o.confidence, source=perception.model))
        masks.append((fid, o.label, o.mask))
    issues = list(res.issues) + mask_issues(masks)
    flagged = {i for iss in issues for i in iss.get("ids", [])}
    fps = [replace(f, needs_review=True) if f.id in flagged else f for f in fps]
    if not fps:
        issues.append({"kind": "no_objects", "ids": [], "label": "", "blocking": True, "acceptable": False,
                       "detail": "no object was segmented; retake the photo"})
    return cal, fps, res.notes, issues


def start_run(photo: bytes, board_w: float, board_h: float, check_len: float, text: str, anchors: dict) -> Run:
    perception, _, planner = get_providers()
    run = Run(id=uuid.uuid4().hex[:12], board_w=board_w, board_h=board_h, check_len=check_len, text=text,
              anchors={k.lower(): tuple(v) for k, v in anchors.items()}, photos={"initial": photo})
    RUNS[run.id] = run
    try:
        run.calibration, run.footprints, run.perception_notes, run.issues = _perceive_and_calibrate(
            run, photo, "initial", perception)
    except (PipelineError, ProviderError, CalibrationError, ValueError) as e:
        run.status, run.error = "failed", str(e)
        raise PipelineError(str(e)) from e
    run.status = "needs_confirmation"
    if run.issues or any(f.needs_review for f in run.footprints):
        # do not spend model calls planning on geometry the person has not accepted
        run.deferred_reason = (blocking_text(run.issues) or
                               "low-fit footprints need review") + "; the planner agent runs after you confirm"
        return run
    _plan_with_agent(run, planner)
    return run


def _plan_with_agent(run: Run, planner) -> None:
    caps = agent_caps()
    ctx = ToolContext(run.footprints, run.board_w, run.board_h, run.anchors, caps)
    user = {"constraint_text": run.text, "budget": {"max_tool_calls": caps.max_steps}}
    run.ctx, run.agent = ctx, AgentState.new("plan", SYSTEM_PLAN, user, caps)
    agent_loop.drive(run.agent, ctx, planner, run.trace)
    _absorb(run)


def _absorb(run: Run) -> None:
    st, ctx = run.agent, run.ctx
    if st.status == "needs_input":
        run.status = "needs_input"
        return
    run.status = "needs_confirmation"
    if st.status == "finished":
        plan = ctx.plans[st.final_plan_id]
        locked = set(run.human_locks) if run.human_locks is not None else set(plan.candidate_set["locked_ids"])
        run.constraint = Constraint(ctx.start_name, ctx.end_name, ctx.clearance_cm, plan.candidate_set["max_moves"],
                                    sorted(locked), "agent")
        run.start, run.end = tuple(ctx.anchors[ctx.start_name]), tuple(ctx.anchors[ctx.end_name])
        run.footprints = [f.locked_as(f.id in locked) for f in run.footprints]
        run.plan, run.route_initial, run.plan_source, run.error = plan, plan.route_before, "agent", None
        return
    run.plan = None
    run.error = (f"planner agent could not finish ({st.reason}); confirm the constraints manually "
                 "(clearance_cm, start_name, end_name) to continue with the deterministic planner")


def answer(run: Run, text: str) -> Run:
    if run.status != "needs_input" or run.agent is None or run.ctx is None:
        raise PipelineError(f"no question is pending in state {run.status}", 409)
    _, _, planner = get_providers()
    try:
        agent_loop.answer(run.agent, text)
    except ValueError as e:
        raise PipelineError(str(e)) from e
    agent_loop.drive(run.agent, run.ctx, planner, run.trace)
    _absorb(run)
    return run


def _known_constraint(run: Run) -> dict:
    if run.constraint:
        c = run.constraint
        return {"clearance_cm": c.clearance_cm, "start_name": c.start_name, "end_name": c.end_name,
                "max_moves": c.max_moves}
    if run.ctx and run.ctx.clearance_cm is not None:
        return {"clearance_cm": run.ctx.clearance_cm, "start_name": run.ctx.start_name,
                "end_name": run.ctx.end_name, "max_moves": 2}
    return {}


def _manual_constraint(run: Run, manual: dict | None) -> Constraint:
    m = {**_known_constraint(run), **{k: v for k, v in (manual or {}).items() if v is not None}}
    missing = [k for k in ("clearance_cm", "start_name", "end_name") if k not in m]
    if missing:
        why = run.agent.reason if run.agent and run.agent.reason else "no agent plan"
        raise PipelineError(f"cannot plan ({why}): provide {missing} in the confirm request", 409)
    caps = agent_caps()
    try:
        c = float(m["clearance_cm"])
        mm = int(m.get("max_moves", 2))
    except (TypeError, ValueError):
        raise PipelineError("clearance_cm and max_moves must be numbers") from None
    s, e = str(m["start_name"]).strip().lower(), str(m["end_name"]).strip().lower()
    if not (20 < c < 300):
        raise PipelineError("clearance_cm must be between 20 and 300")
    if not (1 <= mm <= caps.max_moves_cap):
        raise PipelineError(f"max_moves must be 1 to {caps.max_moves_cap}")
    for n in (s, e):
        if n not in run.anchors:
            raise PipelineError(f"route endpoint '{n}' has no anchor point on the board; mark it", 409)
    if s == e:
        raise PipelineError("start and end must differ")
    return Constraint(s, e, c, mm, [], "human")


def confirm(run: Run, locked_ids: list[str] | None = None, accept_review_ids: list[str] | None = None,
            grid_step_cm: float = 10.0, time_budget_s: float = 8.0, manual: dict | None = None) -> Run:
    if run.status not in ("needs_confirmation", "awaiting_recapture", "failed") or not run.footprints:
        raise PipelineError(f"cannot confirm in state {run.status}", 409)
    if run.calibration is None:
        raise PipelineError("run has no calibration", 409)
    with step(run.trace, "Human", "confirm footprints and locks", "person", f"{len(run.footprints)} footprints",
              kind="human") as s:
        if locked_ids is not None:
            bad = [i for i in locked_ids if i not in {f.id for f in run.footprints}]
            if bad:
                raise PipelineError(f"unknown ids {bad}")
            run.human_locks = set(locked_ids)
            run.footprints = [f.locked_as(f.id in run.human_locks) for f in run.footprints]
        hard = [i for i in run.issues if not i.get("acceptable")]
        if hard:
            raise PipelineError(f"cannot continue: {blocking_text(hard)}", 409)
        pending = [f.id for f in run.footprints if f.needs_review and f.id not in (accept_review_ids or [])]
        if pending:
            raise PipelineError(f"footprints need review/acceptance: {pending}", 409)
        s.out(f"locks {[f.id for f in run.footprints if f.locked]}")
    if run.agent is None and run.plan is None:
        _, _, planner = get_providers()
        _plan_with_agent(run, planner)
        if run.status == "needs_input":
            return run
    cur_locks = {f.id for f in run.footprints if f.locked}
    replan = manual is not None or run.plan is None
    if run.plan is not None and not replan:
        cs = run.plan.candidate_set
        replan = (set(cs["locked_ids"]) != cur_locks or cs["grid_step_cm"] != grid_step_cm
                  or cs["time_budget_s"] != time_budget_s)
    if replan:
        c = _manual_constraint(run, manual)
        if run.human_locks is None:
            ends = {c.start_name, c.end_name}
            cur_locks |= {f.id for f in run.footprints if f.label in ends}
            run.footprints = [f.locked_as(f.id in cur_locks) for f in run.footprints]
        c.locked_ids = sorted(cur_locks)
        run.constraint = c
        run.start, run.end = tuple(run.anchors[c.start_name]), tuple(run.anchors[c.end_name])
        run.status = "planning"
        with step(run.trace, "Planner", "deterministic search with human-confirmed constraints",
                  "none (deterministic)", f"grid {grid_step_cm} cm, budget {time_budget_s}s, "
                  f"locks {sorted(cur_locks)}", kind="deterministic") as s:
            run.plan = plan_moves(run.footprints, run.board_w, run.board_h, run.start, run.end, c.clearance_cm,
                                  c.max_moves, grid_step_cm, time_budget_s)
            run.route_initial = run.plan.route_before
            run.plan_source = "manual" if manual else "human_replan"
            s.out(f"{run.plan.status}: {[m.id for m in run.plan.moves]} {run.plan.total_displacement_cm:.1f} cm")
    run.plan_confirmed = True
    run.status = "ready" if run.plan.status == "already_open" else (
        "awaiting_recapture" if run.plan.moves else "failed")
    run.error = run.plan.message if run.status == "failed" else None
    return run


def recapture(run: Run, photo: bytes) -> Run:
    if run.status != "awaiting_recapture" or run.plan is None:
        raise PipelineError(f"cannot recapture in state {run.status}", 409)
    perception, _, planner = get_providers()
    run.photos["recapture"] = photo
    run.status = "verifying"
    try:
        cal2, observed, _, issues = _perceive_and_calibrate(run, photo, "recapture", perception)
        if issues:
            raise PipelineError(f"recapture perception issues: {blocking_text(issues)}; retake the photo")
        run.cal2, run.observed = cal2, observed
        with step(run.trace, "Verifier", "recaptured poses vs expected, recompute route", "none (deterministic)",
                  f"{len(run.observed)} observed, {len(run.plan.moves)} planned moves") as s:
            run.verification = verify(run.footprints, run.plan, run.observed, run.board_w, run.board_h,
                                      run.start, run.end, run.constraint.clearance_cm)
            s.out(run.verification["verdict"])
    except (ProviderError, CalibrationError, ValueError, PipelineError) as e:
        run.status, run.error = "awaiting_recapture", str(e)
        raise PipelineError(str(e)) from e
    v = run.verification
    run.history.append({"kind": "recovery_plan" if run.recoveries_used else "plan", "verdict": v["verdict"],
                        "n_moves": len(run.plan.moves), "problems": v["problems"]})
    run.status = "ready" if v["pass"] else "needs_remeasurement"
    if not v["pass"] and v["verdict"] in RECOVERABLE and run.recoveries_used < MAX_RECOVERIES:
        _recover(run, planner)
    return run


def _instruction(plan: Plan) -> str:
    if not plan.moves:
        return plan.message
    return " ".join(f"Move {m.id} from ({m.from_center[0]:.0f}, {m.from_center[1]:.0f}) cm to "
                    f"({m.to_center[0]:.0f}, {m.to_center[1]:.0f}) cm ({m.displacement_cm:.0f} cm)."
                    for m in plan.moves) + " Then take a new photo."


def _recover(run: Run, planner) -> None:
    """One recovery turn: the agent re-plans from the OBSERVED layout (locks kept) with the same tools."""
    run.recoveries_used += 1
    c, caps, v = run.constraint, agent_caps(), run.verification
    ends = {c.start_name, c.end_name}
    lock_ids = {f.id for f in run.footprints if f.locked}
    obs = [f.locked_as(f.id in lock_ids or f.label in ends) for f in run.observed]
    ctx = ToolContext(obs, run.board_w, run.board_h, run.anchors, caps,
                      required_lock_ids=frozenset(f.id for f in obs if f.locked), allow_ask_user=False,
                      clearance_cm=c.clearance_cm, start_name=c.start_name, end_name=c.end_name)
    user = {"phase": "recovery", "clearance_cm": c.clearance_cm, "start": c.start_name, "end": c.end_name,
            "max_moves": c.max_moves, "locked_ids": sorted(f.id for f in obs if f.locked),
            "verdict": v["verdict"], "problems": v["problems"][:8],
            "budget": {"max_tool_calls": caps.recovery_max_steps}}
    st = AgentState.new("recovery", SYSTEM_RECOVERY, user, caps)
    agent_loop.drive(st, ctx, planner, run.trace)
    rec = {"agent": st.view(), "verdict_before": v["verdict"], "agent_note": ctx.summary}
    run.recovery = rec
    if st.status != "finished":
        rec.update(status="unresolved", instruction=None,
                   message=f"recovery agent could not finish ({st.reason}); retake the photo or replan manually")
        return
    plan = ctx.plans[st.final_plan_id]
    rec.update(status=plan.status, instruction=_instruction(plan), n_moves=len(plan.moves),
               route_after_status=(plan.route_after or {}).get("status"), message=plan.message)
    if plan.moves:  # rotate: the observed layout is the new baseline for the next verification
        run.footprints, run.plan, run.calibration = obs, plan, run.cal2
        run.plan_source, run.status = "recovery", "awaiting_recapture"
        run.error = None


def _agent_view(run: Run) -> dict:
    if run.agent is None:
        return {"status": "deferred" if run.deferred_reason else "not_run", "reason": run.deferred_reason}
    v = run.agent.view()
    v["note"] = run.ctx.summary if run.ctx else ""  # model free text: shown, never trusted
    v["note_is_unverified"] = True
    return v


def run_view(run: Run) -> dict:
    cal = run.calibration
    return {"id": run.id, "status": run.status, "error": run.error, "scope": SCOPE, "provider": provider_mode(),
            "calibration": cal.summary() if cal else None,
            "constraint": asdict(run.constraint) if run.constraint else None,
            "start_cm": run.start, "end_cm": run.end,
            "footprints": [f.to_dict(cal) for f in run.footprints],
            "route_initial": _route_px(run.route_initial, cal),
            "plan": _plan_view(run), "plan_source": run.plan_source, "plan_confirmed": run.plan_confirmed,
            "agent": _agent_view(run), "issues": run.issues, "recovery": run.recovery, "attempts": run.history,
            "observed": [f.to_dict(run.cal2) for f in run.observed],
            "verification": _verif_view(run), "perception_notes": run.perception_notes}


def trace_view(run: Run) -> dict:
    return {"run_id": run.id, "status": run.status, "steps": run.trace, "totals": totals(run.trace),
            "agent": _agent_view(run), "recovery_agent": (run.recovery or {}).get("agent")}


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
