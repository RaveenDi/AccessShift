"""Deterministic server-side tools for the planner agent.

The model chooses WHICH tool to call and with what arguments; it never computes geometry and can never
declare a route open. Only check_route / search_moves results carry route status, and `finish` accepts only a
plan_id that search_moves produced. Bad tool names or arguments raise ToolError, which the loop turns into a
tool result the model can recover from. Results are deterministic (no clocks, no random ids) so a whole
agent run can be replayed from recordings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import AgentCaps
from ..domain.connectivity import CELL_CM, check_route
from ..domain.footprints import Footprint
from ..domain.search import Plan, input_hash, plan_moves


class ToolError(Exception):
    pass


_PLAN_CACHE: dict[str, Plan] = {}  # complete deterministic results only; keyed by the plan's input hash
_CACHE_MAX = 64


@dataclass
class AskUser:
    question: str


def _spec(name, description, props, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(required),
                           "additionalProperties": False}}


TOOL_SPECS = [
    _spec("get_layout", "Measured layout: board size, anchor points (cm) and every object's id, label, polygon "
          "and lock flag. Takes no arguments.", {}),
    _spec("check_route", "Deterministic geometry check of the route at a clear width. Returns status "
          "open|borderline|blocked. Sets the route used by search_moves.",
          {"clearance_cm": {"type": "number", "description": "clear route width in cm, between 20 and 300"},
           "start": {"type": "string", "description": "anchor name; optional if exactly two anchors exist"},
           "end": {"type": "string", "description": "anchor name; optional if exactly two anchors exist"}},
          ["clearance_cm"]),
    _spec("search_moves", "Deterministic bounded search for the smallest rearrangement that opens the route "
          "(needs a prior check_route). Objects in locked_ids never move. Returns a plan_id.",
          {"max_moves": {"type": "integer", "description": "1 to 3"},
           "locked_ids": {"type": "array", "items": {"type": "string"},
                          "description": "object ids that must stay where they are"}},
          ["max_moves"]),
    _spec("explain_move", "Deterministic facts about one move of the latest search_moves plan: whether the "
          "route still opens without it.", {"move_id": {"type": "string", "description": "id of a moved object"}},
          ["move_id"]),
    _spec("ask_user", "Pause and ask the person ONE short question when the route width, an endpoint or which "
          "object is meant is genuinely ambiguous.", {"question": {"type": "string"}}, ["question"]),
    _spec("finish", "End the run with a plan_id returned by search_moves (also use it when the search found no "
          "solution). summary is optional free text for the person; it is never treated as a fact.",
          {"plan_id": {"type": "string"}, "summary": {"type": "string"}}, ["plan_id"]),
]
ALLOWED_ARGS = {s["name"]: set(s["parameters"]["properties"]) for s in TOOL_SPECS}


@dataclass
class ToolContext:
    footprints: list[Footprint]
    board_w: float
    board_h: float
    anchors: dict
    caps: AgentCaps
    required_lock_ids: frozenset = frozenset()
    allow_ask_user: bool = True
    grid_step_cm: float = 10.0
    time_budget_s: float = 8.0
    clearance_cm: float | None = None
    start_name: str | None = None
    end_name: str | None = None
    asks: int = 0
    plans: dict = field(default_factory=dict)
    last_plan_id: str | None = None
    final_plan_id: str | None = None
    summary: str = ""

    def ids(self) -> list[str]:
        return [f.id for f in self.footprints]


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _str(args, key, lo=1, hi=300) -> str:
    v = args.get(key)
    if not isinstance(v, str) or not (lo <= len(v.strip()) <= hi):
        raise ToolError(f"'{key}' must be a string of {lo}-{hi} characters")
    return v.strip()


def _route_view(r, start, end, clearance) -> dict:
    return {"status": r.status, "clearance_cm": clearance, "start": start, "end": end,
            "margin_cm": round(r.margin_cm, 2), "witness_min_gap_cm": r.witness_clearance_cm,
            "note": r.note, "source": "deterministic geometry"}


def t_get_layout(ctx: ToolContext, args: dict) -> dict:
    return {"board_cm": [ctx.board_w, ctx.board_h],
            "anchors_cm": {k: [round(v[0], 1), round(v[1], 1)] for k, v in ctx.anchors.items()},
            "objects": [{"id": f.id, "label": f.label, "locked": f.locked or f.id in ctx.required_lock_ids,
                         "center_cm": [round(v, 1) for v in f.center], "needs_review": f.needs_review,
                         "polygon_cm": [[round(x, 1), round(y, 1)] for x, y in list(f.polygon.exterior.coords)[:-1]]}
                        for f in ctx.footprints]}


def _endpoints(ctx: ToolContext, start, end) -> tuple[str, str]:
    names = list(ctx.anchors)
    if start is None and end is None:
        if len(names) == 2:
            return names[0], names[1]
        if ctx.start_name and ctx.end_name:
            return ctx.start_name, ctx.end_name
        raise ToolError(f"route endpoints are ambiguous: anchors are {names}; pass start and end, or ask_user")
    if start is None or end is None:
        raise ToolError("pass both start and end, or neither")
    s, e = str(start).strip().lower(), str(end).strip().lower()
    for n in (s, e):
        if n not in ctx.anchors:
            raise ToolError(f"unknown anchor '{n}'; anchors are {names}")
    if s == e:
        raise ToolError("start and end must be different anchors")
    return s, e


def t_check_route(ctx: ToolContext, args: dict) -> dict:
    c = args.get("clearance_cm")
    if not _is_num(c) or not (20 < c < 300):
        raise ToolError("'clearance_cm' must be a number greater than 20 and less than 300")
    s, e = _endpoints(ctx, args.get("start"), args.get("end"))
    r = check_route(ctx.footprints, ctx.board_w, ctx.board_h, tuple(ctx.anchors[s]), tuple(ctx.anchors[e]), float(c))
    ctx.clearance_cm, ctx.start_name, ctx.end_name = float(c), s, e
    return _route_view(r, s, e, float(c))


def _effective_locks(ctx: ToolContext, locked_ids) -> set[str]:
    ends = {ctx.start_name, ctx.end_name}
    return set(locked_ids) | set(ctx.required_lock_ids) | {f.id for f in ctx.footprints if f.label in ends}


def t_search_moves(ctx: ToolContext, args: dict) -> dict:
    if ctx.clearance_cm is None or ctx.start_name is None:
        raise ToolError("call check_route first so the clearance and endpoints are set")
    mm = args.get("max_moves")
    if not isinstance(mm, int) or isinstance(mm, bool) or not (1 <= mm <= ctx.caps.max_moves_cap):
        raise ToolError(f"'max_moves' must be an integer from 1 to {ctx.caps.max_moves_cap}")
    locked = args.get("locked_ids", [])
    if not isinstance(locked, list) or len(locked) > 20 or not all(isinstance(i, str) for i in locked):
        raise ToolError("'locked_ids' must be a list of up to 20 object ids")
    unknown = [i for i in locked if i not in ctx.ids()]
    if unknown:
        raise ToolError(f"unknown object ids {unknown}; valid ids are {ctx.ids()}")
    eff = _effective_locks(ctx, locked)
    fps = [f.locked_as(f.id in eff) for f in ctx.footprints]
    s_xy, e_xy = tuple(ctx.anchors[ctx.start_name]), tuple(ctx.anchors[ctx.end_name])
    # the key must cover everything plan_moves reads: footprints (geometry, locks, margins), board, endpoints,
    # clearance, search bounds and raster cell; it is also the plan_id, so it equals plan.input_hash
    key = input_hash(fps, s_xy, e_xy, ctx.clearance_cm, [mm, ctx.grid_step_cm, CELL_CM], (ctx.board_w, ctx.board_h))
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = plan_moves(fps, ctx.board_w, ctx.board_h, s_xy, e_xy, ctx.clearance_cm, mm, ctx.grid_step_cm,
                          ctx.time_budget_s)
        if plan.complete and len(_PLAN_CACHE) < _CACHE_MAX:  # never cache time-budget-truncated searches
            _PLAN_CACHE[key] = plan
    pid = plan.input_hash
    ctx.plans[pid] = plan
    ctx.last_plan_id = pid
    ev = {k: v for k, v in plan.evaluated.items() if k != "seconds"}  # wall time is not replay-stable
    return {"plan_id": pid, "status": plan.status, "message": plan.message, "n_moves": len(plan.moves),
            "moves": [m.to_dict() for m in plan.moves],
            "total_displacement_cm": round(plan.total_displacement_cm, 1),
            "search_complete": plan.complete, "minimal_within_candidate_set": plan.complete and plan.status == "plan_found",
            "route_after_status": (plan.route_after or {}).get("status"), "locked_ids": sorted(eff),
            "candidate_set": {"grid_step_cm": plan.candidate_set["grid_step_cm"], "max_moves": mm},
            "evaluated": ev, "source": "deterministic search"}


def t_explain_move(ctx: ToolContext, args: dict) -> dict:
    mid = _str(args, "move_id", 1, 60)
    plan = ctx.plans.get(ctx.last_plan_id or "")
    if plan is None:
        raise ToolError("no plan yet; call search_moves first")
    mv = next((m for m in plan.moves if m.id == mid), None)
    if mv is None:
        raise ToolError(f"'{mid}' is not a move of the latest plan; moves are {[m.id for m in plan.moves]}")
    s, e = tuple(ctx.anchors[ctx.start_name]), tuple(ctx.anchors[ctx.end_name])
    by = {f.id: f for f in ctx.footprints}
    only = [f.moved_to(*mv.to_center) if f.id == mid else f for f in ctx.footprints]
    reverted = [by[mid] if f.id == mid else f for f in plan.layout]
    def st(fps):
        return check_route(fps, ctx.board_w, ctx.board_h, s, e, ctx.clearance_cm).status
    without = st(reverted)
    return {"move": mv.to_dict(), "route_before": (plan.route_before or {}).get("status"),
            "route_with_only_this_move": st(only), "route_if_this_move_skipped": without,
            "this_move_is_necessary": without != "open", "source": "deterministic geometry"}


def t_ask_user(ctx: ToolContext, args: dict):
    if not ctx.allow_ask_user:
        raise ToolError("ask_user is not available in this phase; use the information you have")
    q = _str(args, "question", 3, 300)
    if ctx.asks >= ctx.caps.max_ask_user:
        raise ToolError("ask_user limit reached; proceed with the clearest interpretation or finish")
    return AskUser(q)


def t_finish(ctx: ToolContext, args: dict) -> dict:
    pid = _str(args, "plan_id", 1, 64)
    plan: Plan | None = ctx.plans.get(pid)
    if plan is None:
        raise ToolError(f"unknown plan_id; use a plan_id returned by search_moves (known: {sorted(ctx.plans)})")
    summary = args.get("summary", "")
    if not isinstance(summary, str):
        raise ToolError("'summary' must be a string")
    ctx.final_plan_id, ctx.summary = pid, " ".join(summary.split())[:400]
    return {"accepted": True, "plan_status": plan.status, "n_moves": len(plan.moves),
            "route_after_status": (plan.route_after or {}).get("status")}


HANDLERS = {"get_layout": t_get_layout, "check_route": t_check_route, "search_moves": t_search_moves,
            "explain_move": t_explain_move, "ask_user": t_ask_user, "finish": t_finish}


def execute(ctx: ToolContext, name: str, args) -> dict | AskUser:
    if name not in HANDLERS:
        raise ToolError(f"unknown tool '{name}'; available tools are {sorted(HANDLERS)}")
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    extra = set(args) - ALLOWED_ARGS[name]
    if extra:
        raise ToolError(f"unexpected argument(s) {sorted(extra)}; allowed: {sorted(ALLOWED_ARGS[name])}")
    return HANDLERS[name](ctx, args)
