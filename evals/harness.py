"""Evaluation harness: score the whole agentic pipeline against hand-written ground truth.

Rules (see evals/README.md): ground truth is written BEFORE running; tune on `dev` only; report `heldout`
once; ground truth is never sent to a model (only photo, board size, check length, anchors and constraint
text go in, and an outgoing-request guard rejects anything that looks like ground truth).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "evals"))

from app import pipeline  # noqa: E402
from app.domain.connectivity import check_route  # noqa: E402
from app.domain.footprints import Footprint, rect_footprint  # noqa: E402
from app.providers.http import BUDGET, CallBudgetExceeded, ModelClient  # noqa: E402
from app.providers.recording import ReplayMissError  # noqa: E402
from app.telemetry import totals  # noqa: E402
from bruteforce import min_moves_up_to_one  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

SPLITS = ("dev", "heldout")
GT_MARGIN_CM = 2.0  # measurement uncertainty assumed for hand-written ground-truth footprints
FORBIDDEN_KEYS = frozenset({"ground_truth", "expected_min_moves", "route_open", "footprints_mm", "recapture",
                            "expect_review", "clearance_mm", "route_expected_status", "traps", "split",
                            "bruteforce"})
EVIDENCE = {
    "mock": "PLUMBING ONLY: OpenCV colour segmentation and a scripted policy on synthetic images. This says "
            "nothing about Meta models and must not be quoted as model performance.",
    "replay": "Replays recorded Meta responses. Numbers are exactly as good as the recording run; prompt, image "
              "or tool-result changes cause a ReplayMissError rather than a guess.",
    "meta": "Live Meta Model API run (billed, capped by --max-calls). Endpoint and field shapes were UNVERIFIED "
            "until this run succeeded; prices are UNVERIFIED placeholders.",
}


class GroundTruthLeak(Exception):
    pass


# ----------------------------------------------------------------------------------------- request guard
def assert_clean_request(payload, secrets=()) -> None:
    """Raise if an outgoing model request contains ground-truth field names or per-case secret strings."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in FORBIDDEN_KEYS:
                    raise GroundTruthLeak(f"outgoing request contains forbidden key {k!r}")
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(payload)
    text = json.dumps(payload, default=str)
    for s in secrets:
        if s and s in text:
            raise GroundTruthLeak("outgoing request contains a ground-truth/case-identifying string")


_SECRETS: set[str] = set()
_ORIG_POST = ModelClient.post


def install_guard() -> None:
    def guarded(self, path, payload, **kw):
        assert_clean_request(payload, _SECRETS)
        return _ORIG_POST(self, path, payload, **kw)
    ModelClient.post = guarded


def uninstall_guard() -> None:
    ModelClient.post = _ORIG_POST


# ----------------------------------------------------------------------------------------- manifest
class Case:
    def __init__(self, raw: dict, base: Path):
        self.raw, self.base = raw, base
        self.id, self.split = raw["id"], raw["split"]
        self.gt = raw["ground_truth"]
        self.traps = raw.get("traps", [])
        self.answers = raw.get("answers", [])
        self.secrets = {self.id, str(self.gt.get("notes", ""))} - {""}

    def path(self, rel: str) -> Path:
        for b in (ROOT, self.base):
            if (b / rel).exists():
                return b / rel
        raise FileNotFoundError(f"{rel} not found; for the demo manifest run `python scripts/make_demo.py`")

    def model_inputs(self) -> dict:
        """The ONLY data the pipeline (and therefore any model) receives. Ground truth is not in here."""
        r = self.raw
        return {"photo": self.path(r["photo"]).read_bytes(), "board_w": r["board_mm"][0] / 10,
                "board_h": r["board_mm"][1] / 10, "check_len": r["check_len_mm"] / 10, "text": r["constraint_text"],
                "anchors": {k: [v[0] / 10, v[1] / 10] for k, v in r["anchors_mm"].items()}}

    def gt_footprints(self, locked_ids=()) -> list[Footprint]:
        out = []
        for g in self.gt["footprints_mm"]:
            if "polygon_mm" in g:
                poly = Polygon([(x / 10, y / 10) for x, y in g["polygon_mm"]])
                out.append(Footprint(g["id"], g.get("label", g["id"]), poly, GT_MARGIN_CM, g["id"] in locked_ids,
                                     source="ground_truth"))
            else:
                out.append(rect_footprint(g["id"], g.get("label", g["id"]), g["x"] / 10, g["y"] / 10, g["w"] / 10,
                                          g["h"] / 10, g["id"] in locked_ids, GT_MARGIN_CM))
        return out


def load_manifest(path: Path, split: str | None = None) -> list[Case]:
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw = json.loads(line)
            for k in ("id", "photo", "board_mm", "check_len_mm", "anchors_mm", "constraint_text", "split",
                      "ground_truth"):
                if k not in raw:
                    raise KeyError(k)
            if raw["split"] not in SPLITS:
                raise ValueError(f"split must be one of {SPLITS}")
        except (ValueError, KeyError) as e:
            raise SystemExit(f"manifest line {n}: bad or missing field {e}") from e
        if split is None or raw["split"] == split:
            cases.append(Case(raw, Path(path).parent))
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit("manifest ids must be unique")
    return cases


def _constraint_gt(case: Case):
    c = case.gt.get("constraint")
    if not c:
        return None
    ends = {c["start"], c["end"]}
    fps = case.gt_footprints()
    locked = set(c.get("locked_ids", [])) | {f.id for f in fps if f.label in ends}
    return {"clearance_cm": c["clearance_mm"] / 10, "start": c["start"], "end": c["end"], "locked": locked,
            "max_moves": c.get("max_moves", 2), "ends": ends}


def validate_case(case: Case, bruteforce: bool = True) -> list[str]:
    """Author-time checks that the hand-written ground truth agrees with the deterministic geometry."""
    msgs = []
    case.path(case.raw["photo"])
    for rc in case.gt.get("recapture", []):
        case.path(rc["photo"])
        if rc.get("expected") not in ("correct", "wrong"):
            msgs.append(f"{case.id}: recapture.expected must be correct|wrong")
    cg = _constraint_gt(case)
    if cg is None:
        return msgs + [f"{case.id}: no ground_truth.constraint; route/brute-force checks skipped"]
    bw, bh = case.raw["board_mm"][0] / 10, case.raw["board_mm"][1] / 10
    fps = case.gt_footprints(cg["locked"])
    s, e = (tuple(v / 10 for v in case.raw["anchors_mm"][cg[k]]) for k in ("start", "end"))
    phys = check_route(fps, bw, bh, s, e, cg["clearance_cm"], margin_cm=0.0).status == "open"
    if "borderline_clearance" in case.traps:  # physically open by construction; the system must say borderline
        st = check_route(fps, bw, bh, s, e, cg["clearance_cm"]).status
        if st != case.gt.get("route_expected_status"):
            msgs.append(f"{case.id}: route_expected_status={case.gt.get('route_expected_status')} but geometry says {st}")
    elif phys != bool(case.gt["route_open"]):
        msgs.append(f"{case.id}: ground_truth.route_open={case.gt['route_open']} but the geometry check on the "
                    f"ground-truth footprints says {phys}")
    if bruteforce and case.gt.get("expected_min_moves") in (0, 1, None):
        bf = min_moves_up_to_one(fps, bw, bh, s, e, cg["clearance_cm"])
        exp = case.gt.get("expected_min_moves")
        if (exp in (0, 1) and bf != exp) or (exp is None and bf is not None):
            msgs.append(f"{case.id}: expected_min_moves={exp} but coarse brute force found {bf}")
    return msgs


# ----------------------------------------------------------------------------------------- metrics
def match_footprints(pred: list[Footprint], gt: list[Footprint], fallback_cm: float = 25.0):
    cand = sorted((math.dist(p.center, g.center), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)
                  if p.label == g.label)
    pairs, up, ug = [], set(), set()
    for d, i, j in cand:
        if i not in up and j not in ug:
            pairs.append((i, j, False))
            up.add(i), ug.add(j)
    for d, i, j in sorted((math.dist(p.center, g.center), i, j) for i, p in enumerate(pred)
                          for j, g in enumerate(gt)):
        if i not in up and j not in ug and d <= fallback_cm:
            pairs.append((i, j, True))
            up.add(i), ug.add(j)
    return pairs, [j for j in range(len(gt)) if j not in ug], [i for i in range(len(pred)) if i not in up]


def edge_error_mm(p: Footprint, g: Footprint) -> float:
    """Worst-edge error: Hausdorff distance between the two outlines, in mm."""
    return 10.0 * p.polygon.exterior.hausdorff_distance(g.polygon.exterior)


def _perception_metrics(run, case: Case) -> dict:
    gt = case.gt_footprints()
    pairs, missed, spurious = match_footprints(run.footprints, gt)
    errs = [edge_error_mm(run.footprints[i], gt[j]) for i, j, _ in pairs]
    return {"n_pred": len(run.footprints), "n_gt": len(gt), "matched": len(pairs), "missed_gt": [gt[j].id for j in missed],
            "spurious_pred": [run.footprints[i].id for i in spurious], "label_mismatch": sum(1 for *_, m in pairs if m),
            "edge_error_mm": [round(x, 2) for x in errs], "_pairs": [(run.footprints[i].id, gt[j].id) for i, j, _ in pairs]}


def _plan_on_ground_truth(run, case: Case, cg, pairs) -> dict:
    """Apply the agent's moves to the GROUND-TRUTH layout and re-check the route there (independent of perception)."""
    gt = {f.id: f for f in case.gt_footprints()}
    pmap = dict(pairs)
    moved, ok_locks = {}, True
    for m in run.plan.moves:
        gid = pmap.get(m.id)
        if gid is None:
            return {"valid_on_gt": None, "note": f"moved object {m.id} has no ground-truth match"}
        ok_locks &= gid not in cg["locked"]
        f = gt[gid]
        moved[gid] = f.moved_to(f.center[0] + m.to_center[0] - m.from_center[0],
                                f.center[1] + m.to_center[1] - m.from_center[1])
    layout = [moved.get(i, f) for i, f in gt.items()]
    bw, bh = case.raw["board_mm"][0] / 10, case.raw["board_mm"][1] / 10
    s, e = (tuple(v / 10 for v in case.raw["anchors_mm"][cg[k]]) for k in ("start", "end"))
    st = check_route(layout, bw, bh, s, e, cg["clearance_cm"]).status
    return {"valid_on_gt": st == "open", "route_on_gt": st, "locks_respected_on_gt": ok_locks}


def run_case(case: Case, bruteforce: bool = True) -> dict:
    res = {"id": case.id, "split": case.split, "traps": case.traps, "errors": [], "outcome": "", "recapture": []}
    _SECRETS.clear()
    _SECRETS.update(case.secrets)
    t0 = time.perf_counter()
    cg = _constraint_gt(case)
    expect_review = bool(case.gt.get("expect_review"))
    try:
        run = pipeline.start_run(**case.model_inputs())
    except pipeline.PipelineError as e:
        res.update(outcome="failed", errors=[str(e)], wall_s=round(time.perf_counter() - t0, 2))
        return res
    for a in case.answers:
        if run.status == "needs_input":
            pipeline.answer(run, a)
    flagged = bool(run.issues) or any(f.needs_review for f in run.footprints)
    res["review"] = {"expected": expect_review, "flagged": flagged}
    perc = _perception_metrics(run, case) if run.footprints else None
    pairs = perc.pop("_pairs") if perc else []
    res["perception"] = perc
    trace = list(run.trace)
    if run.status == "needs_input":
        res["outcome"] = "needs_input_unanswered"
    elif expect_review:
        res["outcome"] = "flagged_for_review" if flagged else "NOT_flagged"
    else:
        try:
            pipeline.confirm(run, accept_review_ids=[f.id for f in run.footprints if f.needs_review])
        except pipeline.PipelineError as e:
            res["errors"].append(str(e))
        trace = list(run.trace)
        res["outcome"] = {"awaiting_recapture": "planned", "ready": "already_open", "failed": "no_solution"}.get(
            run.status, f"agent_{run.agent.status}" if run.agent else run.status)
    res["agent"] = run.agent.view() if run.agent else None
    # constraint interpretation vs ground truth
    if cg and run.constraint:
        c = run.constraint
        ends = {c.start_name, c.end_name}
        locks = set(c.locked_ids) | {f.id for f in run.footprints if f.label in ends}
        res["constraint"] = {"clearance_ok": abs(c.clearance_cm - cg["clearance_cm"]) <= 1.0,
                             "endpoints_ok": {c.start_name, c.end_name} == cg["ends"],
                             "locks_ok": locks == cg["locked"]}
    # route agreement
    r0 = run.route_initial
    if r0 is not None and not expect_review:
        pred_open, gt_open = r0["status"] == "open", bool(case.gt["route_open"])
        res["route"] = {"predicted_status": r0["status"], "gt_open": gt_open, "agree": pred_open == gt_open,
                        "false_open": pred_open and not gt_open,
                        "conservative_not_open": (not pred_open) and gt_open,
                        "status_matches_expected": (case.gt.get("route_expected_status") in (None, r0["status"]))}
    # plan quality
    if run.plan is not None and not expect_review:
        exp = case.gt.get("expected_min_moves", "unset")
        pl = {"status": run.plan.status, "n_moves": len(run.plan.moves), "expected_min_moves": None if exp == "unset" else exp,
              "total_displacement_cm": round(run.plan.total_displacement_cm, 1)}
        if exp != "unset":
            pl["optimal"] = (exp is None and not run.plan.moves and run.plan.status == "no_solution_in_set") or (
                exp is not None and len(run.plan.moves) == exp and run.plan.status in ("plan_found", "already_open"))
        if cg:
            bw, bh = case.raw["board_mm"][0] / 10, case.raw["board_mm"][1] / 10
            s, e = (tuple(v / 10 for v in case.raw["anchors_mm"][cg[k]]) for k in ("start", "end"))
            if bruteforce:
                bf = min_moves_up_to_one(case.gt_footprints(cg["locked"]), bw, bh, s, e, cg["clearance_cm"])
                pl["bruteforce_min_moves"] = bf  # None = more than one move (or none) on the coarse grid
                if bf is not None:
                    pl["agrees_with_bruteforce"] = len(run.plan.moves) == bf
                else:
                    pl["agrees_with_bruteforce"] = len(run.plan.moves) != 1 or run.plan.status != "plan_found"
            if run.plan.moves:
                pl.update(_plan_on_ground_truth(run, case, cg, pairs))
        res["plan"] = pl
    # recapture / verification / recovery, each from a copy of the planned run
    if run.status == "awaiting_recapture":
        for rc in case.gt.get("recapture", []):
            clone = copy.deepcopy(run)
            n0 = len(clone.trace)
            entry = {"photo": rc["photo"], "expected": rc["expected"]}
            try:
                pipeline.recapture(clone, case.path(rc["photo"]).read_bytes())
                v = clone.verification
                entry.update(verdict=v["verdict"], detected_wrong=v["verdict"] != "verified_open",
                             problems=v["problems"][:4])
                if clone.recovery:
                    entry["recovery"] = {k: clone.recovery.get(k) for k in ("status", "n_moves", "route_after_status")}
            except pipeline.PipelineError as e:
                entry["error"] = str(e)
            res["recapture"].append(entry)
            trace += clone.trace[n0:]
    res["telemetry"] = totals(trace)
    res["wall_s"] = round(time.perf_counter() - t0, 2)
    return res


# ----------------------------------------------------------------------------------------- aggregation
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _rate(num, den):
    return {"n": num, "of": den, "rate": round(num / den, 3) if den else None}


def aggregate(results: list[dict]) -> dict:
    done = [r for r in results if r.get("outcome") != "aborted"]
    edges = [e for r in done if not r.get("review", {}).get("expected") for e in (r.get("perception") or {}).get("edge_error_mm", [])]
    routes = [r["route"] for r in done if "route" in r]
    plans = [r["plan"] for r in done if "plan" in r]
    cons = [r["constraint"] for r in done if "constraint" in r]
    rcs = [(c["expected"], c) for r in done for c in r["recapture"] if "verdict" in c]
    wrong = [c for ex, c in rcs if ex == "wrong"]
    right = [c for ex, c in rcs if ex == "correct"]
    wrong_blocked = [c for c in wrong if c["verdict"] in ("route_still_blocked", "route_borderline")]
    reviews = [r["review"] for r in done if r.get("review", {}).get("expected")]
    tele = [r["telemetry"] for r in done if "telemetry" in r]
    agg = {
        "cases": len(results), "cases_completed": len(done),
        "outcomes": {o: sum(r["outcome"] == o for r in results) for o in sorted({r["outcome"] for r in results})},
        "footprint_edge_error_mm": {"n_objects": len(edges), "mean": _mean(edges), "max": max(edges) if edges else None,
                                    "within_5mm": _rate(sum(e <= 5 for e in edges), len(edges)),
                                    "note": "worst-edge (Hausdorff) per matched object; objects in expect_review cases excluded"},
        "objects_missed_or_spurious": {"missed_gt": sum(len((r.get("perception") or {}).get("missed_gt", [])) for r in done),
                                       "spurious_pred": sum(len((r.get("perception") or {}).get("spurious_pred", [])) for r in done)},
        "constraint_interpretation": {k: _rate(sum(c[k] for c in cons), len(cons))
                                      for k in ("clearance_ok", "endpoints_ok", "locks_ok")},
        "route_agreement": {"agree": _rate(sum(r["agree"] for r in routes), len(routes)),
                            "false_open": sum(r["false_open"] for r in routes),
                            "conservative_not_open": sum(r["conservative_not_open"] for r in routes),
                            "note": "false_open (said open, physically not) is the unsafe error and should be 0"},
        "move_count_optimality": {"vs_expected": _rate(sum(p.get("optimal") is True for p in plans),
                                                       sum("optimal" in p for p in plans)),
                                  "vs_bruteforce": _rate(sum(p.get("agrees_with_bruteforce") is True for p in plans),
                                                         sum("agrees_with_bruteforce" in p for p in plans)),
                                  "plan_valid_on_ground_truth": _rate(sum(p.get("valid_on_gt") is True for p in plans),
                                                                      sum(p.get("valid_on_gt") is not None for p in plans if "valid_on_gt" in p)),
                                  "bruteforce_note": "exact for 0/1 moves on a 40 cm grid; None means more than one move"},
        "wrong_move_detection": {"wrong_detected": _rate(sum(c["detected_wrong"] for c in wrong), len(wrong)),
                                 "correct_accepted": _rate(sum(not c["detected_wrong"] for c in right), len(right)),
                                 "recovery_plan_found_when_route_not_open": _rate(
                                     sum((c.get("recovery") or {}).get("status") == "plan_found" for c in wrong_blocked),
                                     len(wrong_blocked))},
        "review_flag_for_trap_cases": _rate(sum(r["flagged"] for r in reviews), len(reviews)),
        "telemetry": {"agent_tool_calls": sum(t["agent_tool_calls"] for t in tele),
                      "agent_model_turns": sum(t["agent_model_turns"] for t in tele),
                      "api_calls": sum(t["api_calls"] for t in tele), "retries": sum(t["retries"] for t in tele),
                      "input_tokens": _mean([t["input_tokens"] for t in tele]) and sum(t["input_tokens"] or 0 for t in tele),
                      "output_tokens": _mean([t["output_tokens"] for t in tele]) and sum(t["output_tokens"] or 0 for t in tele),
                      "api_latency_ms": round(sum(t["api_latency_ms"] for t in tele), 1),
                      "wall_ms": round(sum(t["wall_ms"] for t in tele), 1),
                      "cost_usd_est": round(sum(t["cost_usd_est"] for t in tele), 6),
                      "per_case_mean_wall_ms": _mean([t["wall_ms"] for t in tele]),
                      "cost_basis": "UNVERIFIED placeholder prices; 0 when no billed call was made"},
    }
    return agg


def render_md(meta: dict, agg: dict, results: list[dict]) -> str:
    e, ro, mo, wm, tl = (agg["footprint_edge_error_mm"], agg["route_agreement"], agg["move_count_optimality"],
                         agg["wrong_move_detection"], agg["telemetry"])
    def r(x):
        return f"{x['n']}/{x['of']}" + (f" ({x['rate']:.0%})" if x["rate"] is not None else "")
    L = [f"# Eval report {meta['stamp']} - {meta['split']} - provider `{meta['provider']}`", "",
         f"> {meta['evidence_note']}", "",
         f"manifest `{meta['manifest']}` (sha256 {meta['manifest_sha256'][:12]}), commit `{meta['commit']}`, "
         f"cases {agg['cases_completed']}/{agg['cases']}" + (f", **ABORTED: {meta['aborted']}**" if meta.get("aborted") else ""), "",
         "## Metrics", "", "| metric | value |", "|---|---|",
         f"| footprint edge error (mm), mean / max | {e['mean']} / {e['max']} over {e['n_objects']} objects; within 5 mm: {r(e['within_5mm'])} |",
         f"| objects missed / spurious | {agg['objects_missed_or_spurious']['missed_gt']} / {agg['objects_missed_or_spurious']['spurious_pred']} |",
         f"| constraint read correctly (clearance / endpoints / locks) | {r(agg['constraint_interpretation']['clearance_ok'])} / {r(agg['constraint_interpretation']['endpoints_ok'])} / {r(agg['constraint_interpretation']['locks_ok'])} |",
         f"| route agreement with ground truth | {r(ro['agree'])}; false-open {ro['false_open']}; conservative not-open {ro['conservative_not_open']} |",
         f"| move count = expected minimum | {r(mo['vs_expected'])} |",
         f"| move count agrees with brute force | {r(mo['vs_bruteforce'])} |",
         f"| plan opens the route on ground-truth layout | {r(mo['plan_valid_on_ground_truth'])} |",
         f"| wrong moves detected | {r(wm['wrong_detected'])} |",
         f"| correct moves accepted | {r(wm['correct_accepted'])} |",
         f"| recovery plan found when a wrong move left the route not open | {r(wm['recovery_plan_found_when_route_not_open'])} |",
         f"| trap cases flagged for review | {r(agg['review_flag_for_trap_cases'])} |",
         f"| agent tool calls / model turns / API calls / retries | {tl['agent_tool_calls']} / {tl['agent_model_turns']} / {tl['api_calls']} / {tl['retries']} |",
         f"| tokens in / out | {tl['input_tokens']} / {tl['output_tokens']} (None = not returned) |",
         f"| latency: API ms / wall ms (mean wall per case) | {tl['api_latency_ms']} / {tl['wall_ms']} ({tl['per_case_mean_wall_ms']}) |",
         f"| estimated cost (USD, UNVERIFIED prices) | {tl['cost_usd_est']} |", "",
         "## Cases", "", "| case | traps | outcome | route | plan | recapture verdicts |", "|---|---|---|---|---|---|"]
    for c in results:
        rt = c.get("route", {})
        pl = c.get("plan", {})
        L.append(f"| {c['id']} | {', '.join(c['traps']) or '-'} | {c['outcome']} | {rt.get('predicted_status', '-')}"
                 f"{' (agree)' if rt.get('agree') else ' (conservative)' if rt.get('conservative_not_open') else ' (DISAGREE)' if rt else ''} | "
                 f"{pl.get('status', '-')} {pl.get('n_moves', '')} moves, optimal={pl.get('optimal', '-')}, valid_on_gt={pl.get('valid_on_gt', '-')} | "
                 + "; ".join(f"{x['expected']}->{x.get('verdict', x.get('error', '?'))}" for x in c["recapture"]) + " |")
    L += ["", "Per-case detail is in the JSON next to this file. Honest limits: see evals/README.md."]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------- CLI
def heldout_gate(report_dir: Path, manifest_sha: str, provider: str, allow: bool) -> Path | None:
    marker = report_dir / f".heldout-{manifest_sha[:8]}-{provider}"
    if provider != "mock" and marker.exists() and not allow:
        raise SystemExit(f"heldout already reported for this manifest and provider ({marker.name}). Report held-out "
                         "once; tuning on it invalidates it. Use --allow-heldout-rerun only for a deliberate re-run.")
    return marker if provider != "mock" else None


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the AccessShift agent pipeline against ground truth.")
    ap.add_argument("--provider", required=True, choices=["mock", "replay", "meta"])
    ap.add_argument("--manifest", default=str(ROOT / "evals" / "manifest.jsonl"))
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--max-calls", type=int, default=None, help="cap on live HTTP calls (required for --provider meta)")
    ap.add_argument("--out-dir", default=str(ROOT / "evals" / "reports"))
    ap.add_argument("--recordings", default=None, help="recordings folder (default evals/recordings)")
    ap.add_argument("--no-bruteforce", action="store_true")
    ap.add_argument("--validate-only", action="store_true", help="only check ground truth against the geometry")
    ap.add_argument("--allow-heldout-rerun", action="store_true")
    a = ap.parse_args(argv)

    if a.provider == "meta" and a.max_calls is None:
        print("--provider meta is billed: pass --max-calls N", file=sys.stderr)
        return 2
    mpath = Path(a.manifest)
    if not mpath.exists():
        print(f"manifest not found: {mpath}. For the synthetic demo run `python scripts/make_demo.py` and pass "
              "--manifest evals/manifests/demo.synthetic.jsonl", file=sys.stderr)
        return 2
    sha = hashlib.sha256(mpath.read_bytes()).hexdigest()
    cases = load_manifest(mpath, a.split)
    if not cases:
        print(f"no cases with split={a.split}", file=sys.stderr)
        return 2
    if a.validate_only:
        bad = [m for c in cases for m in validate_case(c, not a.no_bruteforce)]
        print("\n".join(bad) or "ground truth consistent with the geometry")
        return 1 if any("skipped" not in m for m in bad) else 0

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = heldout_gate(out, sha, a.provider, a.allow_heldout_rerun) if a.split == "heldout" else None
    os.environ["MODEL_PROVIDER"] = a.provider
    os.environ.setdefault("RECORDINGS_DIR", a.recordings or str(ROOT / "evals" / "recordings"))
    if a.recordings:
        os.environ["RECORDINGS_DIR"] = a.recordings
    if a.provider == "meta":
        BUDGET.set(a.max_calls)
    install_guard()
    results, aborted = [], None
    try:
        for c in cases:
            try:
                results.append(run_case(c, not a.no_bruteforce))
            except CallBudgetExceeded as e:
                aborted = str(e)
                results.append({"id": c.id, "split": c.split, "traps": c.traps, "outcome": "aborted", "errors": [aborted],
                                "recapture": []})
                break
            except ReplayMissError as e:
                aborted = str(e)
                results.append({"id": c.id, "split": c.split, "traps": c.traps, "outcome": "aborted", "errors": [aborted],
                                "recapture": []})
                break
    finally:
        uninstall_guard()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    agg = aggregate(results)
    meta = {"stamp": stamp, "split": a.split, "provider": a.provider, "manifest": str(mpath), "manifest_sha256": sha,
            "commit": _commit(), "max_calls": a.max_calls, "live_calls_used": BUDGET.used if a.provider == "meta" else 0,
            "evidence_note": EVIDENCE[a.provider], "aborted": aborted, "prices_verified": False}
    (out / f"{stamp}-{a.split}.json").write_text(json.dumps({"meta": meta, "aggregate": agg, "cases": results}, indent=1),
                                                  encoding="utf-8")
    (out / f"{stamp}-{a.split}.md").write_text(render_md(meta, agg, results), encoding="utf-8")
    if marker is not None and not aborted:
        marker.write_text(f"{stamp}\n", encoding="utf-8")
    print(f"wrote {out / (stamp + '-' + a.split)}.json/.md")
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
