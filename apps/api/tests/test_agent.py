import json

import cv2
import pytest
from fastapi.testclient import TestClient

from app import demo, pipeline
from app.agent import tools as T
from app.config import agent_caps
from app.domain.footprints import rect_footprint
from app.main import app
from app.providers import Providers
from app.providers.base import ModelTurn, ToolCall
from app.providers.mock import MockConstraint, MockPerception
from app.providers.mock_policy import MockPlanner
from conftest import img, start
from fake_meta import FakeMeta


class ScriptedPlanner:
    """Plays fixed turns: a list of (tool, args) calls, a callable(messages) -> calls, or a ModelTurn."""
    name, model = "scripted", "scripted-test"

    def __init__(self, turns):
        self.turns, self.i = list(turns), 0

    def next_turn(self, messages, tools):
        if self.i >= len(self.turns):
            return ModelTurn("I am done.", [], self.model)
        t = self.turns[self.i]
        self.i += 1
        t = t(messages) if callable(t) else t
        if isinstance(t, ModelTurn):
            return t
        return ModelTurn(None, [ToolCall(f"c{self.i}_{j}", n, a) for j, (n, a) in enumerate(t)], self.model)


def scripted(turns):
    return Providers(MockPerception(), MockConstraint(), ScriptedPlanner(turns))


def plan_id(msgs):
    return [json.loads(m["content"])["plan_id"] for m in msgs if m["role"] == "tool" and m["name"] == "search_moves"
            and "plan_id" in m["content"]][-1]


CHECK = ("check_route", {"clearance_cm": 90, "start": "door", "end": "desk"})
SEARCH = ("search_moves", {"max_moves": 2, "locked_ids": ["sofa", "shelf"]})


def test_full_pipeline_through_meta_wire_format(use_providers, demo_dir):
    fm = FakeMeta()
    use_providers(fm.providers())
    run = start()
    assert run.status == "needs_confirmation" and [m.id for m in run.plan.moves] == ["armchair"]
    reqs = [b for _, b in fm.requests if "tools" in b]
    assert len(reqs) == 5 and all(t["type"] == "function" and "parameters" in t["function"] for t in reqs[0]["tools"])
    last = reqs[-1]["messages"]
    calls = [c for m in last if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    tool_ids = [m["tool_call_id"] for m in last if m["role"] == "tool"]
    assert [c["id"] for c in calls] == tool_ids and all(isinstance(c["function"]["arguments"], str) for c in calls)
    pipeline.confirm(run)
    pipeline.recapture(run, img("correct"))
    assert run.verification["verdict"] == "verified_open" and run.status == "ready"


def test_bad_tool_calls_return_errors_not_crashes(use_providers, monkeypatch, demo_dir):
    monkeypatch.setenv("AGENT_MAX_STEPS", "14")
    monkeypatch.setenv("AGENT_MAX_MODEL_CALLS", "14")
    use_providers(scripted([
        [("get_layout", {})], [("nope", {})], [("check_route", {"clearance_cm": "wide"})],
        [("check_route", {"clearance_cm": 90, "bogus": 1})], [("search_moves", {"max_moves": 2})], [CHECK],
        [("search_moves", {"max_moves": 9})], [("search_moves", {"max_moves": 2, "locked_ids": ["ghost"]})],
        [("explain_move", {"move_id": "armchair"})], [("finish", {"plan_id": "made-up"})], [SEARCH],
        lambda m: [("finish", {"plan_id": plan_id(m), "summary": "ok"})]]))
    run = start()
    t = pipeline.trace_view(run)
    assert run.agent.status == "finished" and [m.id for m in run.plan.moves] == ["armchair"]
    errors = [s["output_summary"] for s in t["steps"] if s["status"] == "tool_error"]
    assert len(errors) == 8 and t["totals"]["tool_errors"] == 8
    assert any("unknown tool" in e for e in errors) and any("check_route first" in e for e in errors)


def test_model_cannot_declare_route_open(use_providers, demo_dir):
    use_providers(scripted([[("get_layout", {})], [("finish", {"plan_id": "x", "summary": "The route is open."})],
                            ModelTurn("The route is open, no move needed.", [], "s"),
                            ModelTurn("Really, it is open.", [], "s")]))
    run = start()
    assert run.agent.status == "unresolved" and run.agent.reason.startswith("malformed_output")
    assert run.plan is None and run.route_initial is None and run.status == "needs_confirmation"
    assert "could not finish" in run.error
    with pytest.raises(pipeline.PipelineError) as e:  # no manual constraints -> cannot plan
        pipeline.confirm(run)
    assert e.value.code == 409


def test_step_cap_is_an_honest_partial_status_and_manual_confirm_recovers(use_providers, demo_dir):
    use_providers(scripted([[("get_layout", {})], [CHECK], [SEARCH]] + [[("get_layout", {})]] * 20))
    run = start()
    a = run.agent
    assert a.status == "unresolved" and a.reason.startswith("step_cap") and a.steps == 8
    assert a.partial_plan_id and run.plan is None  # last search is reported, never promoted to a final plan
    assert run.status == "needs_confirmation" and "confirm the constraints manually" in run.error
    run = pipeline.confirm(run, manual={"clearance_cm": 90, "start_name": "door", "end_name": "desk"},
                           locked_ids=["sofa", "shelf", "desk"])
    assert run.plan_source == "manual" and run.status == "awaiting_recapture" and run.plan.moves[0].id == "armchair"
    assert run.constraint.source == "human"


def test_model_call_cap(use_providers, monkeypatch, demo_dir):
    monkeypatch.setenv("AGENT_MAX_MODEL_CALLS", "3")
    use_providers(scripted([[("get_layout", {})]] * 10))
    run = start()
    assert run.agent.status == "unresolved" and run.agent.reason.startswith("model_call_cap")
    assert run.agent.model_calls == 3 and agent_caps().max_model_calls == 3


def test_ask_user_pauses_then_resumes_via_api(use_providers, demo_dir):
    use_providers(Providers(MockPerception(), MockConstraint(), MockPlanner()))
    c = TestClient(app)
    r = c.post("/api/runs", files={"photo": ("a.png", img("blocked"), "image/png")}, data={
        "board_w_cm": demo.BOARD_W, "board_h_cm": demo.BOARD_H, "check_len_cm": demo.CHECK_LEN,
        "constraint_text": "Keep a clear route from the door to the desk. Lock the sofa and the shelf.",
        "anchors": json.dumps(demo.ANCHORS)})
    v = r.json()
    assert v["status"] == "needs_input" and "width" in v["agent"]["pending_question"]
    assert c.post(f"/api/runs/{v['id']}/confirm", json={}).status_code == 409  # paused runs cannot be confirmed
    v2 = c.post(f"/api/runs/{v['id']}/answer", json={"answer": "100 cm please"}).json()
    assert v2["status"] == "needs_confirmation" and v2["constraint"]["clearance_cm"] == 100
    assert v2["plan"]["moves"] and v2["agent"]["status"] == "finished"
    assert c.post(f"/api/runs/{v['id']}/answer", json={"answer": "again"}).status_code == 409
    steps = c.get(f"/api/runs/{v['id']}/trace").json()["steps"]
    assert any(s["role"] == "tool:ask_user" for s in steps)


def test_policy_asks_which_object_when_identity_is_ambiguous():
    lay = {"anchors_cm": {"door": [0, 0], "desk": [1, 1]},
           "objects": [{"id": "armchair", "label": "armchair"}, {"id": "armchair_2", "label": "armchair"}]}
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": json.dumps({"constraint_text": "90 cm route from the door to the desk. "
                                                                         "Lock the armchair."})},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "get_layout", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "1", "name": "get_layout", "content": json.dumps(lay)}]
    turn = MockPlanner().next_turn(msgs, [])
    assert turn.tool_calls[0].name == "ask_user" and "armchair" in turn.tool_calls[0].arguments["question"]


def test_locks_are_enforced_by_the_server_not_the_model(demo_dir):
    run = start()  # mock agent finished; footprints carry the agent's locks
    base = [f.locked_as(False) for f in run.footprints]
    ctx = T.ToolContext(base, run.board_w, run.board_h, run.anchors, agent_caps(),
                        required_lock_ids=frozenset({"armchair"}))
    T.execute(ctx, "check_route", {"clearance_cm": 90, "start": "door", "end": "desk"})
    r = T.execute(ctx, "search_moves", {"max_moves": 1, "locked_ids": []})
    assert "armchair" in r["locked_ids"] and "armchair" not in [m["id"] for m in r["moves"]]
    assert "desk" in r["locked_ids"]  # an endpoint object is always fixed


def test_recovery_turn_produces_corrective_plan_that_verifies_once(demo_dir):
    run = start()
    pipeline.confirm(run)
    pipeline.recapture(run, img("wrong"))
    assert run.verification["verdict"] == "route_still_blocked"
    rec = run.recovery
    assert rec["status"] == "plan_found" and rec["instruction"].startswith("Move") and rec["agent"]["phase"] == "recovery"
    assert run.status == "awaiting_recapture" and run.plan_source == "recovery"
    layout = dict(demo.scenes()["wrong"])
    for m in run.plan.moves:
        w, h = layout[m.id][2:]
        layout[m.id] = (m.to_center[0] - w / 2, m.to_center[1] - h / 2, w, h)
    pipeline.recapture(run, cv2.imencode(".png", demo.render(layout, seed=11))[1].tobytes())
    assert run.verification["verdict"] == "verified_open" and run.status == "ready"
    assert [h["kind"] for h in run.history] == ["plan", "recovery_plan"]


def test_only_one_recovery_per_run(demo_dir):
    run = start()
    pipeline.confirm(run)
    pipeline.recapture(run, img("wrong"))
    turns = pipeline.trace_view(run)["totals"]["agent_model_turns"]
    pipeline.recapture(run, img("blocked"))  # the person did not follow the corrective plan either
    assert run.verification["verdict"] == "route_still_blocked" and run.recoveries_used == 1
    assert run.status == "needs_remeasurement"
    assert pipeline.trace_view(run)["totals"]["agent_model_turns"] == turns


def _scene_ctx(board_w=300.0, board_h=200.0, margin_cm=1.0):
    """A sofa blocking a door-to-desk route; anchors are plain points, so no endpoint object is locked."""
    fps = [rect_footprint("sofa", "sofa", 130, 30, 40, 140, margin_cm=margin_cm)]
    return T.ToolContext(fps, board_w, board_h, {"door": (10.0, 100.0), "desk": (290.0, 100.0)}, agent_caps())


def _search(ctx, clearance=90):
    T.execute(ctx, "check_route", {"clearance_cm": clearance, "start": "door", "end": "desk"})
    return T.execute(ctx, "search_moves", {"max_moves": 1, "locked_ids": []})


@pytest.mark.parametrize("other", [{"board_h": 220.0}, {"board_w": 320.0}, {"margin_cm": 6.0}],
                         ids=["board_height", "board_width", "footprint_margin"])
def test_plan_cache_is_not_shared_across_board_size_or_margin(monkeypatch, other):
    monkeypatch.setattr(T, "_PLAN_CACHE", {})
    a, b = _scene_ctx(), _scene_ctx(**other)
    ra, rb = _search(a), _search(b)
    assert ra["plan_id"] != rb["plan_id"]
    assert a.plans[ra["plan_id"]] is not b.plans[rb["plan_id"]] and len(T._PLAN_CACHE) == 2
    # the same inputs still hit the cache (same plan object, same id)
    c = _scene_ctx(**other)
    rc = _search(c)
    assert rc["plan_id"] == rb["plan_id"] and c.plans[rc["plan_id"]] is b.plans[rb["plan_id"]]
    assert len(T._PLAN_CACHE) == 2
    if "margin_cm" in other:  # the plan really was computed under each context's own margin
        assert a.plans[ra["plan_id"]].route_before["margin_cm"] != b.plans[rb["plan_id"]].route_before["margin_cm"]
