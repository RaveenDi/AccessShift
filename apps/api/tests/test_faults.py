"""Fault injection: provider timeouts, 429/5xx with Retry-After, malformed JSON, SAM no/merged masks, caps."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import demo, pipeline
from app.main import app
from app.providers.base import ProviderError
from app.providers.http import BUDGET, CallBudgetExceeded, ModelClient, parse_retry_after
from app.telemetry import CALL_SINK
from conftest import img, start
from fake_meta import FakeMeta, wire_tool_call

OK = {"choices": [{"message": {"content": "{}"}}]}


def client(handler, **kw):
    c = ModelClient("k", httpx.MockTransport(handler), **kw)
    c.sleeps = []
    c.sleep = c.sleeps.append
    return c


def post(c):
    sink = []
    tok = CALL_SINK.set(sink)
    try:
        return c.post("/chat/completions", {"model": "muse-spark-1.3"}), sink
    finally:
        CALL_SINK.reset(tok)


def script(seq):
    it = iter(seq)

    def handler(req):
        x = next(it)
        if isinstance(x, Exception):
            raise x
        return x
    return handler


def test_timeout_is_retried_with_exponential_backoff_then_succeeds():
    c = client(script([httpx.ReadTimeout("t"), httpx.ConnectTimeout("t"), httpx.Response(200, json=OK)]))
    data, sink = post(c)
    assert data == OK and c.sleeps == [0.5, 1.0]
    assert sink[0].attempts == 3 and sink[0].status == "ok"


def test_gives_up_after_two_retries_with_a_clear_error():
    c = client(script([httpx.ReadTimeout("t")] * 3))
    with pytest.raises(ProviderError) as e:
        post(c)
    assert e.value.kind == "timeout" and "3 attempt" in str(e.value) and c.sleeps == [0.5, 1.0]


def test_429_honours_retry_after_and_5xx_is_retried():
    c = client(script([httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(503),
                       httpx.Response(200, json=OK)]))
    data, sink = post(c)
    assert data == OK and c.sleeps == [3.0, 1.0] and sink[0].attempts == 3


def test_retry_after_beyond_cap_fails_fast_without_waiting():
    c = client(script([httpx.Response(429, headers={"Retry-After": "600"})]))
    with pytest.raises(ProviderError) as e:
        post(c)
    assert e.value.kind == "rate_limited" and "exceeds" in str(e.value) and c.sleeps == []


def test_retry_after_http_date_and_garbage():
    assert parse_retry_after("7") == 7.0 and parse_retry_after("nonsense") is None and parse_retry_after(None) is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # a date in the past means retry now


def test_client_errors_and_non_json_bodies_are_not_retried():
    for resp, kind in [(httpx.Response(400, json={"e": 1}), "http_error"), (httpx.Response(401), "http_error"),
                       (httpx.Response(200, text="<html>oops</html>"), "malformed_response"),
                       (httpx.Response(200, json=[1, 2]), "malformed_response")]:
        c = client(script([resp, httpx.Response(200, json=OK)]))
        with pytest.raises(ProviderError) as e:
            post(c)
        assert e.value.kind == kind and c.sleeps == []


def test_call_budget_stops_live_calls():
    c = client(lambda r: httpx.Response(200, json=OK), live=True)
    BUDGET.set(2)
    try:
        c.post("/x", {"model": "m"})
        c.post("/x", {"model": "m"})
        with pytest.raises(CallBudgetExceeded):
            c.post("/x", {"model": "m"})
    finally:
        BUDGET.set(None)


def test_planner_outage_degrades_to_human_confirmation_not_a_crash(use_providers, demo_dir):
    fm = FakeMeta(hook=lambda p, b, n: httpx.Response(503) if "tools" in b else None)
    use_providers(fm.providers())
    run = start()
    assert run.status == "needs_confirmation" and run.plan is None
    assert run.agent.status == "unresolved" and "model_unavailable (server_error)" in run.agent.reason
    assert fm.sleeps == [0.5, 1.0]  # 3 attempts, 2 backoffs, then give up
    assert pipeline.trace_view(run)["totals"]["retries"] == 2
    run = pipeline.confirm(run, locked_ids=["sofa", "shelf", "desk"],
                           manual={"clearance_cm": 90, "start_name": "door", "end_name": "desk"})
    assert run.status == "awaiting_recapture" and run.plan_source == "manual"


def test_perception_outage_fails_the_run_clearly(use_providers, demo_dir):
    fm = FakeMeta(hook=lambda p, b, n: httpx.ReadTimeout("t") if p.endswith("responses") else None)
    use_providers(fm.providers())
    with pytest.raises(pipeline.PipelineError) as e:
        start()
    assert "timeout" in str(e.value)
    assert [r for r in pipeline.RUNS.values() if r.status == "failed"]


def bad_args(n_bad):
    return lambda n, msgs: ({"content": None, "tool_calls": [wire_tool_call("get_layout", None, f"b{n}", raw="{oops")]}
                            if n <= n_bad else None)


def test_malformed_tool_arguments_get_one_repair_then_succeed(use_providers, demo_dir):
    use_providers(FakeMeta(planner_script=bad_args(1)).providers())
    run = start()
    assert run.agent.status == "finished" and run.agent.repairs == 1
    assert pipeline.trace_view(run)["totals"]["tool_errors"] == 1


def test_malformed_twice_ends_in_a_clear_needs_confirmation_status(use_providers, demo_dir):
    use_providers(FakeMeta(planner_script=bad_args(2)).providers())
    run = start()
    assert run.agent.status == "unresolved" and run.agent.reason.startswith("malformed_output")
    assert run.status == "needs_confirmation" and "confirm the constraints manually" in run.error and run.plan is None


def test_prose_instead_of_a_tool_call_is_repaired_once(use_providers, demo_dir):
    use_providers(FakeMeta(planner_script=lambda n, m: {"content": "Sure, happy to help!"} if n == 1 else None).providers())
    run = start()
    assert run.agent.status == "finished" and run.agent.repairs == 1


def test_malformed_response_structure_is_treated_as_model_unavailable(use_providers, demo_dir):
    fm = FakeMeta(hook=lambda p, b, n: httpx.Response(200, json={"oops": 1}) if "tools" in b else None)
    use_providers(fm.providers())
    run = start()
    assert "model_unavailable (malformed_response)" in run.agent.reason and run.status == "needs_confirmation"


def test_perception_json_gets_one_repair_attempt(use_providers, demo_dir):
    def hook(path, body, n):
        if "tools" not in body and not path.endswith("responses") and n == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "here you go: labels..."}}]})
    fm = FakeMeta(hook=hook)
    use_providers(fm.providers())
    assert start().status == "needs_confirmation"
    label_calls = [b for p, b in fm.requests if p.endswith("completions") and "tools" not in b]
    assert len(label_calls) == 2 and label_calls[1]["messages"][-1]["role"] == "user"
    fm2 = FakeMeta(hook=lambda p, b, n: httpx.Response(200, json={"choices": [{"message": {"content": "no"}}]})
                   if "tools" not in b and not p.endswith("responses") else None)
    use_providers(fm2.providers())
    with pytest.raises(pipeline.PipelineError) as e:
        start()
    assert "unparseable after one repair attempt" in str(e.value)


def test_sam_no_mask_falls_back_to_user_confirmation_without_planning(use_providers, demo_dir):
    fm = FakeMeta(sam={"armchair": "none"})
    use_providers(fm.providers())
    run = start()
    assert run.status == "needs_confirmation" and run.plan is None and run.agent is None
    assert [i["kind"] for i in run.issues] == ["no_mask"] and "armchair" in run.issues[0]["detail"]
    assert not [1 for _, b in fm.requests if "tools" in b]  # no model calls were spent planning on bad geometry
    with pytest.raises(pipeline.PipelineError) as e:
        pipeline.confirm(run, accept_review_ids=["sofa"])
    assert e.value.code == 409 and "no mask" in str(e.value).lower()


def test_sam_merged_masks_are_flagged_and_need_explicit_acceptance(use_providers, demo_dir):
    fm = FakeMeta(sam={"armchair": "merge:plant"})
    use_providers(fm.providers())
    run = start()
    kinds = {i["kind"] for i in run.issues}
    assert "merged_masks" in kinds and run.agent is None and run.status == "needs_confirmation"
    assert next(f for f in run.footprints if f.id == "armchair").needs_review
    with pytest.raises(pipeline.PipelineError) as e:
        pipeline.confirm(run)
    assert e.value.code == 409 and "need review" in str(e.value)
    run = pipeline.confirm(run, accept_review_ids=["armchair", "plant"])  # now the agent runs, then planning
    assert run.agent is not None and run.agent.status == "finished" and run.plan_confirmed
    assert [1 for _, b in fm.requests if "tools" in b]


def test_replay_miss_is_a_loud_502_at_the_api(monkeypatch, tmp_path, demo_dir):
    monkeypatch.setattr("app.providers.load_dotenv", lambda: None)
    monkeypatch.setenv("MODEL_PROVIDER", "replay")
    monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path))
    r = TestClient(app).post("/api/runs", files={"photo": ("a.png", img("blocked"), "image/png")}, data={
        "board_w_cm": demo.BOARD_W, "board_h_cm": demo.BOARD_H, "check_len_cm": demo.CHECK_LEN,
        "constraint_text": demo.TEXT, "anchors": json.dumps(demo.ANCHORS)})
    assert r.status_code == 502 and "replay miss" in r.json()["error"]
