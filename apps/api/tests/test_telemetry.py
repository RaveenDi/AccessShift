import json

from fastapi.testclient import TestClient

from app import demo, pipeline
from app.main import app
from app.telemetry import estimate_cost, totals
from conftest import img, start
from fake_meta import FakeMeta


def test_mock_run_has_step_telemetry_and_zero_cost(demo_dir):
    run = start()
    pipeline.confirm(run)
    t = pipeline.trace_view(run)
    model_turns = [s for s in t["steps"] if s["kind"] == "model_turn"]
    assert len(model_turns) == 5 and all(s["prompt_version"] == "pa-1" and s["model"] for s in model_turns)
    assert all({"duration_ms", "latency_ms", "input_tokens", "output_tokens", "cost_usd_est", "api_calls",
                "retries", "kind"} <= set(s) for s in t["steps"])
    tot = t["totals"]
    assert tot["agent_model_turns"] == 5 and tot["agent_tool_calls"] == 5 and tot["api_calls"] == 0
    assert tot["input_tokens"] is None and tot["cost_usd_est"] == 0.0 and tot["prices_verified"] is True
    assert t["agent"]["steps_used"] == 5 and t["agent"]["steps_cap"] == 8


def test_meta_wire_run_reports_tokens_latency_and_cost(use_providers, demo_dir):
    use_providers(FakeMeta(tokens=True).providers())
    run = start()
    t = pipeline.trace_view(run)
    turns = [s for s in t["steps"] if s["kind"] == "model_turn"]
    assert all(s["api_calls"] == 1 and s["input_tokens"] > 0 and s["output_tokens"] == 20 and s["latency_ms"] >= 0
               and s["cost_usd_est"] > 0 and s["prompt_version"] == "pa-1" for s in turns)
    perc = next(s for s in t["steps"] if s["agent"] == "Perceiver")
    assert perc["api_calls"] == 2 and perc["output_tokens"] == 20  # 1 Spark call + 1 SAM call (all boxes at once)
    tot = t["totals"]
    assert tot["api_calls"] == 2 + 5 and tot["input_tokens"] == sum(s["input_tokens"] or 0 for s in t["steps"])
    assert tot["cost_usd_est"] == round(sum(s["cost_usd_est"] or 0 for s in t["steps"]), 6) and tot["cost_usd_est"] > 0


def test_tokens_are_none_when_the_api_returns_no_usage(use_providers, demo_dir):
    use_providers(FakeMeta(tokens=False).providers())
    run = start()
    turns = [s for s in pipeline.trace_view(run)["steps"] if s["kind"] == "model_turn"]
    assert all(s["input_tokens"] is None and s["output_tokens"] is None and s["cost_usd_est"] is None for s in turns)


def test_cost_estimate_uses_price_table_and_env_override(monkeypatch):
    assert estimate_cost("muse-spark-1.3", 1_000_000, 1_000_000) == 5.5  # published 1.25 + 4.25 USD per M tokens
    assert estimate_cost("unknown-model", 10, 10) is None
    monkeypatch.setenv("PRICE_TABLE_JSON", json.dumps({"m": {"input_per_mtok": 2, "output_per_mtok": 4}}))
    assert estimate_cost("m", 500_000, 250_000) == 2.0


def test_trace_endpoint_exposes_totals(demo_dir):
    c = TestClient(app)
    r = c.post("/api/runs", files={"photo": ("a.png", img("blocked"), "image/png")}, data={
        "board_w_cm": demo.BOARD_W, "board_h_cm": demo.BOARD_H, "check_len_cm": demo.CHECK_LEN,
        "constraint_text": demo.TEXT, "anchors": json.dumps(demo.ANCHORS)}).json()
    t = c.get(f"/api/runs/{r['id']}/trace").json()
    assert t["totals"]["agent_model_turns"] == 5 and "wall_ms" in t["totals"] and t["agent"]["status"] == "finished"
    assert totals([])["steps"] == 0
