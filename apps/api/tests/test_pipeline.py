import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import demo
from app.main import app

DATA = Path(__file__).resolve().parents[1] / "demo_data"


@pytest.fixture(scope="module")
def client():
    demo.ensure_demo(DATA)
    return TestClient(app)


def _img(n):
    return (DATA / f"{n}.png").read_bytes()


def _start(c):
    r = c.post("/api/runs", files={"photo": ("a.png", _img("blocked"), "image/png")},
               data={"board_w_cm": demo.BOARD_W, "board_h_cm": demo.BOARD_H, "check_len_cm": demo.CHECK_LEN,
                     "constraint_text": demo.TEXT, "anchors": json.dumps(demo.ANCHORS)})
    assert r.status_code == 200, r.text
    return r.json()


def _planned(c):
    v = _start(c)
    assert v["route_initial"]["status"] == "blocked"
    assert {f["id"] for f in v["footprints"] if f["locked"]} == {"sofa", "shelf", "desk"}
    r = c.post(f"/api/runs/{v['id']}/confirm", json={})
    assert r.status_code == 200, r.text
    return r.json()


def _recap(c, rid, name):
    r = c.post(f"/api/runs/{rid}/recapture", files={"photo": ("b.png", _img(name), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def test_mock_pipeline_end_to_end_correct_move(client):
    v = _planned(client)
    p = v["plan"]
    assert p["status"] == "plan_found" and p["n_moves"] == 1 and p["moves"][0]["id"] == "armchair"
    assert p["route_after"]["status"] == "open"
    ver = _recap(client, v["id"], "correct")
    assert ver["verdict"] == "verified_open", ver
    tr = client.get(f"/api/runs/{v['id']}/trace").json()["steps"]
    assert [s["agent"] for s in tr] == ["Perceiver", "Calibrator"] + ["Planner Agent"] * 10 + [
        "Human", "Perceiver", "Calibrator", "Verifier"]
    assert [s["role"] for s in tr if s["agent"] == "Planner Agent" and s["kind"] == "tool_call"] == [
        "tool:get_layout", "tool:check_route", "tool:search_moves", "tool:explain_move", "tool:finish"]
    assert all(s["status"] == "ok" and "duration_ms" in s and s["model"] for s in tr)


def test_wrong_move_detected_route_still_blocked(client):
    v = _planned(client)
    ver = _recap(client, v["id"], "wrong")
    assert ver["verdict"] == "route_still_blocked" and not ver["pass"]
    kinds = {o["id"]: o["kind"] for o in ver["objects"]}
    assert kinds["armchair"] == "moved_to_wrong_place" and kinds["plant"] == "unexpected_move"


def test_wrong_move_route_open_but_deviates(client):
    v = _planned(client)
    ver = _recap(client, v["id"], "wrong_open")
    assert ver["verdict"] == "route_open_but_deviates_from_plan" and not ver["pass"]


def test_unchanged_recapture_flags_not_moved(client):
    v = _planned(client)
    ver = _recap(client, v["id"], "blocked")
    assert ver["verdict"] == "route_still_blocked"
    assert {o["id"]: o["kind"] for o in ver["objects"]}["armchair"] == "not_moved"
