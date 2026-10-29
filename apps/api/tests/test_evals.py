import json

import pytest

import harness as H
from app import demo
from app.domain.footprints import rect_footprint as R
from bruteforce import min_moves_up_to_one
from conftest import DATA
from fake_meta import FakeMeta

CANARY = "GT-CANARY-5521"


def raw_case(cid="case-secret-id-77", route_open=False, split="dev"):
    fp = [{"id": k, "label": k, "x": g[0] * 10, "y": g[1] * 10, "w": g[2] * 10, "h": g[3] * 10} for k, g in demo.GT.items()]
    return {"id": cid, "split": split, "photo": str(DATA / "blocked.png"), "board_mm": [5000, 3500], "check_len_mm": 2000,
            "anchors_mm": {k: [v[0] * 10, v[1] * 10] for k, v in demo.ANCHORS.items()},
            "constraint_text": demo.TEXT, "traps": ["blocked_route"],
            "ground_truth": {"footprints_mm": fp, "route_open": route_open, "expected_min_moves": 1, "notes": CANARY,
                             "constraint": {"clearance_mm": 900, "start": "door", "end": "desk",
                                            "locked_ids": ["sofa", "shelf", "desk"], "max_moves": 2},
                             "recapture": [{"photo": str(DATA / "correct.png"), "expected": "correct"},
                                           {"photo": str(DATA / "wrong.png"), "expected": "wrong"}]}}


def test_ground_truth_never_reaches_the_model(use_providers, demo_dir):
    fm = FakeMeta()
    use_providers(fm.providers())
    case = H.Case(raw_case(), DATA)
    H.install_guard()
    try:
        res = H.run_case(case, bruteforce=False)
    finally:
        H.uninstall_guard()
    assert fm.requests and res["outcome"] == "planned"
    for _, body in fm.requests:
        H.assert_clean_request(body, {CANARY, case.id})  # no forbidden key, no canary, no case id
    blob = json.dumps([b for _, b in fm.requests])
    assert CANARY not in blob and case.id not in blob and "expected_min_moves" not in blob
    # the same ground truth is used for scoring only
    assert res["plan"]["optimal"] and res["plan"]["valid_on_gt"] and res["route"]["agree"]
    assert [c["detected_wrong"] for c in res["recapture"]] == [False, True]


def test_guard_blocks_a_leaking_request(use_providers, demo_dir, monkeypatch):
    use_providers(FakeMeta().providers())
    case = H.Case(raw_case(), DATA)
    real = H.Case.model_inputs
    monkeypatch.setattr(H.Case, "model_inputs", lambda self: {**real(self), "text": demo.TEXT + " " + CANARY})
    H.install_guard()
    try:
        with pytest.raises(H.GroundTruthLeak):
            H.run_case(case, bruteforce=False)
    finally:
        H.uninstall_guard()
    for bad in ({"expected_min_moves": 1}, {"a": [{"ground_truth": {}}]}):
        with pytest.raises(H.GroundTruthLeak):
            H.assert_clean_request(bad)


def test_edge_error_and_matching():
    gt = [R("a", "sofa", 0, 0, 100, 50), R("b", "desk", 200, 0, 60, 60)]
    pred = [R("p2", "desk", 200.3, 0, 60, 60), R("p1", "sofa", 0, 0, 100, 50)]
    pairs, missed, spurious = H.match_footprints(pred, gt)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)] and not missed and not spurious
    assert H.edge_error_mm(pred[0], gt[1]) == pytest.approx(3.0, abs=0.01)
    assert H.edge_error_mm(pred[1], gt[0]) == 0
    _, missed, spurious = H.match_footprints(pred[:1], gt)
    assert missed == [0]


def test_heldout_can_be_reported_only_once(tmp_path):
    assert H.heldout_gate(tmp_path, "a" * 64, "mock", False) is None  # mock is plumbing, exempt
    m = H.heldout_gate(tmp_path, "a" * 64, "replay", False)
    m.write_text("done")
    with pytest.raises(SystemExit) as e:
        H.heldout_gate(tmp_path, "a" * 64, "replay", False)
    assert "once" in str(e.value)
    assert H.heldout_gate(tmp_path, "a" * 64, "replay", True) == m
    assert H.heldout_gate(tmp_path, "b" * 64, "replay", False)  # another manifest is a different experiment


def test_meta_provider_requires_a_call_cap_and_missing_manifest_is_reported(tmp_path, capsys):
    assert H.main(["--provider", "meta", "--split", "dev"]) == 2
    assert H.main(["--provider", "mock", "--split", "dev", "--manifest", str(tmp_path / "none.jsonl")]) == 2


def test_validation_catches_inconsistent_ground_truth(demo_dir):
    assert H.validate_case(H.Case(raw_case(), DATA), bruteforce=False) == []
    bad = H.validate_case(H.Case(raw_case(route_open=True), DATA), bruteforce=False)
    assert bad and "route_open" in bad[0]


def test_bruteforce_min_moves():
    def layout(arm):
        return [R("sofa", "sofa", 160, 20, 180, 90, True), R("shelf", "shelf", 160, 240, 180, 50, True),
                R("arm", "arm", arm[0], arm[1], 70, 70)]
    s, e = (60, 175), (440, 180)
    assert min_moves_up_to_one(layout((230, 145)), 500, 350, s, e, 90) == 1
    assert min_moves_up_to_one(layout((40, 40)), 500, 350, s, e, 90) == 0
    assert min_moves_up_to_one([f.locked_as(True) for f in layout((230, 145))], 500, 350, s, e, 90) is None


def test_mock_eval_end_to_end_writes_reports(tmp_path, monkeypatch, demo_dir):
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path / "rec"))
    mf = tmp_path / "m.jsonl"
    mf.write_text(json.dumps(raw_case("c1")) + "\n" + json.dumps(raw_case("c2", split="heldout")) + "\n")
    out = tmp_path / "reports"
    assert H.main(["--provider", "mock", "--manifest", str(mf), "--split", "dev", "--no-bruteforce",
                   "--out-dir", str(out)]) == 0
    js = next(out.glob("*-dev.json"))
    assert next(out.glob("*-dev.md")).exists()
    rep = json.loads(js.read_text())
    a = rep["aggregate"]
    assert rep["meta"]["provider"] == "mock" and "PLUMBING ONLY" in rep["meta"]["evidence_note"]
    assert rep["meta"]["commit"] and len(rep["cases"]) == 1  # heldout case was not run
    assert a["route_agreement"]["agree"]["rate"] == 1.0 and a["route_agreement"]["false_open"] == 0
    assert a["wrong_move_detection"]["wrong_detected"]["rate"] == 1.0
    assert a["wrong_move_detection"]["correct_accepted"]["rate"] == 1.0
    assert a["move_count_optimality"]["vs_expected"]["rate"] == 1.0
    assert a["telemetry"]["agent_tool_calls"] >= 5 and a["footprint_edge_error_mm"]["n_objects"] == 6
