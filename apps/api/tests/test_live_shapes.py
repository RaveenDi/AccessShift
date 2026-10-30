"""Replay of SANITIZED recordings of the first real Meta Model API calls (2026-10-30, synthetic inputs only).

tests/fixtures/live/*.json are {request, response} pairs: images are replaced by their sha256, no auth header, request
id or account id was stored. They pin the *verified* wire shapes (chat, strict json_schema, vision, tool calls, SAM 3.1
on /v1/responses). Responses are served by a mock transport; nothing here touches the network. A stub decoder checks
mask placement; the real payloads are decoded with Meta's meta-sam-parser (skipped if it is not installed)."""
import importlib.util
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app import config
from app.providers import sam as samlib
from app.providers.base import ProviderError
from app.providers.http import ModelClient, _usage
from app.providers.meta import (CONSTRAINT_SCHEMA, MetaPerception, MetaPlanner, _chat, _content, _from_wire,
                                _strict_format, _to_wire)
from app.telemetry import CALL_SINK, estimate_cost

LIVE = Path(__file__).parent / "fixtures" / "live"


def fx(name):
    return json.loads((LIVE / f"{name}.json").read_text(encoding="utf-8"))


def serving(name, seen):
    rec = fx(name)

    def respond(request):
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=rec["response"])
    return httpx.MockTransport(respond)


def black(w=1200, h=900):
    return np.zeros((h, w, 3), np.uint8)


def stub_decoder(payload, width, height):
    return b"\x01" * (width * height)


# ---- chat completions -------------------------------------------------------------------------------------------
def test_chat_body_matches_the_recorded_request_keys_and_usage_shape(monkeypatch):
    monkeypatch.setenv("META_REASONING_EFFORT", "minimal")
    rec = fx("chat-basic")
    body = _chat("muse-spark-1.3", rec["request"]["messages"])
    assert body == rec["request"] | {"max_completion_tokens": config.max_completion_tokens()}
    assert "max_tokens" not in body and "temperature" not in body  # only verified parameters are sent
    assert _content(rec["response"]) == "pong"
    assert _usage(rec["response"]) == (15, 16)
    assert estimate_cost("muse-spark-1.3", 15, 16) == round((15 * 1.25 + 16 * 4.25) / 1e6, 10)


def test_truncated_or_refused_chat_replies_are_malformed_not_json():
    reply = json.loads(json.dumps(fx("chat-basic")["response"]))
    reply["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ProviderError) as e:
        _content(reply)
    assert e.value.kind == "malformed_response"


def test_strict_json_schema_wrapper_has_the_recorded_shape():
    sent = fx("chat-strict-json-schema-vision")["request"]["response_format"]
    ours = _strict_format("ConstraintSpec", CONSTRAINT_SCHEMA)
    assert sent["type"] == ours["type"] == "json_schema"
    assert set(sent["json_schema"]) == set(ours["json_schema"]) == {"name", "strict", "schema"}
    assert ours["json_schema"]["strict"] is True
    assert CONSTRAINT_SCHEMA["additionalProperties"] is False  # strict mode: HTTP 400 without it
    assert sorted(CONSTRAINT_SCHEMA["required"]) == sorted(CONSTRAINT_SCHEMA["properties"])  # every property required
    user = fx("chat-strict-json-schema-vision")["request"]["messages"][1]["content"]
    assert [p["type"] for p in user] == ["text", "image_url"]  # the vision part layout MetaPerception sends
    assert json.loads(_content(fx("chat-strict-json-schema-vision")["response"]))  # reply is JSON text in content


# ---- tool calling -----------------------------------------------------------------------------------------------
def _neutral(wire):
    out = []
    for m in wire:
        if m["role"] == "assistant":
            out.append({"role": "assistant", "content": m.get("content"), "tool_calls": [
                {"id": c["id"], "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
                for c in m.get("tool_calls", [])]})
        else:
            out.append(m)
    return out


def test_recorded_tool_call_replies_parse_including_null_content_and_parallel_calls():
    turn = _from_wire(fx("tools-turn-parallel-calls")["response"], "muse-spark-1.3")
    assert turn.finish_reason == "tool_calls" and len(turn.tool_calls) == 5
    assert all(c.error is None and isinstance(c.arguments, dict) and c.id and c.name for c in turn.tool_calls)
    null_turn = _from_wire(fx("tools-turn-null-content")["response"], "muse-spark-1.3")
    assert null_turn.content is None and len(null_turn.tool_calls) == 2 and null_turn.finish_reason == "tool_calls"


def test_tool_results_go_back_in_the_recorded_message_shape():
    recorded = fx("tools-turn-parallel-calls")["request"]["messages"]
    wire = _to_wire(_neutral(recorded))
    assert len(wire) == len(recorded)
    for ours, theirs in zip(wire, recorded):
        assert ours["role"] == theirs["role"]
        if theirs["role"] == "assistant":
            assert [(c["id"], c["type"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in ours["tool_calls"]] == \
                   [(c["id"], c["type"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in theirs["tool_calls"]]
        if theirs["role"] == "tool":
            assert ours == {"role": "tool", "tool_call_id": theirs["tool_call_id"], "content": theirs["content"]}


def test_planner_request_has_the_verified_keys_only():
    rec, seen = fx("tools-turn-parallel-calls"), []
    tools = [t["function"] for t in rec["request"]["tools"]]
    turn = MetaPlanner("k", serving("tools-turn-parallel-calls", seen)).next_turn(_neutral(rec["request"]["messages"]), tools)
    assert len(turn.tool_calls) == 5
    path, body = seen[0]
    assert path == "/v1/chat/completions"
    assert body["tool_choice"] == "auto" and "temperature" not in body  # any other tool_choice is HTTP 400
    assert body["tools"] == rec["request"]["tools"]
    assert set(body) <= set(rec["request"]) | {"temperature"} and set(rec["request"]) <= set(body) | {"reasoning_effort"}


# ---- SAM 3.1 on /v1/responses -----------------------------------------------------------------------------------
def test_sam_request_matches_the_recorded_request():
    rec = fx("sam-onebit")["request"]
    ours = samlib.sam_request("B64", "orange rectangle")
    img = rec["input"][0]["content"][1]
    assert img["image_url"].startswith("data:image/png;")  # recording keeps only the hash
    assert ours["model"] == rec["model"] == "sam-3.1" and ours["metadata"] == rec["metadata"]
    assert [p["type"] for p in ours["input"][0]["content"]] == [p["type"] for p in rec["input"][0]["content"]]
    assert ours["input"][0]["content"][0] == rec["input"][0]["content"][0]
    assert ours["input"][0]["content"][1]["image_url"] == "data:image/png;base64,B64"
    assert (ours["input"][0]["type"], ours["input"][0]["role"]) == ("message", "user")


@pytest.mark.parametrize("name,marker", [("sam-onebit", "!"), ("sam-lossless", "~")])
def test_recorded_sam_replies_parse_to_boxes_sizes_and_confidence(name, marker):
    recs = samlib.parse_records(samlib.response_text(fx(name)["response"]))
    assert len(recs) == 2 and all(r.payload[0] == marker for r in recs)
    a, b = recs
    assert (a.x1, a.y1, a.x2, a.y2, a.frame_w, a.frame_h) == (894, 181, 1077, 292, 1200, 900)
    assert (a.mask_h, a.mask_w) == (112, 184) and abs(a.confidence - 0.773) < 1e-3
    assert (b.x1, b.y1, b.x2, b.y2) == (576, 388, 710, 520) and (b.mask_h, b.mask_w) == (133, 135)


def test_sam_usage_has_no_tokens_and_is_costed_per_image():
    reply = fx("sam-onebit")["response"]
    assert reply["usage"]["images_processed"] == 1 and reply["usage"]["input_tokens"] == 0
    seen, calls = [], []
    tok = CALL_SINK.set(calls)
    try:
        MetaPerception("k", serving("sam-onebit", seen), mask_decoder=stub_decoder).segment(black(), "orange rectangle")
    finally:
        CALL_SINK.reset(tok)
    assert seen[0][0] == "/v1/responses" and seen[0][1]["model"] == "sam-3.1"
    assert len(calls) == 1 and calls[0].cost_usd_est == 0.0025 and calls[0].input_tokens is None


def test_segment_places_each_decoded_raster_at_its_box_in_the_full_frame():
    objs = MetaPerception("k", serving("sam-onebit", []), mask_decoder=stub_decoder).segment(black(), "orange rectangle")
    assert [o.label for o in objs] == ["orange rectangle"] * 2 and abs(objs[0].confidence - 0.773) < 1e-3
    m = objs[0].mask
    assert m.shape == (900, 1200) and m.dtype == bool and m.sum() == 112 * 184
    assert m[181, 894] and m[292, 1077] and not m[180, 894] and not m[181, 1078]  # x2/y2 are inclusive


def test_sam_no_match_is_an_empty_text_and_an_empty_list():
    assert samlib.response_text(fx("sam-empty")["response"]) == ""
    assert MetaPerception("k", serving("sam-empty", []), mask_decoder=stub_decoder).segment(black(), "giraffe") == []


def test_sam_frame_size_mismatch_is_malformed():
    with pytest.raises(ProviderError) as e:
        MetaPerception("k", serving("sam-onebit", []), mask_decoder=stub_decoder).segment(black(640, 480), "orange rectangle")
    assert e.value.kind == "malformed_response"


def test_sam_text_that_is_not_records_is_malformed_not_guessed():
    with pytest.raises(ProviderError):
        samlib.parse_records("<0f>garbage")
    bad = samlib.response_text(fx("sam-onebit")["response"]).replace("x2=1077", "x2=1200")  # box outside the frame
    with pytest.raises(ProviderError):
        samlib.parse_records(bad)


@pytest.mark.skipif(importlib.util.find_spec("meta_sam_parser") is not None, reason="decoder package installed")
def test_without_meta_sam_parser_decoding_fails_with_a_clear_kind():
    with pytest.raises(ProviderError) as e:
        MetaPerception("k", serving("sam-onebit", [])).segment(black(), "orange rectangle")
    assert e.value.kind == "mask_decoder_missing"


# ---- real mask payloads, decoded with Meta's meta-sam-parser ----------------------------------------------------
needs_parser = pytest.mark.skipif(importlib.util.find_spec("meta_sam_parser") is None, reason="meta-sam-parser missing")


def crops(name):
    recs = samlib.parse_records(samlib.response_text(fx(name)["response"]))
    return recs, [samlib.to_full_frame_mask(r, samlib.library_decoder)[r.y1:r.y2 + 1, r.x1:r.x2 + 1] for r in recs]


@needs_parser
def test_recorded_one_bit_and_lossless_masks_decode_to_the_same_filled_rectangles():
    _, one_bit = crops("sam-onebit")
    _, lossless = crops("sam-lossless")
    assert [c.shape for c in one_bit] == [(112, 184), (133, 135)]
    for a, b in zip(one_bit, lossless):
        assert a.mean() > 0.95  # the synthetic objects are solid rectangles
        assert (a == b).mean() > 0.999


@needs_parser
def test_our_record_parser_agrees_with_meta_stream_parser():
    import meta_sam_parser as msp
    text = samlib.response_text(fx("sam-onebit")["response"])
    parser = msp.image_segmentation_format().create_parser()
    parser.push(text)
    lib = [r for r in parser.finish(msp.CompletedOutcome()).result.records if isinstance(r, msp.SegmentationMaskRecord)]
    recs, ours = crops("sam-onebit")
    assert [int(r.object_id) for r in lib] == [r.object_id for r in recs]
    for r, mine, theirs in zip(recs, ours, lib):
        b = theirs.bounds
        assert (b.left, b.top, b.right - 1, b.bottom - 1) == (r.x1, r.y1, r.x2, r.y2)
        raster = np.frombuffer(msp.decode_mask_to_raster(theirs.mask), np.uint8).reshape(r.mask_h, r.mask_w) > 0
        assert (raster == mine).all()


@needs_parser
def test_a_decoded_sam_mask_becomes_a_metric_footprint_of_the_right_size():
    from app.domain.calibration import calibrate
    from app.domain.footprints import footprint_from_mask
    objs = MetaPerception("k", serving("sam-onebit", [])).segment(black(), "orange rectangle")
    cal = calibrate([(0, 0), (1199, 0), (1199, 899), (0, 899)], 119.9, 89.9, [(100, 850), (300, 850)], 20, (1200, 900))
    fp = footprint_from_mask(objs[0].mask, cal, "o1", "orange rectangle", confidence=objs[0].confidence)
    x0, y0, x1, y1 = fp.polygon.bounds  # 10 px per cm; the box is 184 x 112 px
    assert abs((x1 - x0) - 18.4) < 1.0 and abs((y1 - y0) - 11.2) < 1.0
    assert abs(x0 - 89.4) < 1.0 and abs(y0 - 18.1) < 1.0
