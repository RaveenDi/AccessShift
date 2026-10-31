"""Spark + SAM perception (2026-10-31 fix): normalized 0-1000 coordinates -> pixels, named object boxes, one SAM call
with box prompts (the objects, then the board and the check ruler, whose masks refine the board corners and the ruler
end points). Offline only: the recorded Spark reply of the first live eval pins the coordinate scale, a fake transport
pins the SAM request text and the id -> name mapping. SAM box-prompt REPLIES are UNVERIFIED (see sam.py)."""
import json
import re
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fake_meta import fake_mask_decoder, sam_text

from app.providers import sam as samlib
from app.providers.base import ProviderError
from app.providers.meta import (MAX_OBJECTS, PERCEPTION_PROMPT, PERCEPTION_SCHEMA, MetaPerception, _content,
                                parse_perception)
from app.telemetry import CALL_SINK

LIVE = Path(__file__).parent / "fixtures" / "live"


def recorded():
    return json.loads((LIVE / "spark-perception-text-tags.json").read_text(encoding="utf-8"))


def meta(corners=None, check=None, objects=None):
    return {"board_corners_norm1000": corners or [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
            "check_points_norm1000": check or [[100, 100], [300, 100]],
            "objects": objects if objects is not None else [{"name": "SOFA", "box_norm1000": [100, 100, 300, 200]}]}


# ---- normalized 0-1000 -> pixels, on the recorded Spark reply ----------------------------------------------------
def test_recorded_spark_coordinates_are_normalized_not_pixels_and_convert_to_pixels():
    rec = recorded()
    w, h = rec["image_size"]
    old = json.loads(_content(rec["response"]))
    assert max(v for p in old["board_corners_px"] for v in p) < 1000 < w  # on the 0-1000 scale, not on 2336x1744 pixels
    corners, check, objs = parse_perception(
        meta(old["board_corners_px"], old["check_points_px"], [{"name": "SOFA", "box_norm1000": [100, 40, 400, 140]}]),
        (w, h))
    assert corners == [[68, 65], [2245, 56], [2273, 1678], [63, 1676]]  # round(v / 1000 * extent), per axis
    assert check == [[1332, 194], [2060, 190]]
    (tl, br), (mtl, mbr) = (corners[0], corners[2]), rec["measured_tl_br_px_approx"]
    assert abs(tl[0] - mtl[0]) < 0.03 * w and abs(tl[1] - mtl[1]) < 0.03 * h  # within a few % of the measured corners ...
    assert abs(br[0] - mbr[0]) < 0.03 * w and abs(br[1] - mbr[1]) < 0.03 * h
    assert abs(old["board_corners_px"][2][0] - mbr[0]) > 0.5 * w  # ... whereas read as pixels the far corner was ~1300 px off
    assert objs == [("sofa", (234, 70, 934, 244))]


def test_conversion_clamps_rounds_and_orders_boxes():
    w, h = 1200, 900
    corners, check, objs = parse_perception(meta(
        corners=[[-10, -5], [1010, 0], [1000, 1000], [0.4, 999.9]], check=[[500.5, 250], [1000, 1000]],
        objects=[{"name": "  Big   Sofa ", "box_norm1000": [800, 900, 200, 100]}]), (w, h))  # x1>x2, y1>y2: reordered
    assert corners == [[0, 0], [1199, 0], [1199, 899], [0, 899]]  # clamped to 0..1000, then inside the image
    assert check == [[601, 225], [1199, 899]]  # points stay inside the image (<= w-1, h-1)
    assert objs == [("big sofa", (240, 90, 960, 810))]  # name stripped, lowercased, spaces collapsed; half-open box


@pytest.mark.parametrize("bad", [
    meta(corners=[[0, 0]] * 3),                                           # 3 corners
    meta(check=[[0, 0]] * 3),                                             # 3 check points
    meta(objects=[]),                                                     # no objects
    meta(objects=[{"name": "", "box_norm1000": [1, 2, 3, 4]}]),           # blank name
    meta(objects=[{"name": "a", "box_norm1000": [1, 2, 3]}]),             # box with 3 numbers
    meta(objects=[{"name": "a", "box_norm1000": [100, 100, 100, 200]}]),  # zero-width box
    meta(objects=[{"name": "a", "box_norm1000": [100, 100, 100.2, 200]}]),  # rounds to zero width
    meta(objects=[{"name": "a", "box_norm1000": [29, 37, 2292, 1664]}]),  # pixel-scale values, not normalized
    meta(corners=[[0, 0], [1, 1], [2, 2], [1500, 3]]),                    # a corner far outside the scale
    meta(corners=[[0, 0], [1, 1], [2, 2], [True, 3]]),                    # bool is not a number
    meta(corners=[[0, 0], [1, 1], [2, 2], [float("nan"), 3]]),            # nan is not a coordinate
    meta(objects=[{"name": "a", "box_norm1000": [1, 2, 30, 40]}] * (MAX_OBJECTS + 1)),
    {"labels": ["SOFA"], "board_corners_px": [], "check_points_px": []},  # the old schema
])
def test_invalid_perception_is_rejected_not_guessed(bad):
    with pytest.raises((ValueError, KeyError, TypeError)):
        parse_perception(bad, (1200, 900))


# ---- the Spark request ------------------------------------------------------------------------------------------
def test_perception_schema_is_strict_and_the_prompt_states_the_scale_and_the_markers():
    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)
    nodes = list(objects(PERCEPTION_SCHEMA))
    assert len(nodes) == 2  # the root and the object entries
    for n in nodes:
        assert n["additionalProperties"] is False and sorted(n["required"]) == sorted(n["properties"])  # strict mode
    assert set(PERCEPTION_SCHEMA["properties"]) == {"board_corners_norm1000", "check_points_norm1000", "objects"}
    entry = PERCEPTION_SCHEMA["properties"]["objects"]["items"]
    assert set(entry["properties"]) == {"name", "box_norm1000"}  # the unit is in the key names
    assert "NORMALIZED to the range 0 to 1000" in PERCEPTION_PROMPT and "Do not use pixels" in PERCEPTION_PROMPT
    assert "printed on its tag" in PERCEPTION_PROMPT and "Markers are not objects" in PERCEPTION_PROMPT
    assert "door marker" in PERCEPTION_PROMPT and "CHECK" in PERCEPTION_PROMPT


# ---- fake transport: Spark + SAM ----------------------------------------------------------------------------------
W, H = 1200, 900


class Wire:
    """Fake Meta transport. Spark answers `spark` (a dict, as the model's JSON); SAM answers boxes by id with the mask
    in `masks[id]` or else a rectangle filling the box, for `answer_ids` (default: all prompted ids)."""

    def __init__(self, spark, answer_ids=None, masks=None):
        self.spark, self.answer_ids, self.masks, self.chat, self.sam = spark, answer_ids, masks or {}, [], []

    def transport(self):
        return httpx.MockTransport(self.handle)

    def handle(self, request):
        body = json.loads(request.content)
        if request.url.path.endswith("/chat/completions"):
            self.chat.append(body)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(self.spark)},
                                                          "finish_reason": "stop"}]})
        text = body["input"][0]["content"][0]["text"]
        self.sam.append((body, text))
        ids, masks = [], []
        for m in _boxes(text):
            oid, x1, y1, x2, y2 = (int(g) for g in m.groups())
            if self.answer_ids is None or oid in self.answer_ids:
                if oid in self.masks:
                    mask = self.masks[oid]
                else:
                    mask = np.zeros((H, W), bool)
                    mask[y1:y2 + 1, x1:x2 + 1] = True
                ids.append(oid)
                masks.append(mask)
        return httpx.Response(200, json={"status": "completed", "error": None, "output": [{
            "type": "message", "content": [{"type": "output_text", "text": sam_text(masks, ids)}]}],
            "usage": {"input_tokens": 0, "images_processed": 1}})


def _boxes(text):
    return re.finditer(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=\d+;h=\d+\|>", text)


def perception(wire):
    p = MetaPerception("k", wire.transport(), mask_decoder=fake_mask_decoder)
    p.c.sleep = lambda s: None
    return p


def entry(name, box):
    return {"name": name, "box_norm1000": box}


def test_perceive_sends_one_box_prompt_call_and_maps_ids_back_to_tag_names():
    spark = meta(objects=[entry("SOFA", [100, 100, 400, 200]), entry("Desk", [600, 300, 700, 800]),
                          entry("PLANT", [10, 800, 90, 900])])
    wire = Wire(spark, answer_ids={0, 2, 3, 4})  # SAM has no mask for the desk (id 1)
    calls = []
    tok = CALL_SINK.set(calls)
    try:
        res = perception(wire).perceive(np.zeros((H, W, 3), np.uint8))
    finally:
        CALL_SINK.reset(tok)
    # Spark was asked once, in the strict schema, with the scale stated in the prompt
    assert len(wire.chat) == 1
    rf = wire.chat[0]["response_format"]["json_schema"]
    assert rf["strict"] is True and rf["schema"] == PERCEPTION_SCHEMA
    assert "0 to 1000" in wire.chat[0]["messages"][0]["content"][0]["text"]
    # SAM: ONE request, text = Meta's box-prompt grammar over the converted pixel boxes (right/bottom half-open)
    assert len(wire.sam) == 1
    body, text = wire.sam[0]
    assert text == ("<0f>0<|box;x1=120;y1=90;x2=479;y2=179;w=1200;h=900|>,"
                    "1<|box;x1=720;y1=270;x2=839;y2=719;w=1200;h=900|>,"
                    "2<|box;x1=12;y1=720;x2=107;y2=809;w=1200;h=900|>,"
                    "3<|box;x1=0;y1=0;x2=1199;y2=899;w=1200;h=900|>,"     # the board, after the objects
                    "4<|box;x1=84;y1=63;x2=396;y2=117;w=1200;h=900|>")   # the check ruler: its points + 3% of the image
    assert body["model"] == "sam-3.1" and body["metadata"]["mask_encoding"] == "one_bit"
    assert [p["type"] for p in body["input"][0]["content"]] == ["input_text", "input_image"]
    # names come back from the ids; the desk has no mask -> the existing per-object issue, no invented mask
    assert [o.label for o in res.objects] == ["sofa", "plant"]
    assert res.objects[0].mask[90, 120] and res.objects[0].mask[179, 479] and res.objects[0].mask.sum() == 360 * 90
    assert [(i["kind"], i["label"], i["blocking"]) for i in res.issues] == [("no_mask", "desk", True)]
    assert "desk" in res.issues[0]["detail"]
    # pixels, not the model's 0-1000 numbers. The board mask is a rectangle on the corners, so they are refined to
    # themselves; the ruler "mask" is its padded box, 1.3x the seed's length, so the seed is kept (and says so).
    assert res.board_corners_px == [[0, 0], [1199, 0], [1199, 899], [0, 899]] and res.check_points_px == [[120, 90], [360, 90]]
    assert res.sources == {"board_corners": "refined", "check_points": "seed"}
    assert res.notes[1].startswith("board corners: refined (") and res.notes[2].startswith("check ruler end points: seed (")
    # telemetry: 1 Spark call + 1 SAM call, SAM costed per image
    assert [c.purpose for c in calls] == ["perceiver_labels", "sam_segment"]
    assert calls[1].cost_usd_est == 0.0025 and calls[1].input_tokens is None


def test_more_than_16_objects_use_one_sam_call_per_16_with_ids_restarting_at_zero():
    objs = [entry(f"O{i}", [10 + 40 * (i % 20), 10 + 40 * (i // 20), 40 + 40 * (i % 20), 40 + 40 * (i // 20)])
            for i in range(20)]
    wire = Wire(meta(objects=objs))
    res = perception(wire).perceive(np.zeros((H, W, 3), np.uint8))
    assert len(wire.sam) == 2  # 20 objects + the board + the ruler = 22 prompts
    ids = [[int(m.group(1)) for m in _boxes(t)] for _, t in wire.sam]
    assert ids == [list(range(16)), list(range(6))]  # the board and the ruler are ids 4 and 5 of the second call
    assert [o.label for o in res.objects] == [f"o{i}" for i in range(20)] and not res.issues  # offset applied to chunk 2
    x16 = round((10 + 40 * 16) / 1000 * W)  # object 16 is id 0 of the second call
    assert res.objects[16].mask[9, x16] and not res.objects[0].mask[9, x16]


def test_sam_empty_text_means_no_mask_for_every_object_and_two_calls_are_not_made():
    wire = Wire(meta(), answer_ids=set())
    res = perception(wire).perceive(np.zeros((H, W, 3), np.uint8))
    assert len(wire.sam) == 1 and res.objects == []
    assert [(i["kind"], i["label"]) for i in res.issues] == [("no_mask", "sofa")]
    assert res.sources == {"board_corners": "seed", "check_points": "seed"}  # nothing came back: Spark's values, unchanged
    assert "SAM returned no mask" in res.notes[1] and "SAM returned no mask" in res.notes[2]


def test_perceive_repairs_a_normalized_reply_that_looks_like_pixels_once():
    replies = [meta(objects=[entry("SOFA", [29, 37, 2292, 1664])]), meta()]

    class Seq(Wire):
        def handle(self, request):
            if request.url.path.endswith("/chat/completions"):
                self.spark = replies[len(self.chat)]
            return super().handle(request)
    wire = Seq(None)
    res = perception(wire).perceive(np.zeros((H, W, 3), np.uint8))
    assert len(wire.chat) == 2 and wire.chat[1]["messages"][-1]["role"] == "user"
    assert "normalized 0-1000, not pixels" in wire.chat[1]["messages"][-1]["content"]
    assert [o.label for o in res.objects] == ["sofa"]
    wire = Wire({"labels": ["SOFA"]})  # never valid
    with pytest.raises(ProviderError) as e:
        perception(wire).perceive(np.zeros((H, W, 3), np.uint8))
    assert e.value.kind == "malformed_response" and "unparseable after one repair attempt" in str(e.value) and not wire.sam


def test_box_prompt_is_meta_builders_text_and_respects_the_16_object_limit():
    from meta_sam_parser import FramePrompt, PromptBox, PromptObject, build_object_prompt
    boxes = [(0, 0, 10, 10), (5, 6, 1200, 900)]
    expect = build_object_prompt(size=(W, H), objects=[
        PromptObject(id=i, prompts=[FramePrompt(frame=0, box=PromptBox(*b))]) for i, b in enumerate(boxes)])
    assert samlib.box_prompt((W, H), boxes) == expect == (
        "<0f>0<|box;x1=0;y1=0;x2=9;y2=9;w=1200;h=900|>,1<|box;x1=5;y1=6;x2=1199;y2=899;w=1200;h=900|>")
    assert samlib.MAX_OBJECTS_PER_CALL == 16
    for bad in ([], [(0, 0, 1, 1)] * 17, [(0, 0, 1300, 10)], [(5, 5, 5, 9)]):  # none, too many, outside, empty
        with pytest.raises(ProviderError):
            samlib.box_prompt((W, H), bad)


# ---- board corners and check ruler refined from their masks -------------------------------------------------------
BOARD_PX = np.array([[60, 50], [1150, 40], [1170, 860], [40, 850]], float)  # a keystoned board in the 1200x900 image
RULER_PX = np.array([[420.0, 130.0], [900.0, 130.0]])                       # a 480 px ruler with end ticks


def _norm1000(px):
    return [[round(x / W * 1000, 1), round(y / H * 1000, 1)] for x, y in px]


def _quad_mask(q):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    return m.astype(bool)


def _ruler_mask(a, b):
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (int(a[0]), int(a[1]) - 3), (int(b[0]), int(b[1]) + 3), 1, -1)  # the line
    for p in (a, b):
        cv2.rectangle(m, (int(p[0]) - 3, int(p[1]) - 14), (int(p[0]) + 3, int(p[1]) + 14), 1, -1)  # the end ticks
    return m.astype(bool)


def coarse_spark(corner_offset):
    """Spark's reading of the same scene: the corners and the ruler ends a little off (about 1-2% of the image)."""
    return meta(corners=_norm1000(BOARD_PX + corner_offset), check=_norm1000(RULER_PX + [[-14, 12], [22, -9]]),
                objects=[entry("SOFA", [100, 100, 300, 200])])


def test_board_corners_and_ruler_ends_are_read_from_the_masks_of_two_extra_box_prompts():
    wire = Wire(coarse_spark(np.array([[14, 11], [-12, 9], [15, -13], [-11, -9]], float)),
                masks={1: _quad_mask(BOARD_PX), 2: _ruler_mask(*RULER_PX)})
    res = perception(wire).perceive(np.zeros((H, W, 3), np.uint8))
    assert len(wire.sam) == 1 and [int(m.group(1)) for m in _boxes(wire.sam[0][1])] == [0, 1, 2]  # sofa, board, ruler
    assert res.sources == {"board_corners": "refined", "check_points": "refined"}
    assert np.abs(np.array(res.board_corners_px) - BOARD_PX).max() <= 1.5  # exact, where Spark's seed was ~12 px off
    assert np.abs(np.array(res.check_points_px) - RULER_PX).max() <= 1.0   # the tick centres
    seed = np.array(wire.spark["board_corners_norm1000"]) * [W, H] / 1000
    assert np.abs(seed - BOARD_PX).max() > 8
    assert "board corners: refined (approxPolyDP" in res.notes[1] and "end ticks" in res.notes[2]
    assert [o.label for o in res.objects] == ["sofa"]  # the board and the ruler are not objects
    assert not res.issues


@pytest.mark.parametrize("answer_ids, board_mask, ruler_mask, board_src, ruler_src, why", [
    ({0}, None, None, "seed", "seed", "SAM returned no mask"),                          # SAM answers the sofa only
    ({0, 1}, _quad_mask(BOARD_PX), None, "refined", "seed", "SAM returned no mask"),    # no ruler mask
    ({0, 1, 2}, _quad_mask(BOARD_PX + [160, 0]), _ruler_mask(*RULER_PX), "seed", "refined", "from the Spark seed"),
    ({0, 1, 2}, _quad_mask(BOARD_PX), _ruler_mask(RULER_PX[0], RULER_PX[0] + [240, 0]), "refined", "seed", "x the Spark seed's"),
])
def test_a_missing_or_implausible_mask_keeps_sparks_value_and_records_it(answer_ids, board_mask, ruler_mask, board_src,
                                                                      ruler_src, why):
    masks = {k: v for k, v in {1: board_mask, 2: ruler_mask}.items() if v is not None}
    wire = Wire(coarse_spark(np.zeros((4, 2))), answer_ids=answer_ids, masks=masks)
    res = perception(wire).perceive(np.zeros((H, W, 3), np.uint8))
    assert res.sources == {"board_corners": board_src, "check_points": ruler_src}
    spark = wire.spark
    if board_src == "seed":  # exactly what Spark said, converted to pixels
        assert res.board_corners_px == [[min(round(x / 1000 * W), W - 1), min(round(y / 1000 * H), H - 1)]
                                        for x, y in spark["board_corners_norm1000"]]
    if ruler_src == "seed":
        assert res.check_points_px == [[round(x / 1000 * W), round(y / 1000 * H)] for x, y in spark["check_points_norm1000"]]
    assert any(why in n for n in res.notes[1:]) and not res.issues  # the reason is in the notes; no object was lost
    assert [o.label for o in res.objects] == ["sofa"]
