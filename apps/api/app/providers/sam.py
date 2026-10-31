"""SAM 3.1 wire layer. VERIFIED live on 2026-10-30 (tests/fixtures/live/sam-*.json); mask decoding VERIFIED 2026-10-30.

Route   : POST /v1/responses (not a dedicated endpoint, not chat completions), model "sam-3.1".
Request : {"model": "sam-3.1", "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
           "input": [{"type": "message", "role": "user", "content": [
               {"type": "input_text", "text": "<prompt>"},
               {"type": "input_image", "image_url": "data:image/png;base64,..."}]}]}
Prompt  : VERIFIED live: one short noun phrase (matches nothing on a photo of tagged blocks: the first live eval sent
          the printed tags "SOFA", "DESK", ... and got empty text 27 times). Used from 2026-10-31 instead: ONE call with
          box prompts for all objects, text built by Meta's `meta_sam_parser.build_object_prompt` (documented on
          dev.meta.ai/docs/sam/client-libraries): "<0f>" then "ID<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H|>" per object,
          x2/y2 inclusive (the builder takes half-open right/bottom). At most 16 objects per call (API limit); object
          ids are 0..n-1 within a call and map back to the names Spark gave. From 2026-10-31 the board and the check
          ruler are two more prompts in the same call, after the objects (see meta.py, domain/refine.py).
          UNVERIFIED until the tower's next live run: how the live API answers a box-prompt request (that it returns
          one record per prompted id, in the record format below, and no record for an object it cannot segment).
          Only the request text is checked offline (against the library) and the reply parsing on the recorded
          noun-phrase replies; a record with an id we did not send is rejected as malformed.
Response: a Responses-API object; the result is TEXT in output[0].content[0].text (type "output_text"):
          one line per frame starting "<0f>", then records "ID<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H;c=C|>
          <|mask;x=0;y=0;c=C;data=H,W,PAYLOAD|>" separated by commas. Boxes are pixel corners, x2/y2 inclusive,
          w/h the frame size; the mask raster (H x W) covers exactly the box (H = y2-y1+1, W = x2-x1+1; x=0;y=0 are
          always literal zeros on the wire). No match -> output_text is the empty string (HTTP 200).
          usage has no tokens, only "images_processed" (billed $0.0025 per image).
Payload : first char is the encoding marker ("!" one_bit, "~" lossless), then a 5-char base85 length prefix, then an
          arithmetic-coded body, decoded by Meta's pure-Python `meta-sam-parser` (0.1.0, SAM License, zero deps; named
          on dev.meta.ai/docs/sam/client-libraries). Checked on both recorded live replies: one_bit and lossless decode
          to the same rasters, our record parser agrees with the library's own stream parser, and the solid synthetic
          rectangles fill 96-98% of their boxes. Without the package the provider raises kind="mask_decoder_missing".
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import ProviderError

SAM_MODEL = "sam-3.1"
SAM_METADATA = {"include_confidence": "true", "mask_encoding": "one_bit"}

_RECORD = re.compile(
    r"(?P<id>\d+)<\|box;(?P<box>[^|]*)\|><\|mask;(?P<mask>[^|]*?);data=(?P<h>\d+),(?P<w>\d+),(?P<payload>[^|<>,;]+)\|>")
MAX_RECORDS = 64  # a malformed or hostile reply must not allocate unbounded masks

MAX_OBJECTS_PER_CALL = 16  # SAM API limit on box-prompted objects per request
Box = tuple[int, int, int, int]  # left, top, right, bottom in image pixels; right/bottom are half-open

Decoder = Callable[[str, int, int], bytes]  # (payload incl. marker, width, height) -> H*W bytes of 0/1


@dataclass
class SamRecord:
    object_id: int
    x1: int
    y1: int
    x2: int  # inclusive
    y2: int  # inclusive
    frame_w: int
    frame_h: int
    mask_h: int
    mask_w: int
    payload: str
    confidence: float | None


def sam_request(image_png_b64: str, text: str) -> dict:
    """The verified /v1/responses request; `text` is the prompt (see box_prompt)."""
    return {"model": SAM_MODEL, "metadata": dict(SAM_METADATA), "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": text}, {"type": "input_image", "image_url": "data:image/png;base64," + image_png_b64}]}]}


def box_prompt(size: tuple[int, int], boxes: Sequence[Box]) -> str:
    """Object-prompt text for one frame: object i gets id i and the box `boxes[i]` (left, top, right, bottom; right and
    bottom half-open). Built by Meta's own builder so the grammar is never hand-written."""
    if not 1 <= len(boxes) <= MAX_OBJECTS_PER_CALL:
        raise ProviderError(f"sam takes 1-{MAX_OBJECTS_PER_CALL} box prompts per call, got {len(boxes)}")
    try:
        from meta_sam_parser import FramePrompt, ObjectPromptError, PromptBox, PromptObject, build_object_prompt
    except ImportError:
        raise ProviderError("building SAM box prompts needs the optional package: pip install meta-sam-parser==0.1.0",
                            kind="mask_decoder_missing") from None
    try:
        return build_object_prompt(size=(int(size[0]), int(size[1])), objects=[
            PromptObject(id=i, prompts=[FramePrompt(frame=0, box=PromptBox(*(int(v) for v in b)))])
            for i, b in enumerate(boxes)])
    except ObjectPromptError as e:
        raise ProviderError(f"sam box prompt rejected ({e.code})") from None


def _bad(msg: str) -> ProviderError:
    return ProviderError(f"sam response unparseable: {msg}", kind="malformed_response")


def response_text(data: dict) -> str:
    """The concatenated output_text of a completed Responses-API object ('' when SAM found nothing)."""
    try:
        if data.get("status") not in (None, "completed") or data.get("error"):
            raise _bad(f"status {data.get('status')!r}")
        parts = [c["text"] for item in data["output"] if item.get("type") == "message"
                 for c in item["content"] if c.get("type") == "output_text"]
    except (KeyError, TypeError, AttributeError):
        raise _bad("no output[].content[].text") from None
    if not parts:
        raise _bad("no output_text part")
    return "".join(p for p in parts if isinstance(p, str))


def _kv(chunk: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in chunk.split(";") if "=" in p)


def parse_records(text: str) -> list[SamRecord]:
    """Parse SAM output text into records. Strict: any non-empty text that yields no record, or any inconsistent
    record, is a malformed response (nothing is guessed)."""
    if not text.strip():
        return []
    out = []
    for m in _RECORD.finditer(text):
        box, mask = _kv(m["box"]), _kv(m["mask"])
        try:
            x1, y1, x2, y2, w, h = (int(box[k]) for k in ("x1", "y1", "x2", "y2", "w", "h"))
            conf = float(box["c"]) if "c" in box else (float(mask["c"]) if "c" in mask else None)
        except (KeyError, ValueError):
            raise _bad("box token fields") from None
        mh, mw = int(m["h"]), int(m["w"])
        if not (0 <= x1 <= x2 < w and 0 <= y1 <= y2 < h):
            raise _bad("box outside the frame")
        if (mh, mw) != (y2 - y1 + 1, x2 - x1 + 1):
            raise _bad("mask size does not match its box")
        out.append(SamRecord(int(m["id"]), x1, y1, x2, y2, w, h, mh, mw, m["payload"], conf))
        if len(out) > MAX_RECORDS:
            raise _bad("too many records")
    if not out:
        raise _bad("text has no box+mask records")
    return out


def library_decoder(payload: str, width: int, height: int) -> bytes:
    """Default decoder: Meta's `meta-sam-parser`. The payload keeps its marker ("!" one_bit, "~" lossless)."""
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    except ImportError:
        raise ProviderError("decoding SAM masks needs the optional package: pip install meta-sam-parser==0.1.0",
                            kind="mask_decoder_missing") from None
    try:
        encoding = {"!": "one_bit", "~": "lossless"}[payload[:1]]
        return bytes(decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=width, height=height)))
    except KeyError:
        raise _bad("unknown mask encoding marker") from None
    except Exception as e:  # InvalidSegmentationMaskError and anything else the library rejects
        raise _bad(f"mask payload rejected ({type(e).__name__})") from None


def to_full_frame_mask(rec: SamRecord, decoder: Decoder) -> np.ndarray:
    """Bool (frame_h, frame_w) mask: the decoded raster placed at the record's box (x1, y1)."""
    raw = decoder(rec.payload, rec.mask_w, rec.mask_h)
    if len(raw) != rec.mask_h * rec.mask_w:
        raise _bad("decoded raster has the wrong size")
    crop = np.frombuffer(raw, np.uint8).reshape(rec.mask_h, rec.mask_w) > 0
    full = np.zeros((rec.frame_h, rec.frame_w), bool)
    full[rec.y1:rec.y2 + 1, rec.x1:rec.x2 + 1] = crop
    return full
