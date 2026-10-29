"""Offline stand-in for the Meta Model API used by tests (wire-format and fault-injection checks).

Perception answers come from the repo's own colour-segmentation mock; planner answers come from the scripted
mock policy, translated to/from the OpenAI-style wire shape the Meta adapter assumes. It exercises plumbing
only and says nothing about Muse Spark or SAM quality. Everything the server receives is kept in `requests`.
"""
from __future__ import annotations

import base64
import hashlib
import json

import cv2
import httpx
import numpy as np

from app.providers.meta import MetaConstraint, MetaPerception, MetaPlanner
from app.providers.mock import MockPerception
from app.providers.mock_policy import MockPlanner


def _neutral(wire_msgs):
    out = []
    for m in wire_msgs:
        if m["role"] == "assistant":
            out.append({"role": "assistant", "content": m.get("content"), "tool_calls": [
                {"id": c["id"], "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
                for c in m.get("tool_calls", [])]})
        else:
            out.append(m)
    return out


def wire_tool_call(name, args, cid="call_x", raw=None):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args)}}


class FakeMeta:
    def __init__(self, hook=None, planner_script=None, sam=None, tokens=True):
        self.hook, self.planner_script, self.sam, self.tokens = hook, planner_script, sam or {}, tokens
        self.requests: list[tuple[str, dict]] = []
        self.counts: dict[str, int] = {}
        self._perc, self._cache = MockPerception(), {}

    def transport(self):
        return httpx.MockTransport(self.handle)

    def providers(self):
        from app.providers import Providers
        t = self.transport()
        return Providers(MetaPerception("k", t), MetaConstraint("k", t), MetaPlanner("k", t))

    def _perceive(self, b64):
        k = hashlib.sha256(b64.encode()).hexdigest()
        if k not in self._cache:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            self._cache[k] = self._perc.perceive(img)
        return self._cache[k]

    def handle(self, request):
        path, body = request.url.path, json.loads(request.content)
        self.requests.append((path, body))
        n = self.counts[path] = self.counts.get(path, 0) + 1
        if self.hook:
            r = self.hook(path, body, n)
            if isinstance(r, Exception):
                raise r
            if r is not None:
                return r
        if path.endswith("/images/segmentations"):
            return httpx.Response(200, json=self._segment(body))
        msgs = body["messages"]
        if "tools" in body:
            return httpx.Response(200, json=self._planner(body, msgs, n))
        content = msgs[0]["content"]
        img = next(p for p in content if p["type"] == "image_url")["image_url"]["url"].split(",", 1)[1]
        res = self._perceive(img)
        return httpx.Response(200, json=self._chat(json.dumps({
            "labels": sorted({o.label for o in res.objects}), "board_corners_px": res.board_corners_px,
            "check_points_px": res.check_points_px}), None, body))

    def _chat(self, content, tool_calls, body):
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        out = {"choices": [{"message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}]}
        if self.tokens:
            n = len(json.dumps(body)) // 4
            out["usage"] = {"prompt_tokens": n, "completion_tokens": 20, "total_tokens": n + 20}
        return out

    def _planner(self, body, msgs, n):
        if self.planner_script is not None:
            wire = self.planner_script(n, _neutral(msgs))
            if wire is not None:
                return self._chat(wire.get("content"), wire.get("tool_calls"), body)
        turn = MockPlanner().next_turn(_neutral(msgs), [])
        calls = [wire_tool_call(c.name, c.arguments, c.id) for c in turn.tool_calls]
        return self._chat(turn.content, calls, body)

    def _segment(self, body):
        res = self._perceive(body["image"])
        label = body["prompt"]
        masks = [o.mask for o in res.objects if o.label == label]
        mode = self.sam.get(label)
        if mode == "none":
            masks = []
        elif isinstance(mode, str) and mode.startswith("merge:"):  # union with another label's masks
            masks = [np.logical_or.reduce([*masks, *[o.mask for o in res.objects if o.label == mode[6:]]])]
        out = []
        for m in masks:
            ok, buf = cv2.imencode(".png", (m.astype(np.uint8)) * 255)
            out.append({"mask_png_b64": base64.b64encode(buf.tobytes()).decode(), "score": 0.9})
        return {"masks": out}
