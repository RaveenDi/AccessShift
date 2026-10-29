"""Scripted deterministic stand-in for the planner model (mock mode and offline tests).

It drives the SAME loop, tools and message protocol as the Meta planner, reading only the conversation
(first user message + tool results), like a model would. It says nothing about Muse Spark quality.
"""
from __future__ import annotations

import json
import re

from .base import ModelTurn, ToolCall
from .mock import _LOCK, _NUM

_ANSWER_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|centimet\w*|mm|millimet\w*|m|metres?|meters?)?\b", re.I)


def _to_cm(v: float, unit: str | None) -> float:
    u = (unit or "cm").lower()
    return v / 10 if u.startswith("mm") or u.startswith("millimet") else (v if u.startswith("c") else v * 100)


def _width(text: str) -> tuple[float | None, str]:
    for sent in re.split(r"[.;\n]", text):
        m = _NUM.search(sent)
        if m and re.search(r"route|path|clear|wide|width|wheelchair", sent, re.I):
            return _to_cm(float(m.group(1)), m.group(2)), ""
    if re.search(r"wheelchair", text, re.I):
        return 90.0, "no width stated; assumed 90 cm for wheelchair"
    return None, ""


class _Convo:
    def __init__(self, messages: list[dict]):
        self.user = json.loads(next(m["content"] for m in messages if m["role"] == "user"))
        self.calls: list[list] = []  # [name, args, result|None]
        by_id: dict[str, list] = {}
        for m in messages:
            if m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    by_id[tc["id"]] = [tc["name"], tc["arguments"], None]
                    self.calls.append(by_id[tc["id"]])
            elif m["role"] == "tool" and m.get("tool_call_id") in by_id:
                by_id[m["tool_call_id"]][2] = json.loads(m["content"])
        self.n_calls = len(self.calls)

    def result(self, name: str):
        for n, _, r in reversed(self.calls):
            if n == name and r is not None and "error" not in r:
                return r
        return None

    def errored(self, name: str) -> bool:
        return any(n == name and r is not None and "error" in r for n, _, r in self.calls)

    @property
    def answers(self) -> list[str]:
        return [r["user_answer"] for n, _, r in self.calls if n == "ask_user" and r and "user_answer" in r]

    @property
    def asks(self) -> int:
        return sum(n == "ask_user" for n, _, _ in self.calls)

    def explained(self) -> set[str]:
        return {a.get("move_id") for n, a, r in self.calls if n == "explain_move" and r and "error" not in r}


class MockPlanner:
    name, model = "mock", "mock-scripted-policy"

    def next_turn(self, messages: list[dict], tools: list[dict]) -> ModelTurn:
        cv = _Convo(messages)
        phase = cv.user.get("phase", "plan")
        nxt = self._recovery(cv) if phase == "recovery" else self._plan(cv)
        if nxt is None:  # the policy has nothing sensible to do: plain text, the loop treats it as malformed
            return ModelTurn("I cannot determine the constraints.", [], self.model, "stop")
        name, args = nxt
        return ModelTurn(None, [ToolCall(f"call_{cv.n_calls + 1}", name, args)], self.model, "tool_calls")

    # -- recovery: the observed layout is re-planned with the confirmed locks
    def _recovery(self, cv: _Convo):
        u = cv.user
        if cv.result("get_layout") is None:
            return "get_layout", {}
        if cv.result("check_route") is None:
            return "check_route", {"clearance_cm": u["clearance_cm"], "start": u["start"], "end": u["end"]}
        return self._search_explain_finish(cv, u["max_moves"], u["locked_ids"])

    # -- planning: parse the constraint text the way MockConstraint does, but ask when it is ambiguous
    def _plan(self, cv: _Convo):
        lay = cv.result("get_layout")
        if lay is None:
            return "get_layout", {}
        text = cv.user["constraint_text"]
        anchors, objs = list(lay["anchors_cm"]), lay["objects"]
        answers = cv.answers
        can_ask = cv.asks < 2

        width, _ = _width(text)
        for a in answers:
            if width is None and (m := _ANSWER_NUM.search(a)):
                width = _to_cm(float(m.group(1)), m.group(2))
        if width is None:
            return ("ask_user", {"question": "What clear route width, in cm, should be kept?"}) if can_ask else None

        ends = None
        m = re.search(r"from\s+(?:the\s+)?([\w ]+?)\s+to\s+(?:the\s+)?([\w ]+?)(?:[.,;]|$|\s+(?:and|while|lock|keep))",
                      text, re.I)
        if m and m.group(1).strip().lower() in anchors and m.group(2).strip().lower() in anchors:
            ends = (m.group(1).strip().lower(), m.group(2).strip().lower())
        elif len(anchors) == 2:
            ends = (anchors[0], anchors[1])
        else:
            for a in answers:
                found = sorted((a.lower().find(n), n) for n in anchors if n in a.lower())
                if len(found) >= 2:
                    ends = (found[0][1], found[1][1])
        if ends is None:
            q = f"Which two marked points should the route connect? Anchors: {anchors}"
            return ("ask_user", {"question": q}) if can_ask else None

        labels = sorted({o["label"] for o in objs})
        lock_labels = []
        for sent in re.split(r"[.;\n]", text):
            if _LOCK.search(sent) and not re.search(r"\bfrom\b.*\bto\b", sent, re.I):
                lock_labels += [lb for lb in labels if re.search(rf"\b{re.escape(lb)}s?\b", sent, re.I)]
        lock_ids: list[str] = []
        for lb in sorted(set(lock_labels)):
            ids = [o["id"] for o in objs if o["label"] == lb]
            if len(ids) > 1:
                named = [i for a in answers for i in ids if re.search(rf"\b{re.escape(i)}\b", a, re.I)]
                if named:
                    ids = sorted(set(named))
                elif any(re.search(r"\b(both|all)\b", a, re.I) for a in answers):
                    pass
                else:
                    q = f"Which {lb} should stay where it is? Options: {ids}, or say both"
                    return ("ask_user", {"question": q}) if can_ask else None
            lock_ids += ids
        mm = re.search(r"(?:at most|max(?:imum)?|up to)\s+(\d)\s+moves?", text, re.I)
        max_moves = int(mm.group(1)) if mm else 2
        max_moves = min(max(max_moves, 1), 3)

        if cv.result("check_route") is None:
            if cv.errored("check_route"):
                return None
            return "check_route", {"clearance_cm": width, "start": ends[0], "end": ends[1]}
        return self._search_explain_finish(cv, max_moves, sorted(set(lock_ids)))

    def _search_explain_finish(self, cv: _Convo, max_moves: int, lock_ids: list[str]):
        plan = cv.result("search_moves")
        if plan is None:
            if cv.errored("search_moves"):
                return None
            return "search_moves", {"max_moves": max_moves, "locked_ids": lock_ids}
        todo = [m["id"] for m in plan["moves"] if m["id"] not in cv.explained()]
        if todo and not cv.errored("explain_move"):
            return "explain_move", {"move_id": todo[0]}
        ids = [m["id"] for m in plan["moves"]]
        summary = (f"Move {', '.join(ids)}." if ids else plan["message"]) + " (scripted mock policy)"
        return "finish", {"plan_id": plan["plan_id"], "summary": summary}
