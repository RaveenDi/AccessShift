import json

import httpx
import pytest

from app.providers import get_providers, provider_mode
from app.providers.meta import MetaConstraint
from app.providers.recording import (RecordingTransport, ReplayMissError, ReplayTransport, normalize,
                                     request_key)

LABELS = ["sofa", "shelf", "desk"]
SECRET = "sk-test-SECRET-0123456789"
SPEC = {"start_name": "door", "end_name": "desk", "clearance_cm": 90, "locked_labels": ["sofa"]}


def _fake_meta(calls):
    def handler(req):
        calls.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(SPEC)}}]})
    return httpx.MockTransport(handler)


def test_key_is_stable_under_key_order_and_hashes_long_strings():
    a = request_key("POST", "/v1/x", {"a": 1, "b": {"c": [1, 2]}, "img": "A" * 5000})
    b = request_key("post", "/v1/x", {"img": "A" * 5000, "b": {"c": [1, 2]}, "a": 1})
    assert a == b
    assert a != request_key("POST", "/v1/x", {"a": 1, "b": {"c": [1, 3]}, "img": "A" * 5000})
    assert a != request_key("POST", "/v1/x", {"a": 1, "b": {"c": [1, 2]}, "img": "A" * 5001})
    assert "A" * 100 not in json.dumps(normalize({"img": "A" * 5000}))


def test_record_then_replay_roundtrip_without_key_or_network(tmp_path):
    calls = []
    rec = RecordingTransport(_fake_meta(calls), tmp_path)
    live = MetaConstraint(SECRET, rec).interpret("route text", LABELS)
    assert len(calls) == 1 and rec.saved == 1
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    # the API key never reaches disk
    assert SECRET not in files[0].read_text(encoding="utf-8")
    assert "authorization" not in files[0].read_text(encoding="utf-8").lower()

    replayed = MetaConstraint("replay-no-key", ReplayTransport(tmp_path)).interpret("route text", LABELS)
    assert replayed == live and len(calls) == 1  # no second network call


def test_replay_miss_fails_loudly_and_is_not_a_provider_error(tmp_path):
    from app.providers.base import ProviderError
    c = MetaConstraint("replay-no-key", ReplayTransport(tmp_path))
    with pytest.raises(ReplayMissError) as e:
        c.interpret("never recorded", LABELS)
    assert "replay miss" in str(e.value) and not isinstance(e.value, ProviderError)


def test_errors_are_not_recorded(tmp_path):
    rec = RecordingTransport(httpx.MockTransport(lambda r: httpx.Response(500, json={"e": 1})), tmp_path)
    with pytest.raises(Exception):
        MetaConstraint("k", rec).interpret("x", LABELS)
    assert list(tmp_path.glob("*.json")) == []


def test_get_providers_modes(monkeypatch, tmp_path):
    monkeypatch.setattr("app.providers.load_dotenv", lambda: None)
    monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_PROVIDER", "replay")
    p, c = get_providers()
    assert p.name == "meta" and isinstance(c.c.http._transport, ReplayTransport)
    monkeypatch.setenv("MODEL_PROVIDER", "meta")
    with pytest.raises(RuntimeError):
        get_providers()
    monkeypatch.setenv("MODEL_API_KEY", SECRET)
    _, c = get_providers()
    assert isinstance(c.c.http._transport, RecordingTransport)
    monkeypatch.setenv("RECORD_MODEL_CALLS", "0")
    _, c = get_providers()
    assert not isinstance(c.c.http._transport, RecordingTransport)
    monkeypatch.setenv("MODEL_PROVIDER", "bogus")
    with pytest.raises(RuntimeError):
        provider_mode()
