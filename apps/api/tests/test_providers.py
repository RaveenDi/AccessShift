import json

import httpx
import pytest

from app.providers import get_providers, provider_mode
from app.providers.base import ProviderError
from app.providers.meta import MetaConstraint
from app.providers.mock import MockConstraint

LABELS = ["sofa", "shelf", "desk", "armchair"]


def test_mock_constraint_parser():
    s = MockConstraint().interpret("Keep a 90 cm wheelchair route from the door to the desk. Lock the sofa.", LABELS)
    assert (s.start_name, s.end_name, s.clearance_cm, s.locked_labels) == ("door", "desk", 90, ["sofa"])
    with pytest.raises(ProviderError):
        MockConstraint().interpret("make it nicer", LABELS)


def test_default_provider_is_mock_without_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.setattr("app.providers.load_dotenv", lambda: None)
    assert get_providers()[0].name == "mock"


@pytest.mark.parametrize("blank", ["", "   ", "	"])
def test_blank_model_provider_is_treated_as_unset(monkeypatch, blank):
    monkeypatch.setattr("app.providers.load_dotenv", lambda: None)
    monkeypatch.setenv("MODEL_PROVIDER", blank)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert provider_mode() == "mock" and get_providers()[0].name == "mock"
    monkeypatch.setenv("MODEL_API_KEY", "test-key-not-real")  # unset provider + a key still defaults to meta
    assert provider_mode() == "meta"
    monkeypatch.setenv("MODEL_PROVIDER", "bogus")  # a real typo is still rejected
    with pytest.raises(RuntimeError):
        provider_mode()


def test_meta_constraint_validates_model_output_offline():
    def handler(req):
        bad = req.headers["authorization"] == "Bearer bad"
        spec = {"start_name": "door", "end_name": "desk", "clearance_cm": 90,
                "locked_labels": ["ghost"] if bad else ["sofa"]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(spec)}}]})

    ok = MetaConstraint("k", httpx.MockTransport(handler)).interpret("x", LABELS)
    assert ok.locked_labels == ["sofa"]
    with pytest.raises(ProviderError):  # model may not invent labels
        MetaConstraint("bad", httpx.MockTransport(handler)).interpret("x", LABELS)
