import os

from dotenv import load_dotenv

from .mock import MockConstraint, MockPerception


def get_providers():
    """MODEL_PROVIDER=mock|meta; defaults to mock when MODEL_API_KEY is absent. Key is never logged."""
    load_dotenv()
    key = os.environ.get("MODEL_API_KEY", "").strip()
    want = os.environ.get("MODEL_PROVIDER", "meta" if key else "mock").lower()
    if want == "meta":
        if not key:
            raise RuntimeError("MODEL_PROVIDER=meta needs MODEL_API_KEY")
        from .meta import MetaConstraint, MetaPerception
        return MetaPerception(key), MetaConstraint(key)
    return MockPerception(), MockConstraint()
