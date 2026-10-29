import os
from typing import NamedTuple

from dotenv import load_dotenv

from ..config import record_enabled, recordings_dir
from .mock import MockConstraint, MockPerception
from .mock_policy import MockPlanner

MODES = ("mock", "meta", "replay")


def provider_mode() -> str:
    """MODEL_PROVIDER=mock|meta|replay; defaults to meta when MODEL_API_KEY is set, else mock."""
    load_dotenv()
    key = os.environ.get("MODEL_API_KEY", "").strip()
    want = os.environ.get("MODEL_PROVIDER", "meta" if key else "mock").strip().lower()
    if want not in MODES:
        raise RuntimeError(f"MODEL_PROVIDER must be one of {MODES}, got {want!r}")
    return want


class Providers(NamedTuple):
    perception: object
    constraint: object
    planner: object


def get_providers() -> Providers:
    """Returns (perception, constraint, planner). The key is read only from the environment and never logged.
    meta   : live Meta calls; every successful request/response pair is recorded under RECORDINGS_DIR.
    replay : answers from those recordings, no key and no network; a missing recording raises ReplayMissError."""
    mode = provider_mode()
    if mode == "mock":
        return Providers(MockPerception(), MockConstraint(), MockPlanner())
    from .meta import MetaConstraint, MetaPerception, MetaPlanner
    if mode == "replay":
        from .recording import ReplayTransport
        t = ReplayTransport(recordings_dir())
        return Providers(MetaPerception("replay-no-key", t), MetaConstraint("replay-no-key", t),
                         MetaPlanner("replay-no-key", t))
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MODEL_PROVIDER=meta needs MODEL_API_KEY")
    t = None
    if record_enabled():
        from .recording import live_transport
        t = live_transport(recordings_dir())
    return Providers(MetaPerception(key, t), MetaConstraint(key, t), MetaPlanner(key, t))
