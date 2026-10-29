import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evals"))

from app import demo  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "demo_data"


@pytest.fixture(scope="session")
def demo_dir():
    demo.ensure_demo(DATA)
    return DATA


@pytest.fixture
def use_providers(monkeypatch):
    """Route pipeline and API provider lookups to a given Providers tuple."""
    def _set(providers):
        monkeypatch.setattr("app.pipeline.get_providers", lambda: providers)
        monkeypatch.setattr("app.main.get_providers", lambda: providers)
        return providers
    return _set


def img(name):
    return (DATA / f"{name}.png").read_bytes()


def start(text=None, photo="blocked", anchors=None):
    from app import pipeline
    return pipeline.start_run(img(photo), demo.BOARD_W, demo.BOARD_H, demo.CHECK_LEN, text or demo.TEXT,
                              anchors or demo.ANCHORS)
