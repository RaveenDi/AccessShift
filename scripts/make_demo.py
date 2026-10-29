"""Generate the synthetic demo photos into apps/api/demo_data (run from anywhere)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.demo import ensure_demo  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "apps" / "api" / "demo_data"
    for p in out.glob("*.png"):
        p.unlink()
    print(ensure_demo(out))
