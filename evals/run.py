"""Entry point: python evals/run.py --provider mock|replay|meta --manifest evals/manifest.jsonl --split dev|heldout --max-calls N

Run with the API virtualenv (apps/api/.venv). See evals/README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
