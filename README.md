# AccessShift

From an overhead photo of a calibrated tabletop layout plus a plain-language constraint ("keep a 90 cm route from the door to the desk, lock the sofa"), propose the smallest rearrangement that opens the route, then verify the physical result from a recapture photo. Deterministic geometry decides connectivity; the model only perceives and interprets.

**Scope:** a calibrated tabletop model only. It is **not** accessibility compliance and not a real-world measurement. Meta model calls are **UNVERIFIED / NOT TESTED**; the default mock mode tests plumbing only.

## Quick start (mock mode, no API key)
```bash
cd apps/api
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock.txt   # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest -q                              # 18 tests
.venv/Scripts/python -m uvicorn app.main:app --port 8000       # open http://localhost:8000
```
In the UI: Load demo -> Confirm -> a recapture button (correct move / wrong move / other place). Demo photos are generated on first use (or `python scripts/make_demo.py`) into `apps/api/demo_data/` (gitignored).

API: `POST /api/runs` (photo, board_w_cm, board_h_cm, check_len_cm, constraint_text, anchors) -> `POST /api/runs/{id}/confirm` -> `GET /api/runs/{id}/plan` -> `POST /api/runs/{id}/recapture` -> verification; `GET /api/runs/{id}/trace`.

## Live Meta mode (not run yet)
Copy `.env.example` to `.env`, set `MODEL_API_KEY`, `MODEL_PROVIDER=meta`. Endpoint shapes in `apps/api/app/providers/meta.py` are UNVERIFIED and must be checked against dev.meta.ai first.

## Layout
`apps/api` (domain, providers, pipeline, tests) · `apps/web/index.html` (no build step) · `scripts/make_demo.py` · `docs/AGENT-ARCHITECTURE.md` · `docs/` planning · `spike/` original probe.
