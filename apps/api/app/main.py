from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import pipeline
from .providers import get_providers, provider_mode
from .providers.http import CallBudgetExceeded
from .providers.recording import ReplayMissError

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "apps" / "web"
DEMO = ROOT / "apps" / "api" / "demo_data"
MAX_UPLOAD = 15 * 1024 * 1024

app = FastAPI(title="AccessShift", version="0.1.0")


@app.exception_handler(pipeline.PipelineError)
async def _pe(_, exc: pipeline.PipelineError):
    return JSONResponse({"error": str(exc)}, status_code=exc.code)


@app.exception_handler(ReplayMissError)
async def _replay_miss(_, exc: ReplayMissError):
    return JSONResponse({"error": str(exc), "kind": "replay_miss"}, status_code=502)


@app.exception_handler(CallBudgetExceeded)
async def _budget(_, exc: CallBudgetExceeded):
    return JSONResponse({"error": str(exc), "kind": "call_budget"}, status_code=503)


def _run(rid: str) -> pipeline.Run:
    r = pipeline.RUNS.get(rid)
    if not r:
        raise HTTPException(404, "unknown run")
    return r


async def _read(f: UploadFile) -> bytes:
    b = await f.read(MAX_UPLOAD + 1)
    if len(b) > MAX_UPLOAD:
        raise HTTPException(413, "photo too large")
    return b


@app.get("/api/health")
def health():
    p, c, pl = get_providers()
    return {"ok": True, "perception": p.model, "constraints": c.model, "planner": pl.model,
            "provider": provider_mode(), "mock": p.name == "mock", "scope": pipeline.SCOPE}


@app.post("/api/runs")
async def create_run(photo: UploadFile = File(...), board_w_cm: float = Form(...), board_h_cm: float = Form(...),
                     check_len_cm: float = Form(...), constraint_text: str = Form(..., max_length=2000),
                     anchors: str = Form("{}")):
    try:
        anc = json.loads(anchors)
        anc = {str(k): [float(v[0]), float(v[1])] for k, v in anc.items()}
    except Exception:
        raise HTTPException(422, "anchors must be JSON {name: [x_cm, y_cm]}")
    run = pipeline.start_run(await _read(photo), board_w_cm, board_h_cm, check_len_cm, constraint_text, anc)
    return pipeline.run_view(run)


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    return pipeline.run_view(_run(rid))


class ConfirmBody(BaseModel):
    locked_ids: list[str] | None = None
    accept_review_ids: list[str] | None = None
    grid_step_cm: float = 10.0
    time_budget_s: float = 8.0
    # manual constraints: the human-confirmation fallback when the planner agent could not finish
    clearance_cm: float | None = None
    start_name: str | None = None
    end_name: str | None = None
    max_moves: int | None = None


@app.post("/api/runs/{rid}/confirm")
def confirm(rid: str, body: ConfirmBody = ConfirmBody()):
    manual = {k: getattr(body, k) for k in ("clearance_cm", "start_name", "end_name", "max_moves")
              if getattr(body, k) is not None}
    run = pipeline.confirm(_run(rid), body.locked_ids, body.accept_review_ids,
                           max(body.grid_step_cm, 5.0), min(body.time_budget_s, 30.0), manual or None)
    return pipeline.run_view(run)


class AnswerBody(BaseModel):
    answer: str = Field(min_length=1, max_length=500)


@app.post("/api/runs/{rid}/answer")
def answer(rid: str, body: AnswerBody):
    """Reply to the planner agent's ask_user question; the agent resumes from where it paused."""
    return pipeline.run_view(pipeline.answer(_run(rid), body.answer))


@app.get("/api/runs/{rid}/plan")
def get_plan(rid: str):
    v = pipeline.run_view(_run(rid))
    if not v["plan"]:
        raise HTTPException(409, "not planned yet; confirm first")
    return v["plan"]


@app.post("/api/runs/{rid}/recapture")
async def recapture(rid: str, photo: UploadFile = File(...)):
    run = pipeline.recapture(_run(rid), await _read(photo))
    return pipeline.run_view(run)["verification"]


@app.get("/api/runs/{rid}/trace")
def trace(rid: str):
    return pipeline.trace_view(_run(rid))


@app.get("/api/runs/{rid}/photo/{which}")
def photo(rid: str, which: str):
    b = _run(rid).photos.get(which)
    if not b:
        raise HTTPException(404, "no such photo")
    return Response(b, media_type="image/png")


@app.get("/api/demo")
def demo_info():
    from .demo import ensure_demo
    return ensure_demo(DEMO)


@app.get("/api/demo/{name}.png")
def demo_image(name: str):
    from .demo import ensure_demo
    ensure_demo(DEMO)
    p = DEMO / f"{name}.png"
    if name not in {"blocked", "correct", "wrong", "wrong_open"} or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
