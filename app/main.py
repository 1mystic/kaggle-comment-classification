"""FastAPI backend for the Comment Trust & Safety dashboard.

Serves the static frontend (static/) and exposes the JSON API it calls.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine
import rag

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="Comment Trust & Safety Dashboard")


class AnalyzeRequest(BaseModel):
    comment: str = Field(..., min_length=1)
    if_1: float = 0
    if_2: float = 5
    upvotes: float = 0
    downvotes: float = 0
    race: str = "none"
    religion: str = "none"
    gender: str = "none"


@app.get("/api/health")
def health():
    return {"status": "ok", "model_active": engine.is_model_active(), "model_status": engine.MODEL_STATUS_TEXT}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    result = engine.predict(
        comment_text=req.comment,
        if_1=req.if_1,
        if_2=req.if_2,
        upvotes=req.upvotes,
        downvotes=req.downvotes,
        race=req.race,
        religion=req.religion,
        gender=req.gender,
    )
    result["explanation"] = rag.explain(req.comment, result)
    return result


# Static assets (css/js) under /static, index.html served at the root.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
