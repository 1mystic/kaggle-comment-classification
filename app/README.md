# Comment Trust & Safety Dashboard (v2)

A real-time comment moderation platform: the original 7-model classical-ML stacked ensemble (LightGBM, Logistic Regressions, NB-SVM, LinearSVC — unchanged from v1), wrapped with a RAG policy-explanation layer, a LangGraph agent-escalation workflow, an audit trail, and a model-health/drift monitoring panel. Custom light-theme UI (FastAPI + hand-built HTML/CSS/JS), no Gradio.

See `HANDOFF.md` at the repo root for the full architecture writeup and why this exists.

## Features

- **Live classification** — real-time multiclass prediction across Normal, Offensive, Hate Speech, and Severe/Violent, via the real trained model artifact.
- **Policy-grounded explanations** — every flagged comment is paired with a retrieved policy clause (TF-IDF over a local policy corpus) and a human-readable explanation: Claude-generated when `ANTHROPIC_API_KEY` is set, a deterministic template otherwise. Works with zero secrets configured.
- **Agent escalation** — a small LangGraph workflow routes each decision to auto-action, human review, or auto-clear, with an appeal path that re-evaluates and logs a new, linked decision.
- **Audit trail** — every decision logged to SQLite (ephemeral on Render's free tier — ok for a demo, not a production data-retention claim).
- **Model health / drift monitoring** — proxy signals only (no ground-truth labels exist for live traffic): predicted-class distribution drift vs. the validation-set baseline (Population Stability Index), average confidence, and appeal rate. Explicitly not a live F1/accuracy score.

## Run locally

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

## Module map

| File | Purpose |
|---|---|
| `engine.py` | Text cleaning, feature engineering, model loading, prediction |
| `main.py` | FastAPI app — REST endpoints + static file serving |
| `rag.py` | Policy retrieval (TF-IDF) + explanation generation (Claude API / template fallback) |
| `agent.py` | LangGraph escalation workflow + appeal handling |
| `audit.py` | SQLite decision logging |
| `monitoring.py` | Drift/confidence/appeal-rate proxy signals |
| `static/` | Frontend (`index.html`, `css/style.css`, `js/app.js`) |
| `policy/` | Local policy corpus (IT Rules 2021 excerpt + authored community guidelines) |
| `old_archive/gradio_app/` | The v1 Gradio dashboard, preserved as a self-contained, independently deployable unit |

## Deployment

Live on Render via the `Dockerfile` in this directory (`uvicorn main:app`, binds to `$PORT`). See root `README.md` for the live link and `HANDOFF.md` for deployment history/gotchas (HF Spaces was tried and abandoned; Render is what's live).
