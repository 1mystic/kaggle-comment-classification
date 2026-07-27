---
title: Comment Toxicity & Category Classification Dashboard
emoji: 🛡️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
---

# Archived: standalone classical-ML Gradio dashboard (v1)

This is the **v1 deployment** of the comment-toxicity classifier — a self-contained
Gradio dashboard backed by the real trained 7-model stacked ensemble, preserved
here after the project moved to the FastAPI + custom-UI v2 stack (see
`app/main.py` and `app/static/` one directory up, and `HANDOFF.md` at the repo
root for the full v1 → v2 story).

**This folder is deliberately self-contained** — everything needed to build and
run this exact version independently, with no dependency on anything else in
the repo:

- `app.py` — the Gradio app, unchanged from the last working v1 deployment.
- `comment_classifier_pipeline.joblib` — the real trained model artifact
  (single-fold-trimmed to fit a 512MB memory budget; see the root README for
  why it's trimmed and what that cost in accuracy — negligible).
- `requirements.txt` — pinned to the exact versions verified working.
- `Dockerfile` — builds and runs this app on plain CPU, binding to a dynamic
  `$PORT` (Render-compatible) with a `7860` fallback for local `docker run` or
  other PaaS hosts.

## Run locally

```bash
cd app/old_archive/gradio_app
pip install -r requirements.txt
python app.py
```

## Deploy elsewhere

Point any Docker-based host (Render, Fly.io, a plain VM, etc.) at this folder
as the build context — it needs nothing from the rest of the repo:

```bash
cd app/old_archive/gradio_app
docker build -t comment-classifier-gradio .
docker run -p 7860:7860 comment-classifier-gradio
```

## Why this was archived, not deleted

The v2 stack (FastAPI + custom UI + RAG policy explanations + LangGraph agent
routing + audit trail) fully supersedes this for the live deployment, but this
version is kept working and deployable on its own in case you want a minimal,
dependency-light classical-ML demo without the Stage-2 layers — e.g. for a
quick standalone portfolio link, or as a reference implementation.
