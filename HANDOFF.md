# Handoff — comment-class

Self-contained context doc for any AI agent (or human) picking up this project in a new session/window. Read this before touching anything. Updated as work progresses.

## Standing rules (do not violate)

1. **Never run git commands on this repo yourself.** Give the user the exact commands to run in their own terminal and wait for them to paste back output. This applies even to routine things (`git add`, `git commit`, `git push`) unless the user explicitly says otherwise in a given moment.
2. **Never add a `Co-Authored-By: Claude` (or any AI) trailer to commit messages, and never add a "Generated with Claude Code" line to commits or PRs.** The user's org does not allow AI co-authorship attribution. If you're using a harness that adds this automatically, disable it or strip it before ever suggesting a commit message.
   - Context: an earlier commit (`e9e9fcb...`, now rewritten) carried this trailer. It took a multi-step `git rebase -i` + amend + force-push to remove it — see git log for the real history. It's fixed; don't reintroduce it.
3. **No Docker build/run verification** for this project going forward — the user asked to stop doing local Docker checks. Verify with local `uvicorn`/Python only.
4. User is on **Windows, PowerShell**. Watch for native-command quoting gotchas — PowerShell mangles nested double-quotes when passing arguments to `git.exe` and other native exes. Prefer single-line commands, or temp script files with absolute paths, over inline multi-line quoting.
5. `.claude/` is gitignored and untracked at the user's request (local Claude Code config, not project content) — don't re-track it.
6. Keep this file (`HANDOFF.md`) updated as phases complete or context shifts — this is the source of truth for "what's going on," not the conversation history.

## What this project is

`comment-class`: a 4-class comment-toxicity classifier (Normal / Offensive / Hate Speech / Severe-Violent) that placed 21st/2,744 (top 0.8%, macro-F1 ≈0.835) in a private Kaggle competition (IIT Madras community competition). Classical ML stack — no deep learning: a 7-model stacked ensemble (LightGBM + Logistic Regression + NB-SVM + LinearSVC) trained via the Kaggle notebook `app/deployment_training_fast.ipynb`, exported as `app/comment_classifier_pipeline.joblib`.

Originally this was "just" a Kaggle writeup repo. It's being upgraded into a portfolio-grade, deployed, explainable moderation platform — see "v2 plan" below for why and what.

## Current deployment status (v1 — LIVE)

**Live URL:** `https://comment-classification-bs2t.onrender.com`

- Hosted on **Render** (free tier, 512MB RAM), deployed via a `Dockerfile` in `app/` that binds to Render's dynamic `$PORT` env var.
- The deployed app is currently `app/app.py` — a **Gradio** dashboard. This will be replaced by the v2 FastAPI + custom-UI stack (in progress, see below), but as of now Gradio is still what's live.
- The model artifact (`app/comment_classifier_pipeline.joblib`) is the **real** Kaggle-trained model, trimmed to a single fold (from 3-fold bagging) specifically to fit Render's 512MB cap — full 3-fold version used ~520MB and OOM'd; trimmed version uses ~287MB. This trimming was verified to not meaningfully hurt prediction quality.
- Getting here took real debugging: HF Spaces was tried first and abandoned (Python-version/sklearn-pin build failures, then a ZeroGPU hardware requirement neither Gradio nor Docker SDK could avoid without a paid plan) — Render was the fallback that worked. Full history is in the conversation/commit log if needed; not repeated here since it's resolved.
- Deployment is git-push-triggered on Render's side — pushing to `main` (or whatever branch Render is watching) redeploys automatically.

**Everything is currently clean**: git history has no AI co-author trailers anywhere (verified via `git log --all --oneline | grep Co-Authored-By` returning empty), local `main` matches `origin/main`, no uncommitted changes except the new v2 files being built (untracked, see below).

## v2 plan — why and what

**Why:** Two things converged. (1) Fixing a cosmetic bug in the Gradio UI (an empty box from Gradio's built-in Settings/API toolbar) surfaced that Gradio's chrome fights any real custom design, and the user wants a genuinely well-built portfolio UI — feasible now that deployment is a plain Dockerfile on Render, not HF Spaces' managed Gradio SDK. (2) This unblocks turning the project from "a toxicity classifier" into an **explainable, auditable moderation platform** — RAG policy-grounding + agent escalation + audit trail — which hits the specific skills 2026 India AI/ML fresher/associate job postings screen for (RAG, agents, LLM integration, a real deployed service) far better than a Kaggle-writeup Gradio demo does.

**Full plan doc:** `C:\Users\athar\.claude\plans\yea-but-this-as-giggly-alpaca.md` (on the machine that has it — if unavailable, this section is the authoritative summary).

**Design direction:** Light theme (per user's reference image: a cream/white crypto-portfolio dashboard with stat cards, donut charts, an area chart), with structural patterns borrowed from a second reference (a dark fintech dashboard called "Loud" — activity heatmap, stat-row layout, top nav) recolored for the light palette. Indigo/purple accent, consistent with the old Gradio brand.

**Hard constraint:** Render free tier = 512MB RAM, and the classifier alone already uses ~287MB. Every new v2 dependency must avoid heavy ML libraries — no `sentence-transformers`, no `torch`, no vector-DB server. RAG retrieval uses **TF-IDF** (already an `sklearn` dependency, no new install). The LLM explanation layer is a **network call** to the Anthropic API (`claude-opus-5`), not a locally-loaded model — zero local memory cost — with a deterministic template fallback when no `ANTHROPIC_API_KEY` is configured, so the public demo works with zero secrets.

**Architecture — replace Gradio with FastAPI + hand-built HTML/CSS/JS, add three layers on top of the existing (proven) classifier:**

1. **`app/engine.py`** — inference logic (text cleaning, feature engineering, model loading, bagged-fold prediction) lifted out of `app/app.py`, Gradio-free. Single source of truth for the FastAPI server.
2. **`app/rag.py`** — small curated policy corpus (IT Rules 2021 intermediary-guidelines excerpt + an authored platform community-guidelines doc, local markdown files), TF-IDF-retrieved against the flagged comment. Explanation generator: Claude API call if a key is set, else a template built from the retrieved clause + lexicon signals `engine.py` already computes.
3. **`app/agent.py`** — a small **LangGraph** state graph (pure Python, negligible memory): `classify → decide → {auto_action | human_review | auto_clear}`, plus an `appeal` path that re-runs classification+RAG with the user's appeal context. LangGraph specifically (not a hand-rolled if/else) because it's the concrete "agent orchestration" artifact recruiters look for, and it's light enough for the memory budget.
4. **`app/audit.py`** — SQLite (stdlib `sqlite3`, zero new dependency) logging every decision. **Disclosed limitation:** Render free-tier disk is ephemeral, resets on redeploy/restart — fine and disclosed for a demo, not a claim of production data retention.
5. **`app/main.py`** — FastAPI app serving the static frontend + REST endpoints: `POST /api/analyze`, `POST /api/appeal/{id}`, `GET /api/audit`, `GET /api/health`.
6. **`app/static/`** — the new custom UI: `index.html`, `css/style.css`, `js/app.js`. Light theme, card-based, top nav with **Moderate** and **Audit Log** views, hand-rolled SVG/CSS charts (no external chart-library CDN — keeps the page self-contained).

`app/app.py` (Gradio) moves to `app/old_archive/app_gradio.py` at the end — kept for history, not deployed. `Dockerfile` CMD switches from `python app.py` to `uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}`.

### Phases

- **Phase A — Backend + UI shell.** `engine.py` + `main.py` + static shell, Moderate view working end-to-end with real predictions, no RAG/agent yet.
- **Phase B — RAG policy grounding.** `rag.py` + policy corpus, explanation card wired into the Moderate view.
- **Phase C — Agent escalation + audit.** `agent.py` (LangGraph) + `audit.py` (SQLite), Audit Log view (stat cards, heatmap, recent-decisions list).
- **Phase D — Cleanup & redeploy prep.** Archive Gradio app, update `Dockerfile`/`requirements.txt`/READMEs, verify locally, hand back to user for GitHub push + Render redeploy.

## Phase status (detailed)

- [x] **Phase A — in progress, nearly done:**
  - [x] `app/engine.py` — done, verified standalone (loads real model, predictions match previously-verified values exactly: Severe/Violent 97.4%, Normal 99.9%, Hate Speech 46.2%). Gotcha handled: the model artifact was pickled from a Kaggle notebook where `NBTransformer` lived in `__main__`, so `engine.py` patches `sys.modules['__main__'].NBTransformer` so unpickling works regardless of who imports it.
  - [x] `app/main.py` — done (Phase A scope: `/api/analyze`, `/api/health`, static file serving).
  - [x] `app/static/index.html` — done (Moderate view: form + hero result + prob bars + linguistic signals; Audit Log view is a stub for now).
  - [x] `app/static/css/style.css` — done (light theme design system).
  - [x] `app/static/js/app.js` — done (tabs, sliders, presets, fetch → render).
  - [ ] **Next: local verification** — `uvicorn app.main:app --reload` from the `app/` directory, then exercise `/api/analyze` via curl and the browser UI for the 4 preset cases (Normal/Offensive/Hate Speech/Severe-Violent), confirm predictions match v1's already-verified values.
- [ ] **Phase B:** not started.
- [ ] **Phase C:** not started.
- [ ] **Phase D:** not started.

## Key file map

```
app/
  engine.py          # inference: clean_text, build_features, model load, predict() — DONE
  main.py            # FastAPI app — DONE (Phase A endpoints only so far)
  rag.py             # NOT YET BUILT (Phase B)
  agent.py           # NOT YET BUILT (Phase C)
  audit.py           # NOT YET BUILT (Phase C)
  static/
    index.html       # DONE (Moderate view; Audit Log stub)
    css/style.css    # DONE
    js/app.js        # DONE
  policy/            # NOT YET BUILT (Phase B) — it_rules_2021.md, community_guidelines.md
  comment_classifier_pipeline.joblib   # real trained artifact, already present, DO NOT retrain
  app.py             # current Gradio app — STILL THE DEPLOYED ENTRYPOINT until Phase D archives it
  requirements.txt   # still Gradio-era; Phase D updates this
Dockerfile           # still `CMD python app.py`; Phase D switches to uvicorn
```

All the new v2 files above (`engine.py`, `main.py`, `static/`) are currently **untracked** in git — not yet committed. The user has not been asked to commit/push v2 work yet; that happens at the end of Phase D per the "no git commands from me" rule (I hand over exact commands).
