# Handoff — comment-class

Self-contained context doc for any AI agent (or human) picking up this project in a new session/window. Read this before touching anything. Updated as work progresses.

## Standing rules (do not violate)

1. **Never run git commands on this repo yourself.** Give the user the exact commands to run in their own terminal and wait for them to paste back output. This applies even to routine things (`git add`, `git commit`, `git push`) unless the user explicitly says otherwise in a given moment.
2. **Never add a `Co-Authored-By: Claude` (or any AI) trailer to commit messages, and never add a "Generated with Claude Code" line to commits or PRs.** The user's org does not allow AI co-authorship attribution.
3. **No Docker build/run verification** for this project — the user asked to stop doing local Docker checks. Verify with local `uvicorn`/Python only. (Memory-budget checks can still be done with plain Python + `psutil`, that's fine — just not `docker build`/`docker run`.)
4. User is on **Windows, PowerShell**. Watch for native-command quoting gotchas — PowerShell mangles nested double-quotes when passing arguments to `git.exe` and other native exes. Prefer single-line commands, or temp script files with absolute paths, over inline multi-line quoting.
5. `.claude/` is gitignored and untracked at the user's request — don't re-track it.
6. Keep this file updated as phases complete or context shifts.

## What this project is

`comment-class`: a 4-class comment-toxicity classifier (Normal / Offensive / Hate Speech / Severe-Violent), 21st/2,744 in a private Kaggle competition (macro-F1 ≈0.835). Classical ML — 7-model stacked ensemble (LightGBM + Logistic Regression + NB-SVM + LinearSVC), trained via `app/deployment_training_fast.ipynb`, exported as `comment_classifier_pipeline.joblib`.

Upgraded from "a Kaggle writeup" into a deployed, explainable moderation **platform** — see "v2" below.

## Current deployment status

**Live URL:** `https://comment-classification-bs2t.onrender.com` — as of this writing still serving **v1 (Gradio)**; v2 is built and locally verified but **not yet pushed/deployed** (see "What's left" below).

- Hosted on Render (free tier, 512MB RAM), via a Dockerfile in `app/`.
- Getting a real model onto Render took real debugging (HF Spaces tried and abandoned — Python-version/sklearn-pin build failures, then a ZeroGPU hardware requirement neither Gradio nor Docker SDK could avoid without a paid plan). Render + a memory-trimmed model artifact (single fold instead of 3-fold bagging, ~287MB vs ~520MB) is what's live.
- Git history confirmed clean of AI co-author trailers (`git log --all --oneline | grep Co-Authored-By` returns empty).

## v2 — DONE (this session), not yet deployed

**Why:** Gradio's built-in chrome fights custom design, and the user wants a portfolio-grade UI. This also unblocks the "explainable, auditable moderation platform" angle (RAG + agent + audit) that hits the specific skills 2026 India AI/ML fresher/associate job postings screen for.

**What was built — all verified locally working end-to-end via real HTTP requests (not just unit-level):**

1. **`app/engine.py`** — inference logic (text cleaning, feature engineering, model loading, bagged-fold prediction), lifted from the old `app/app.py`, Gradio-free. Gotcha handled: the model artifact was pickled from a Kaggle notebook where `NBTransformer` lived in `__main__`; `engine.py` patches `sys.modules['__main__'].NBTransformer` so unpickling works regardless of who imports it.
2. **`app/rag.py`** — TF-IDF retrieval (scikit-learn, no new heavy dependency) over `app/policy/*.md` (IT Rules 2021 excerpt + authored community guidelines). Explanation generation: Claude API (`claude-opus-5`) if `ANTHROPIC_API_KEY` is set, else a deterministic template. Verified both the retrieval-picks-the-right-section behavior and the template fallback path end-to-end; the LLM path is implemented but untested in this environment (no API key available here) — code gracefully falls back if `anthropic` import fails or no key is set.
3. **`app/agent.py`** — LangGraph state graph: `assess_risk` → conditional routing → `auto_action` / `human_review` / `auto_clear`, all logged to the audit trail. Plus `agent.appeal(decision_id, reason)` — re-runs classification+RAG, logs a new linked decision (`appeal_of` FK), doesn't overwrite the original (audit trail stays immutable).
4. **`app/audit.py`** — SQLite (stdlib, zero new dependency) at `app/audit.db` (gitignored, `*.db`). Functions: `log_decision`, `get_decision`, `update_decision`, `get_recent`, `get_stats`, `get_label_distribution`, `get_appeal_counts`, `get_activity_heatmap`. **Deliberately kept as local SQLite, not migrated to a cloud DB (Neon/Turso considered and explicitly declined)** — user confirmed the Render-redeploy reset behavior is an acceptable, disclosed demo limitation.
5. **`app/monitoring.py`** — model-health/drift proxy signals: Population Stability Index (predicted-class distribution vs. the README-documented validation baseline `{Normal: 0.50, Offensive: 0.072, Hate Speech: 0.40, Severe/Violent: 0.028}`), average confidence, appeal rate. **Explicitly not a live F1 score** — there's no ground truth for live traffic, and the code/UI/README all say so. Thresholds: PSI <0.10 healthy, 0.10–0.25 watch, ≥0.25 drift_detected (standard MLOps convention).
6. **`app/main.py`** — FastAPI app: `POST /api/analyze` (predict + explain + route + log), `POST /api/appeal/{id}`, `GET /api/audit`, `GET /api/monitoring`, `GET /api/health`, serves `static/`.
7. **`app/static/`** — light theme (per user's reference image), card-based, top nav with **Moderate** and **Audit Log** tabs. Moderate view: form + hero result + probability bars + linguistic signals + policy-explanation card + agent-status chip + appeal box. Audit Log view: stat-card row + model-health panel (drift badge, distribution-vs-baseline bars) + activity heatmap (CSS grid, day×hour) + recent-decisions list. No external chart-library CDN — hand-rolled SVG/CSS.
8. **`app/old_archive/gradio_app/`** — the v1 Gradio app archived as a **fully self-contained, independently deployable unit**: its own `app.py`, `requirements.txt`, `Dockerfile`, a **copy** of the trained model artifact, and its own `README.md` with run/deploy instructions. Verified it boots and loads the real model correctly from this location on its own (ran it as `python app.py` from inside that folder — "Full Stacked Ensemble Active").
9. **Memory check (plain Python + `psutil`, not Docker):** full v2 stack (engine + rag + agent + audit + monitoring + fastapi + langgraph + anthropic client) uses ~340MB RSS, vs. ~287MB for v1 alone — still comfortably under Render's 512MB cap (~172MB headroom).
10. **Cleanup done:** `app/Dockerfile` now runs `uvicorn main:app` (was `python app.py`); `app/requirements.txt` swapped `gradio` for `fastapi`/`uvicorn[standard]`/`anthropic`/`langgraph`; root `README.md` and `app/README.md` rewritten for v2; `.gitignore` updated (`*.db`, `.env*`, exception for the archived joblib copy); added `.env.example`.

## What's left / next steps

- **Nothing left to build for this pass.** Everything above is implemented and verified locally.
- **Not yet committed or pushed.** Per the "never run git commands" rule, the user needs to run the commit/push themselves — exact commands were provided in the conversation (see the final assistant message of this session, or re-derive: `git add` the new/modified files, commit, push to `main`; Render should auto-redeploy from there).
- **Open question flagged to user, unresolved as of this doc's last update:** `Competition.png` and `Leaderboard.png` (root-level duplicates of `assets/leaderboard.png` / `assets/competition-overview.png`) show as deleted from the working directory in `git status`, but this wasn't something done intentionally in this session's visible actions. The canonical copies under `assets/` are untouched and that's what `README.md` actually links to, so no content was lost — but confirm with the user whether to include this deletion in the commit or restore the files before committing.
- **Once deployed:** confirm on the live Render URL that `/api/health` shows the real model active, run through the same 4 preset cases in the browser, and spot-check that `ANTHROPIC_API_KEY` is (or isn't, deliberately) set on Render depending on whether the user wants live Claude explanations vs. the template fallback in production.

## Key file map

```
app/
  engine.py, main.py, rag.py, agent.py, audit.py, monitoring.py   # all DONE, verified
  static/{index.html, css/style.css, js/app.js}                  # DONE
  policy/{it_rules_2021.md, community_guidelines.md}              # DONE
  comment_classifier_pipeline.joblib                              # real trained artifact, DO NOT retrain
  requirements.txt                                                # DONE (v2 deps)
  Dockerfile                                                       # DONE (uvicorn CMD)
  old_archive/gradio_app/                                          # DONE — self-contained v1, verified working standalone
    app.py, requirements.txt, Dockerfile, comment_classifier_pipeline.joblib, README.md
README.md, app/README.md                                          # DONE (rewritten for v2)
.gitignore, .env.example                                          # DONE
```

All new/modified v2 files are currently **uncommitted** (untracked or modified in git status) — nothing has been pushed yet.
