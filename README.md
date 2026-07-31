# Comment Category Prediction: Top 0.8% (Rank 21 / 2,744) with Classical ML Only

Multi-class comment-toxicity classification on a severely imbalanced, obfuscated text dataset: **0.835 macro F1** (rank 21 of 2,744 participants) using a stacked ensemble of gradient-boosted trees and linear/Bayesian text models, no deep learning.

- **Competition:** [Comment Category Prediction Challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge/overview) (Kaggle, private community competition for IIT Madras students)
- **Result:** Rank **21 / 2,744** participants (3,154 entrants), top **0.8%**, score **0.83499** (macro-F1-based custom metric)
- **Stack:** scikit-learn (Logistic Regression, LinearSVC, ComplementNB), LightGBM, engineered features, no deep learning involved
- **Live Demo:** [comment-classification-bs2t.onrender.com](https://comment-classification-bs2t.onrender.com)

![Leaderboard](assets/leaderboard.png)

---

## Problem / Dataset

The task: predict the final moderation `label` (0–3) assigned to a short user comment on a discussion platform, given:

- **Text:** the raw `comment`
- **Engagement signals:** `upvote`, `downvote`, `emoticon_1/2/3`
- **Opaque platform signals:** `if_1`, `if_2`: undocumented internal scores computed by the platform's own moderation system
- **Topic/identity flags:** `race`, `religion`, `gender`, `disability`: booleans/categoricals indicating whether the system detected references to these topics
- **Metadata:** `post_id`, `created_date`

**Dataset size:** 198,000 training rows × 15 columns, 102,000 test rows × 14 columns. No exact-duplicate rows were found. `race`/`religion`/`gender` are missing on 145,423 rows (73.45%) with identical null counts across all three, which points at a single group-entry failure rather than three independent optional fields; `comment` has exactly 1 null.

Four target classes, heavily imbalanced (exact training shares):

| Class | Interpretation (inferred from EDA, not documented by the organisers) | Share |
|---|---|---|
| 0 | Normal | **57.66%** |
| 1 | Identity-targeted / offensive | **8.04%** |
| 2 | Hate speech / general toxicity | **31.54%** |
| 3 | Severe / violent threats | **2.76%** (smallest) |

Note on naming: the deployed `CLASS_NAMES` map labels 1 as *Offensive* and 2 as *Hate Speech*, while the EDA evidence points the other way (Class 1 rows carry explicit identity terms; Class 2 rows are general insult/toxicity). The competition never published the mapping — every class interpretation here is inferred.

The text itself contains **obfuscated toxic language** (leet-speak substitutions such as `k1ll`, `d34d`, `h4te`, `$t@b`, deliberately used to dodge naive keyword filters), which is the core NLP challenge on top of the class imbalance.

## My Approach

### 1. Text cleaning (obfuscation handling)
Order matters here: obfuscation decoding has to happen **before** non-alphabetic characters are stripped, otherwise the substituted digits/symbols are destroyed along with the noise:

1. Lowercase
2. Strip URLs / HTML tags
3. **Leet-speak decode** (`0→o, 1→i, 3→e, 4→a, 5→s, 7→t, @→a, $→s, !→i`) while digits/symbols still exist
4. Expand common abbreviations (`kys` → "kill yourself", `wtf` → "what the fuck", etc.)
5. Strip remaining non-alphabetic characters, collapse whitespace

### 2. Feature engineering
**56** hand-crafted numeric features, validated by enrichment ratio / mutual information / Kruskal-Wallis before inclusion (features below a ~2× class-enrichment threshold were dropped — `race_flag`/`religion_flag` came in at ~0.76× and were cut):

- **Text complexity:** word/char counts, lexical diversity, capitalization ratio
- **Violence lexicon:** `violence_count`, `violence_ratio` (Class-3 mean 0.041 vs Class-0 0.005, **8.2×**), `short_violent` (≤15 words + a violent term: **42%** of such rows are Class 3, against a 2.76% base rate)
- **Sentiment lexicon:** hand-built positive/negative word-count features
- **Platform "zone" features:** `if_1 × if_2` interaction terms (`if_prod`, `if_ratio`, `zone_safe`, `golden_c1`): the single strongest signal in the dataset — Kruskal-Wallis H = **146,133** for `if_2` and **29,686** for `if_1`, against 1,102 for `downvote` and 28.8 for `emoticon_1`. `if_2 ≤ 7` alone identifies rows that are 99.2% Class 0. The platform's own opaque moderation score is itself a distillation of many hidden signals.
- **Identity reference flags:** derived from `race`/`religion`/`gender`, kept as their own category rather than collapsing `NaN` and `"none"` together: a user explicitly selecting "none" behaves differently from one who left the field blank
- **Temporal:** cyclical hour-of-day encoding (`sin`/`cos`), weekend flag; weak but free signal
- **Engagement:** vote ratio, controversy score

Top features by mutual information: `zone_safe` 0.513, `if_2` 0.498, `if_sum` 0.465, `golden_c2` 0.329, `zone_c23` 0.268, `danger_zone` 0.267, `if_ratio` 0.197, `if_prod` 0.156 — platform signals sweep the top of the table, ahead of every text-derived feature.

Text itself is represented with **three complementary TF-IDF spaces**, concatenated into one sparse matrix (55,000 text columns; with the 56 numeric and 3 ordinal-encoded categorical columns the full matrix is **55,059** wide):
- Word-level (1–2 grams, 35,000 features, `min_df=2`, `max_df=0.92`, `sublinear_tf`): general vocabulary/semantics
- **Character-level (char_wb, 3–5 grams, 10,000 features, `min_df=3`)**: the main defense against obfuscation; catches `k1ll`/`kill` as near-identical patterns even after imperfect leet-decoding
- Phrase-level (2–3 grams, 10,000 features, `min_df=3`): multi-word toxic phrases (`"kill yourself"`)

Each base model that consumes text directly carries its *own* vectoriser inside a `sklearn.Pipeline` (LR-Word 60k word 1–3 grams; LR-Char 30k char 3–6 grams; NB-SVM 50k word 1–2 grams), so those are fit on training folds only.

### 3. Imbalance handling
- Per-class weights in LightGBM (`{0: 1.0, 1: 3.5, 2: 1.5, 3: 15.0}`): tuned rather than derived directly from the raw imbalance ratio, to balance precision/recall rather than just oversampling the loss on the rarest class
- `class_weight='balanced'` on all linear/SVM baselines
- **Post-hoc per-class threshold multipliers**, tuned via Nelder-Mead, applied to the final probability vector before `argmax`: the meta-model systematically under-predicted the 2.76%-share Class 3 even after class weighting, so thresholds were loosened specifically for it. Final multipliers `[1.032, 0.951, 0.85, 1.25]`, found by Nelder-Mead against OOF macro F1 with bounds `[0.8–1.2, 0.7–2.5, 0.85–1.1, 0.8–1.25]` — deliberately tight, to avoid overfitting the multipliers themselves

### 4. Models & validation
Seven base models, each trained under **3-fold Stratified K-Fold CV** (`SEED=42`) to produce out-of-fold (OOF) predictions, with their individual OOF macro F1:

| Model | Input | Role | OOF macro F1 |
|---|---|---|---|
| LightGBM (multiclass) | Full feature matrix (TF-IDF + engineered + categorical) | Primary model; handles non-linear interactions natively | **0.8237** |
| NB-SVM | Word TF-IDF × NB log-count ratio | Bayesian-weighted linear model (Wang & Manning, 2012) | 0.6915 |
| Logistic Regression (word n-grams) | Word TF-IDF | Fast, diverse baseline | 0.6869 |
| Logistic Regression (char n-grams) | Char TF-IDF | Obfuscation-robust linear view | 0.6590 |
| Calibrated LinearSVC | Word TF-IDF | Max-margin diversity, probability-calibrated | 0.649–0.656 per fold |
| LightGBM (Class-3-vs-rest) | Full feature matrix | Specialist binary detector for the rarest class | 0.799–0.801 (binary) |
| LightGBM (Class-3-vs-Class-2) | Full feature matrix | Specialist for the hardest pairwise confusion | 0.839–0.845 (binary) |

Final hyperparameters, tuned via `RandomizedSearchCV`/`GridSearchCV` against macro-F1 on held-out folds: LightGBM `n_estimators=1000, learning_rate=0.03, num_leaves=127, max_bin=127, colsample_bytree=0.5, subsample=0.9, min_child_samples=10`, early stopping at 30 rounds (fires around 400–600); LR-Word `C=5.395`; LR-Char `C=4.464`; NB-SVM `C=2.0`; LinearSVC `C=0.184`; ComplementNB `alpha=0.5` (tuned, but not part of the final seven).

**Leakage caveat, stated plainly:** the three shared TF-IDF vectorisers and the `OrdinalEncoder` feeding the LightGBM feature matrix are fit once on the full training set *before* the CV loop, so the OOF estimate carries mild unsupervised leakage (vocabulary and IDF statistics, no labels). The per-model pipelines listed above do not. The final leaderboard score landed within 0.002 of the OOF estimate, which is the empirical evidence it did not inflate the result materially — but fitting inside the fold is the correct fix.

### 5. Ensembling
The 7 base models' OOF probability outputs are concatenated into a 24-column matrix (4+4+4+4+2+4+2) and fed into a **shallow LightGBM meta-learner** (`num_leaves=16, n_estimators=100, learning_rate=0.05, min_child_samples=200`), trained with its own 5-fold CV: this beat a Logistic Regression meta-learner in internal validation (**0.8339 vs 0.8296** OOF macro F1). A meta-learner was preferred over simple averaging because it learns *which base model to trust per class*, rather than treating all seven as equally reliable everywhere.

## Results

| Stage | Macro F1 (internal OOF) |
|---|---|
| Dummy baseline (always predict the majority class) | 0.213 |
| Best untuned model in the 13-model bake-off (MLP) | 0.6724 |
| Best single base model (tuned LightGBM) | 0.8237 |
| Stacked meta-learner (before threshold tuning) | 0.8337 |
| **Final (meta-learner + per-class thresholds)** | **0.8357** (+0.0020) |
| **Official leaderboard score** | **0.83499 (rank 21 / 2,744, top 0.8%)** |

Per-class F1 at the final configuration — overall accuracy 0.9192:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 Normal | 0.9782 | 0.9467 | **0.9622** | 114,173 |
| 1 Offensive | 0.7828 | 0.8247 | **0.8032** | 15,918 |
| 2 Hate Speech | 0.8758 | 0.9126 | **0.8939** | 62,440 |
| 3 Severe/Violent | 0.6718 | 0.6956 | **0.6835** | 5,469 |

![Leaderboard](assets/leaderboard.png)
![Competition overview](assets/competition-overview.png)

Class 3 (severe/violent, 2.76% of data) was the consistent bottleneck across every model: the smallest class, with the least identity-based signal to lean on, and the one requiring the largest threshold correction. The threshold multipliers deliberately bought Class-3 recall (0.6442 → 0.6956) at the cost of precision (0.7121 → 0.6718), because under macro F1 — and under the real-world cost of a missed threat — that trade is the right way round. Those multipliers were fitted on the same OOF predictions the 0.8357 figure is computed from, so that number is mildly optimistic; the tight search bounds are the mitigation, and the leaderboard result is the independent check.

## What I Learned / Key Insights

- **The platform's own opaque signals (`if_1`, `if_2`) were the single strongest predictor**, stronger than any hand-built lexicon feature. Sometimes the most valuable feature engineering is recognizing and properly encoding a signal you didn't build, rather than building a better one from scratch.
- **Obfuscation handling is an ordering problem, not just a mapping problem.** Decoding leet-speak before stripping symbols (and backing it up with character n-grams as a safety net) mattered more than the sophistication of the substitution table itself.
- **A shallow meta-learner beat naive averaging** by learning per-class trust across a diverse set of 7 base models. Diversity (linear, Bayesian, tree-based, specialist binary classifiers) mattered more than any single model's raw strength.
- **Fixing class imbalance is not one lever.** Class weights, stratified CV, and post-hoc threshold tuning each fixed a different failure mode; none of them alone was sufficient for the rarest class.
- **Classical ML remained fully competitive** on this task: the signal lived in engineered platform-interaction features and TF-IDF text representations, not in representations that would obviously benefit from a transformer.

## Reproducibility Note

This repo does not include competition data or the notebook I actually submitted (see below). The `src/` skeleton illustrates the pipeline structure (cleaning → feature engineering → TF-IDF → stacked ensemble) but is a **simplified, independently rewritten reference implementation**, not a drop-in reproduction. To reproduce meaningfully you would need:
- The original `train.csv`/`test.csv` (not published here; see Kaggle competition page, which may no longer be publicly accessible since this was a private competition)
- The full hand-tuned hyperparameter grids (only final chosen values are illustrated in `src/train.py`)
- The exact per-class threshold multipliers, which were fit once via Nelder-Mead on this specific dataset and would not transfer directly to another

## Note on Code Availability

This was a **private, competition-only Kaggle notebook** (course-hosted community competition, closed since March 2026). Kaggle's rules for private competitions mean the original submission notebook is not mine to freely redistribute, and this repo does not include it. What's here instead:

- An honest description of the methodology actually used (above), reconstructed directly from my own notebook
- An **independently rewritten**, simplified illustrative version of the pipeline in `src/`, containing no competition data and no verbatim competition code
- [`Notebook-main.ipynb`](Notebook-main.ipynb): a reconstructed copy of my approach, written after the fact. This is not the notebook I actually submitted to the competition, but a recreated version of it
- Leaderboard/result screenshots as evidence of the outcome, not the process

## Deployment & Interactive Web Platform (v2)

- **Production Training Notebook:** [`app/deployment_training_fast.ipynb`](app/deployment_training_fast.ipynb)
  Kaggle notebook containing the exact 7-model stacked ensemble pipeline, feature engineering, vectorization, meta-learner, and threshold multipliers, tuned to fit Kaggle's CPU time limit. Each base learner's `N_FOLDS` cross-validation copies are kept and bagged (averaged) at inference time instead of doing a separate full-data refit. When executed on Kaggle, it exports all fitted fold models to a compressed `comment_classifier_pipeline.joblib` artifact matching the schema `app/engine.py` expects.
- **Live app:** [`app/main.py`](app/main.py) — a FastAPI backend + custom HTML/CSS/JS dashboard (no Gradio) that turns the classifier into an explainable, auditable moderation platform:
  - Real-time classification via the same trained model, same feature pipeline.
  - **RAG-grounded explanations** — every flagged comment cites a retrieved policy clause (TF-IDF over a local corpus covering India's IT Rules 2021 + an authored community-guidelines doc), with a Claude-generated explanation when `ANTHROPIC_API_KEY` is set, or a deterministic template otherwise — the demo works with zero secrets.
  - **Agent escalation** — a small LangGraph workflow (`app/agent.py`) routes each decision to auto-action / human review / auto-clear, with an appeal path.
  - **Audit trail** — every decision logged to SQLite (`app/audit.py`).
  - **Model health / drift monitoring** (`app/monitoring.py`) — predicted-class distribution drift vs. the training-set class distribution documented above (Population Stability Index), average confidence, appeal rate. Explicitly a proxy signal set, not a live F1 score — there's no ground truth for live traffic.
  - See [`app/README.md`](app/README.md) for the full module map and how to run it locally.
- **Archived v1:** [`app/old_archive/gradio_app/`](app/old_archive/gradio_app/) — the original Gradio dashboard, preserved as a fully self-contained, independently deployable unit (its own `Dockerfile`, `requirements.txt`, and a copy of the trained model artifact) if you want the minimal classical-ML demo without the v2 layers.

### Running locally

```bash
cd app
pip install -r requirements.txt

# Optional — only needed for live Claude-generated explanations. Without this,
# the app falls back to a deterministic template automatically, no error, no
# missing feature — it's a fully valid default, not a degraded one.
cp ../.env.example ../.env
# then edit ../.env and set ANTHROPIC_API_KEY=sk-ant-...

uvicorn main:app --reload --port 8000
```

Open `http://127.0.0.1:8000`. The `.env` file (repo root, alongside `.env.example`) is auto-loaded via `python-dotenv` — no need to export the variable in your shell. It's gitignored, so it never gets committed.

To run the **archived v1 Gradio version** instead, see [`app/old_archive/gradio_app/README.md`](app/old_archive/gradio_app/README.md).

## Stack

`numpy` · `pandas` · `scikit-learn` · `lightgbm` · `joblib` · `fastapi` · `langgraph` · `anthropic`

See [requirements.txt](requirements.txt) and [app/requirements.txt](app/requirements.txt).

