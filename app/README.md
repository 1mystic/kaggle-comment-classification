---
title: Comment Toxicity & Category Classification Dashboard
emoji: 🛡️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.21.0
app_file: app.py
pinned: false
license: mit
---

# Comment Category Prediction & Toxicity Intelligence Dashboard

An interactive real-time comment classification dashboard powered by a stacked 7-model ensemble (LightGBM, Logistic Regressions, NB-SVM, LinearSVC) trained on toxic text and moderation signals.

## Features
- **Live Classification**: Real-time multiclass prediction across Normal (C-0), Offensive (C-1), Hate Speech (C-2), and Severe/Violent (C-3).
- **Confidence Distribution**: Per-class probability scores with dynamic threshold multipliers.
- **Linguistic Analysis**: Token extraction, leet-speak decoding, sentiment balance, and platform signal detection.
- **Preset Testing**: One-click benchmark samples to evaluate toxicity boundaries.

## Usage
Run locally with:
```bash
cd app
pip install -r requirements.txt
python app.py
```
