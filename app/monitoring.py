"""Model health / drift monitoring - proxy signals, not a live F1 score.

Important honesty note: there is no ground truth for live traffic (nobody
confirms whether a prediction was actually correct), so a real live F1/recall
cannot be computed here. What CAN be computed honestly from data we already
log in audit.py:

  - Class-distribution drift: how far the live predicted-label mix has moved
    from the known validation-set baseline (documented in the root README),
    measured via Population Stability Index (PSI) - a standard MLOps drift
    metric, industry-conventional thresholds below.
  - Average prediction confidence over logged decisions.
  - Appeal rate: the fraction of original decisions that got appealed, a weak
    proxy for perceived misclassification rate (not a substitute for real
    labels, but a real, honestly-described signal).
"""

import math

import audit

# Actual training-set class shares, measured on all 198,000 training rows
# (Notebook-main.ipynb, `label.value_counts(normalize=True)`) and documented in
# the root README. An earlier version of this baseline was copied from a
# now-corrected README table and was wrong (0.50/0.072/0.40/0.028), which made
# every PSI reading below meaningless - the numbers here are the measured ones.
BASELINE_DISTRIBUTION = {
    "Normal": 0.5766,
    "Offensive": 0.0804,
    "Hate Speech": 0.3154,
    "Severe/Violent": 0.0276,
}

PSI_WATCH_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.25


def _live_distribution() -> dict:
    counts = audit.get_label_distribution()
    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: n / total for label, n in counts.items()}


def _population_stability_index(baseline: dict, live: dict) -> float:
    """Standard PSI: sum((live_pct - base_pct) * ln(live_pct / base_pct)) over categories."""
    eps = 1e-4
    psi = 0.0
    for label, base_pct in baseline.items():
        live_pct = live.get(label, 0.0)
        base_pct = max(base_pct, eps)
        live_pct_adj = max(live_pct, eps)
        psi += (live_pct_adj - base_pct) * math.log(live_pct_adj / base_pct)
    return psi


def _drift_status(psi: float) -> str:
    if psi >= PSI_DRIFT_THRESHOLD:
        return "drift_detected"
    if psi >= PSI_WATCH_THRESHOLD:
        return "watch"
    return "healthy"


def _appeal_rate() -> float:
    total_original, total_appealed = audit.get_appeal_counts()
    return (total_appealed / total_original * 100) if total_original else 0.0


def get_model_health() -> dict:
    live = _live_distribution()
    psi = _population_stability_index(BASELINE_DISTRIBUTION, live) if live else 0.0
    stats = audit.get_stats()
    return {
        "baseline_distribution": BASELINE_DISTRIBUTION,
        "live_distribution": live,
        "psi": round(psi, 4),
        "drift_status": _drift_status(psi) if live else "insufficient_data",
        "avg_confidence": stats["avg_confidence"],
        "appeal_rate": round(_appeal_rate(), 2),
        "sample_size": stats["total_decisions"],
        "disclaimer": (
            "Proxy signals only - no ground-truth labels exist for live traffic, "
            "so this is distributional drift + confidence + appeal-rate monitoring, "
            "not a live F1/accuracy measurement."
        ),
    }
