"""Illustrative reimplementation of the feature-engineering step described in the README.

NOT the original competition notebook: rewritten independently, after the fact,
to demonstrate the approach. Contains no competition data. Word lists below are
illustrative examples, not necessarily the exact lists used in the original work.
"""

import numpy as np
import pandas as pd

VIOLENCE_WORDS = frozenset({
    "kill", "killed", "killing", "shoot", "shooting", "shot", "death", "dead",
    "die", "murder", "attack", "burn", "gun", "weapon", "bomb", "destroy",
    "threat", "threaten",
})

POSITIVE_WORDS = frozenset({
    "good", "great", "excellent", "amazing", "love", "best", "nice", "thanks",
})

NEGATIVE_WORDS = frozenset({
    "bad", "terrible", "hate", "worst", "stupid", "trash", "idiot", "die", "kill",
})


def _word_count(text: str, vocab: frozenset) -> int:
    return sum(1 for w in text.split() if w in vocab)


def add_text_features(df: pd.DataFrame, text_column: str = "comment_clean") -> pd.DataFrame:
    """Add length, lexicon, and sentiment features derived from cleaned text."""
    df = df.copy()
    text = df[text_column].astype(str)

    df["char_count"] = text.str.len()
    df["word_count"] = text.str.split().str.len()
    df["lexical_div"] = text.apply(lambda t: len(set(t.split())) / (len(t.split()) + 1))

    df["violence_count"] = text.apply(lambda t: _word_count(t, VIOLENCE_WORDS))
    df["violence_ratio"] = df["violence_count"] / (df["word_count"] + 1)
    df["has_violence"] = (df["violence_count"] > 0).astype(int)
    df["short_violent"] = ((df["word_count"] <= 15) & (df["violence_count"] > 0)).astype(int)

    df["pos_count"] = text.apply(lambda t: _word_count(t, POSITIVE_WORDS))
    df["neg_count"] = text.apply(lambda t: _word_count(t, NEGATIVE_WORDS))
    df["sent_balance"] = df["pos_count"] - df["neg_count"]

    return df


def add_platform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between the platform's opaque if_1/if_2 signals.

    These were the single strongest predictors in the original work (see README):
    the platform's own moderation score already encodes signal that raw lexicon
    features can only partially reconstruct.
    """
    df = df.copy()
    df["if_prod"] = df["if_1"] * df["if_2"]
    df["if_sum"] = df["if_1"] + df["if_2"]
    df["if_ratio"] = df["if_1"] / (df["if_2"] + 1)
    df["zone_safe"] = (df["if_2"] <= 7).astype(int)
    df["if1_low"] = (df["if_1"] <= 1).astype(int)
    return df


def add_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag presence of identity-topic references, treating explicit 'none' distinctly from NaN."""
    df = df.copy()
    identity_cols = [c for c in ("race", "religion", "gender") if c in df.columns]
    has_identity = np.zeros(len(df), dtype=int)
    for col in identity_cols:
        has_identity |= df[col].notna().values & (df[col].astype(str).str.lower() != "none")
    df["has_identity"] = has_identity
    return df


def add_temporal_features(df: pd.DataFrame, date_column: str = "created_date") -> pd.DataFrame:
    """Cyclical hour-of-day encoding plus a weekend flag; weak but cheap signal."""
    df = df.copy()
    dt = pd.to_datetime(df[date_column])
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["is_weekend"] = dt.dt.weekday.isin([5, 6]).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full engineered-feature pipeline. Expects `comment_clean` already present."""
    df = add_text_features(df)
    df = add_platform_features(df)
    df = add_identity_features(df)
    df = add_temporal_features(df)
    return df
