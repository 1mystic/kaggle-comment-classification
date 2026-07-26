"""Feature-engineering step shared by the training pipeline and the Gradio app.

~50 hand-crafted numeric features (text complexity, violence/sentiment lexicon,
platform "zone" interactions, identity flags, temporal, engagement) plus the
NB-log-count-ratio transformer used by the NB-SVM base model. See README for
the rationale behind each feature group.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

VIOLENCE_WORDS = frozenset({
    "kill", "killed", "killing", "shoot", "shooting", "shot", "death", "dead",
    "die", "dying", "murder", "murdered", "murders", "attack", "attacking",
    "attacked", "burn", "burning", "burned", "fire", "firing", "guns", "gun",
    "weapon", "weapons", "bomb", "bombing", "destroy", "destroying", "violence",
    "violent", "rape", "hang", "hanging", "decapitate", "decapitated", "torture",
    "tortured", "stab", "stabbing", "threat", "threaten",
})

POSITIVE_WORDS = frozenset({
    "good", "great", "excellent", "amazing", "wonderful", "best", "love", "perfect",
    "nice", "fantastic", "awesome", "beautiful", "brilliant", "outstanding", "happy",
    "glad", "pleased", "thankful", "thanks", "appreciate", "agree", "helpful", "useful",
    "interesting", "impressive", "well", "better", "fine", "cool", "fun", "enjoy",
    "liked", "support", "positive", "fair", "kind", "respect", "right", "correct",
})

NEGATIVE_WORDS = frozenset({
    "bad", "terrible", "awful", "hate", "worst", "horrible", "disgusting", "poor",
    "pathetic", "stupid", "ugly", "wrong", "annoying", "boring", "dumb", "useless",
    "trash", "garbage", "waste", "sucks", "lame", "crap", "fail", "failed", "worse",
    "angry", "mad", "upset", "disappointed", "sad", "broken", "ruined", "ridiculous",
    "offensive", "toxic", "evil", "idiot", "fool", "shut", "die", "kill",
})

# Categorical columns fed through OrdinalEncoder in the final feature matrix.
CATEGORICAL_COLUMNS = ["race", "religion", "gender"]


class NBTransformer(BaseEstimator, TransformerMixin):
    """Scales TF-IDF features by the NB log-count ratio (Wang & Manning, 2012)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y=None):
        if y is None:
            self.r_ = np.ones(X.shape[1], dtype=np.float32)
            return self
        labels = np.unique(y)
        num_cols = X.shape[1]
        ratios = np.zeros((len(labels), num_cols), dtype=np.float64)
        for i, lbl in enumerate(labels):
            mask = y == lbl
            p_counts = np.asarray(X[mask].sum(axis=0)).flatten() + self.alpha
            n_counts = np.asarray(X[~mask].sum(axis=0)).flatten() + self.alpha
            p_freq = p_counts / p_counts.sum()
            n_freq = n_counts / n_counts.sum()
            ratios[i] = np.log(p_freq) - np.log(n_freq)
        self.r_ = np.abs(ratios).max(axis=0).astype(np.float32)
        return self

    def transform(self, X):
        return X.multiply(self.r_)


def _count_unique(text: str) -> int:
    return len(set(str(text).split()))


def _count_pos(text: str) -> int:
    return sum(1 for w in str(text).split() if w in POSITIVE_WORDS)


def _count_neg(text: str) -> int:
    return sum(1 for w in str(text).split() if w in NEGATIVE_WORDS)


def _count_caps(text: str) -> int:
    return sum(1 for w in str(text).split() if w.isupper() and len(w) > 1)


def _count_violence(text: str) -> int:
    return sum(1 for w in str(text).split() if w in VIOLENCE_WORDS)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full engineered-feature pipeline. Expects `comment_clean` already present
    (via preprocess.clean_text_column). Missing optional columns (upvote, if_1, race, ...)
    default to neutral values so the same code path works for both training data and a
    single live inference row from the Gradio app.
    """
    df = df.copy()
    raw = df["comment"].astype(str)
    cln = df.get("comment_clean", raw.str.lower())
    word_count = cln.str.split().str.len().fillna(0).astype(int)

    if1 = df.get("if_1", pd.Series(0, index=df.index)).fillna(0)
    if2 = df.get("if_2", pd.Series(0, index=df.index)).fillna(0)

    df["char_count"] = cln.str.len().astype(np.float32)
    df["word_count"] = word_count.astype(np.float32)
    df["unique_words"] = cln.apply(_count_unique).astype(np.float32)
    df["lexical_div"] = (df["unique_words"] / (word_count + 1)).astype(np.float32)
    df["avg_word_len"] = (df["char_count"] / (word_count + 1)).astype(np.float32)

    df["caps_count"] = raw.str.count(r"[A-Z]").astype(np.float32)
    df["caps_ratio"] = (df["caps_count"] / (df["char_count"] + 1)).astype(np.float32)
    df["exclaim"] = raw.str.count("!").astype(np.float32)
    df["question"] = raw.str.count(r"\?").astype(np.float32)
    df["punct_count"] = raw.str.count(r"[^\w\s]").astype(np.float32)
    df["sent_count"] = raw.str.count(r"[.!?]+").clip(lower=1).astype(np.float32)
    df["avg_sent_len"] = (word_count / (df["sent_count"] + 1)).astype(np.float32)
    df["all_caps_words"] = raw.apply(_count_caps).astype(np.float32)

    df["pos_count"] = cln.apply(_count_pos).astype(np.float32)
    df["neg_count"] = cln.apply(_count_neg).astype(np.float32)
    df["sent_balance"] = (df["pos_count"] - df["neg_count"]).astype(np.float32)

    df["violence_count"] = cln.apply(_count_violence).astype(np.float32)
    df["has_violence"] = (df["violence_count"] > 0).astype(np.int8)
    df["violence_ratio"] = (df["violence_count"] / (word_count + 1)).astype(np.float32)
    df["violence_score"] = (df["violence_count"] * 2.0 + df["violence_ratio"] * 10.0).astype(np.float32)
    df["if1_x_violence"] = (if1 * df["violence_count"]).astype(np.float32)

    df["upvote"] = df.get("upvote", pd.Series(0, index=df.index)).fillna(0)
    df["downvote"] = df.get("downvote", pd.Series(0, index=df.index)).fillna(0)

    df["total_votes"] = (df["upvote"] + df["downvote"]).astype(np.float32)
    df["vote_ratio"] = (df["upvote"] / (df["total_votes"] + 1)).astype(np.float32)
    df["zero_downvote"] = (df["downvote"] == 0).astype(np.int8)
    df["controversy"] = np.minimum(df["upvote"], df["downvote"]).astype(np.float32) * 2

    emo_cols = ["emoticon_1", "emoticon_2", "emoticon_3"]
    for ec in emo_cols:
        if ec not in df.columns:
            df[ec] = 0
    df[emo_cols] = df[emo_cols].fillna(0)
    df["total_emo"] = df[emo_cols].sum(axis=1).astype(np.float32)
    df["has_emo"] = (df["total_emo"] > 0).astype(np.int8)

    parsed_dt = pd.to_datetime(df.get("created_date", pd.Series(None, index=df.index)), errors="coerce")
    hour = parsed_dt.dt.hour.fillna(12).astype(int)
    day_of_week = parsed_dt.dt.dayofweek.fillna(3).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7).astype(np.float32)
    df["is_weekend"] = (day_of_week >= 5).astype(np.int8)

    race_col = df.get("race", pd.Series("_", index=df.index)).fillna("_").str.lower()
    religion_col = df.get("religion", pd.Series("_", index=df.index)).fillna("_").str.lower()
    gender_col = df.get("gender", pd.Series("_", index=df.index)).fillna("_").str.lower()

    has_race = (race_col != "_") & (race_col != "none")
    has_religion = (religion_col != "_") & (religion_col != "none")
    has_gender = (gender_col != "_") & (gender_col != "none")

    df["has_identity"] = (has_race | has_religion | has_gender).astype(np.int8)
    df["identity_count"] = (has_race.astype(int) + has_religion.astype(int) + has_gender.astype(int)).astype(np.int8)
    df["disability_flag"] = (
        df.get("disability", pd.Series(False, index=df.index))
        .map({True: 1, False: 0, "True": 1, "False": 0})
        .fillna(0)
    ).astype(np.int8)

    df["if_prod"] = (if1 * if2).astype(np.float32)
    df["if_ratio"] = (if1 / (if2 + 1)).astype(np.float32)
    df["if_sum"] = (if1 + if2).astype(np.float32)

    df["zone_safe"] = (if2 <= 7).astype(np.int8)
    df["zone_c1_trigger"] = ((if2 >= 8) & (if1.isin([4, 6, 10]))).astype(np.int8)
    df["zone_c23"] = ((if1 == 0) & (if2 >= 8)).astype(np.int8)
    df["if1_low"] = (if1 <= 1).astype(np.int8)
    df["if1_nonzero"] = (if1 > 0).astype(np.int8)

    df["golden_c1"] = ((if2 >= 8) & df["has_identity"].astype(bool)).astype(np.int8)
    df["golden_c2"] = ((if2 >= 8) & ~df["has_identity"].astype(bool)).astype(np.int8)

    df["is_short"] = (word_count <= 15).astype(np.int8)
    df["short_violent"] = (df["is_short"] * df["has_violence"]).astype(np.int8)

    df["danger_zone"] = ((df["zone_c23"] == 1) & (df["has_identity"] == 0)).astype(np.int8)
    df["violence_in_dz"] = (df["violence_count"] * df["danger_zone"]).astype(np.float32)
    df["c3_signal"] = (
        df["is_short"] * df["zero_downvote"] * (df["has_identity"] == 0) * df["zone_c23"]
    ).astype(np.int8)

    numeric_dtype_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_dtype_cols] = df[numeric_dtype_cols].fillna(0)

    return df


def numeric_feature_columns(df: pd.DataFrame) -> list:
    """All engineered numeric columns eligible for the model matrix (excludes raw/text/id columns)."""
    exclude = {
        "comment", "comment_clean", "post_id", "created_date", "label",
        *CATEGORICAL_COLUMNS, "disability",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in exclude]
