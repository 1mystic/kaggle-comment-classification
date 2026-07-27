"""Inference engine: text cleaning, feature engineering, and model loading/prediction.

Lifted directly from the working Gradio app (app/app.py) — same logic, same
artifact-loading contract, just without any Gradio dependency so it can be
imported by the FastAPI server (main.py) and, going forward, anything else.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. Preprocessing & Feature Engineering
# ==============================================================================

LEET_MAP = str.maketrans({
    '0': 'o', '1': 'i', '3': 'e', '4': 'a',
    '5': 's', '7': 't', '@': 'a', '$': 's', '!': 'i'
})

ABBREV_MAP = {
    'u': 'you', 'r': 'are', 'ur': 'your', 'gonna': 'going to', 'wanna': 'want to',
    'kys': 'kill yourself', 'kms': 'kill myself', 'wtf': 'what the fuck',
    'stfu': 'shut the fuck up', 'idk': 'i do not know', 'ngl': 'not gonna lie',
}

POSITIVE_WORDS = frozenset({
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'perfect',
    'nice', 'fantastic', 'awesome', 'beautiful', 'brilliant', 'outstanding', 'happy',
    'glad', 'pleased', 'thankful', 'thanks', 'appreciate', 'agree', 'helpful', 'useful',
    'interesting', 'impressive', 'well', 'better', 'fine', 'cool', 'fun', 'enjoy',
    'liked', 'support', 'positive', 'fair', 'kind', 'respect', 'right', 'correct'
})

NEGATIVE_WORDS = frozenset({
    'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'poor',
    'pathetic', 'stupid', 'ugly', 'wrong', 'annoying', 'boring', 'dumb', 'useless',
    'trash', 'garbage', 'waste', 'sucks', 'lame', 'crap', 'fail', 'failed', 'worse',
    'angry', 'mad', 'upset', 'disappointed', 'sad', 'broken', 'ruined', 'ridiculous',
    'offensive', 'toxic', 'evil', 'idiot', 'fool', 'shut', 'die', 'kill'
})

VIOLENCE_WORDS = frozenset({
    'kill', 'killed', 'killing', 'shoot', 'shooting', 'shot', 'death', 'dead', 'die',
    'dying', 'murder', 'murdered', 'murders', 'attack', 'attacking', 'attacked',
    'burn', 'burning', 'burned', 'fire', 'firing', 'guns', 'gun', 'weapon', 'weapons',
    'bomb', 'bombing', 'destroy', 'destroying', 'violence', 'violent', 'rape', 'hang',
    'hanging', 'decapitate', 'decapitated', 'torture', 'tortured', 'stab', 'stabbing',
    'threat', 'threaten'
})


def expand_abbrevs(sentence):
    words = sentence.split()
    return ' '.join([ABBREV_MAP.get(w, w) for w in words])


def clean_text_column(df):
    df = df.copy()
    text = df['comment'].fillna('').astype(str).str.lower()
    text = text.str.replace(r'http\S+|www\.\S+', ' ', regex=True)
    text = text.str.replace(r'\w+\.(com|org|net|co|us)', ' ', regex=True)
    text = text.str.replace(r'<[^>]+>', ' ', regex=True)
    text = text.apply(lambda s: s.translate(LEET_MAP))
    text = text.apply(expand_abbrevs)
    text = text.str.replace(r'[^a-z\s]', ' ', regex=True)
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    text = text.replace('', 'empty_comment').fillna('missing_comment')
    df['comment_clean'] = text
    return df


class NBTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y=None):
        if y is None:
            self.r_ = np.ones(X.shape[1], dtype=np.float32)
            return self
        labels = np.unique(y)
        num_cols = X.shape[1]
        ratios = np.zeros((len(labels), num_cols), dtype=np.float64)
        for i, lbl in enumerate(labels):
            mask = (y == lbl)
            p_counts = np.asarray(X[mask].sum(axis=0)).flatten() + self.alpha
            n_counts = np.asarray(X[~mask].sum(axis=0)).flatten() + self.alpha
            p_freq = p_counts / p_counts.sum()
            n_freq = n_counts / n_counts.sum()
            ratios[i] = np.log(p_freq) - np.log(n_freq)
        self.r_ = np.abs(ratios).max(axis=0).astype(np.float32)
        return self

    def transform(self, X):
        return X.multiply(self.r_)


# The model artifact was pickled from a Kaggle notebook where NBTransformer lived
# in __main__ (notebook cells run as __main__). Unpickling looks up the class by
# that module path regardless of how this module is imported elsewhere, so patch
# it onto whatever __main__ actually is (a script, uvicorn, a REPL, ...).
sys.modules['__main__'].NBTransformer = NBTransformer


def count_unique(text):
    return len(set(str(text).split()))


def count_pos(text):
    return sum(1 for w in str(text).split() if w in POSITIVE_WORDS)


def count_neg(text):
    return sum(1 for w in str(text).split() if w in NEGATIVE_WORDS)


def count_caps(text):
    return sum(1 for w in str(text).split() if w.isupper() and len(w) > 1)


def count_violence(text):
    return sum(1 for w in str(text).split() if w in VIOLENCE_WORDS)


def build_features(df):
    df = df.copy()
    raw = df['comment'].astype(str)
    cln = df.get('comment_clean', raw.str.lower())
    word_count = cln.str.split().str.len().fillna(0).astype(int)

    if1 = df.get('if_1', pd.Series(0, index=df.index)).fillna(0)
    if2 = df.get('if_2', pd.Series(0, index=df.index)).fillna(0)

    df['char_count'] = cln.str.len().astype(np.float32)
    df['word_count'] = word_count.astype(np.float32)
    df['unique_words'] = cln.apply(count_unique).astype(np.float32)
    df['lexical_div'] = (df['unique_words'] / (word_count + 1)).astype(np.float32)
    df['avg_word_len'] = (df['char_count'] / (word_count + 1)).astype(np.float32)

    df['caps_count'] = raw.str.count(r'[A-Z]').astype(np.float32)
    df['caps_ratio'] = (df['caps_count'] / (df['char_count'] + 1)).astype(np.float32)
    df['exclaim'] = raw.str.count('!').astype(np.float32)
    df['question'] = raw.str.count(r'\?').astype(np.float32)
    df['punct_count'] = raw.str.count(r'[^\w\s]').astype(np.float32)
    df['sent_count'] = raw.str.count(r'[.!?]+').clip(lower=1).astype(np.float32)
    df['avg_sent_len'] = (word_count / (df['sent_count'] + 1)).astype(np.float32)
    df['all_caps_words'] = raw.apply(count_caps).astype(np.float32)

    df['pos_count'] = cln.apply(count_pos).astype(np.float32)
    df['neg_count'] = cln.apply(count_neg).astype(np.float32)
    df['sent_balance'] = (df['pos_count'] - df['neg_count']).astype(np.float32)

    df['violence_count'] = cln.apply(count_violence).astype(np.float32)
    df['has_violence'] = (df['violence_count'] > 0).astype(np.int8)
    df['violence_ratio'] = (df['violence_count'] / (word_count + 1)).astype(np.float32)
    df['violence_score'] = (df['violence_count'] * 2.0 + df['violence_ratio'] * 10.0).astype(np.float32)
    df['if1_x_violence'] = (if1 * df['violence_count']).astype(np.float32)

    df['upvote'] = df.get('upvote', pd.Series(0, index=df.index)).fillna(0)
    df['downvote'] = df.get('downvote', pd.Series(0, index=df.index)).fillna(0)

    df['total_votes'] = (df['upvote'] + df['downvote']).astype(np.float32)
    df['vote_ratio'] = (df['upvote'] / (df['total_votes'] + 1)).astype(np.float32)
    df['zero_downvote'] = (df['downvote'] == 0).astype(np.int8)
    df['controversy'] = np.minimum(df['upvote'], df['downvote']).astype(np.float32) * 2

    emo_cols = ['emoticon_1', 'emoticon_2', 'emoticon_3']
    for ec in emo_cols:
        if ec not in df.columns:
            df[ec] = 0
    df[emo_cols] = df[emo_cols].fillna(0)
    df['total_emo'] = df[emo_cols].sum(axis=1).astype(np.float32)
    df['has_emo'] = (df['total_emo'] > 0).astype(np.int8)

    parsed_dt = pd.to_datetime(df.get('created_date', pd.Series(None, index=df.index)), errors='coerce')
    hour = parsed_dt.dt.hour.fillna(12).astype(int)
    day_of_week = parsed_dt.dt.dayofweek.fillna(3).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7).astype(np.float32)
    df['is_weekend'] = (day_of_week >= 5).astype(np.int8)

    race_col = df.get('race', pd.Series('_', index=df.index)).fillna('_').str.lower()
    religion_col = df.get('religion', pd.Series('_', index=df.index)).fillna('_').str.lower()
    gender_col = df.get('gender', pd.Series('_', index=df.index)).fillna('_').str.lower()

    has_race = (race_col != '_') & (race_col != 'none')
    has_religion = (religion_col != '_') & (religion_col != 'none')
    has_gender = (gender_col != '_') & (gender_col != 'none')

    df['has_identity'] = (has_race | has_religion | has_gender).astype(np.int8)
    df['identity_count'] = (has_race.astype(int) + has_religion.astype(int) + has_gender.astype(int)).astype(np.int8)
    df['disability_flag'] = (
        df.get('disability', pd.Series(False, index=df.index)).map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0)
    ).astype(np.int8)

    df['if_prod'] = (if1 * if2).astype(np.float32)
    df['if_ratio'] = (if1 / (if2 + 1)).astype(np.float32)
    df['if_sum'] = (if1 + if2).astype(np.float32)

    df['zone_safe'] = (if2 <= 7).astype(np.int8)
    df['zone_c1_trigger'] = ((if2 >= 8) & (if1.isin([4, 6, 10]))).astype(np.int8)
    df['zone_c23'] = ((if1 == 0) & (if2 >= 8)).astype(np.int8)
    df['if1_low'] = (if1 <= 1).astype(np.int8)
    df['if1_nonzero'] = (if1 > 0).astype(np.int8)

    df['golden_c1'] = ((if2 >= 8) & df['has_identity'].astype(bool)).astype(np.int8)
    df['golden_c2'] = ((if2 >= 8) & ~df['has_identity'].astype(bool)).astype(np.int8)

    df['is_short'] = (word_count <= 15).astype(np.int8)
    df['short_violent'] = (df['is_short'] * df['has_violence']).astype(np.int8)

    df['danger_zone'] = ((df['zone_c23'] == 1) & (df['has_identity'] == 0)).astype(np.int8)
    df['violence_in_dz'] = (df['violence_count'] * df['danger_zone']).astype(np.float32)
    df['c3_signal'] = (df['is_short'] * df['zero_downvote'] * (df['has_identity'] == 0) * df['zone_c23']).astype(np.int8)

    numeric_dtype_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_dtype_cols] = df[numeric_dtype_cols].fillna(0)

    return df


# ==============================================================================
# 2. Pipeline Artifact Loader
# ==============================================================================

MODEL_PATHS = [
    'comment_classifier_pipeline.joblib',
    'app/comment_classifier_pipeline.joblib',
    '../comment_classifier_pipeline.joblib',
    '/kaggle/working/comment_classifier_pipeline.joblib',
]

# See app/deployment_training_fast.ipynb, section 9 "Export Deployment Artifact".
# Each *_models entry is a list of N_FOLDS fitted fold copies of that base
# learner, bagged (averaged) at inference time.
REQUIRED_ARTIFACT_KEYS = {
    'tfidf_word', 'tfidf_char', 'tfidf_phrase', 'ordinal_enc',
    'numeric_col_names', 'cat_col_names', 'meta_model', 'thresholds',
    'lgb_models', 'lr_word_models', 'lr_char_models', 'nbsvm_models',
    'lgb_bin_models', 'lgb_c23_models', 'svc_models',
}

CLASS_NAMES = {0: 'Normal', 1: 'Offensive', 2: 'Hate Speech', 3: 'Severe/Violent'}

MODEL_ARTIFACT = None
MODEL_STATUS_TEXT = ""


def _load_artifact():
    global MODEL_ARTIFACT, MODEL_STATUS_TEXT
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                candidate = joblib.load(path)
                missing = REQUIRED_ARTIFACT_KEYS - set(candidate.keys())
                if missing:
                    print(f"Skipping {path}: artifact missing keys {sorted(missing)}")
                    continue
                MODEL_ARTIFACT = candidate
                n_folds = MODEL_ARTIFACT.get('n_folds', len(MODEL_ARTIFACT['lgb_models']))
                MODEL_STATUS_TEXT = f"Full Stacked Ensemble Active ({os.path.basename(path)}, {n_folds}-fold bagged)"
                print(MODEL_STATUS_TEXT)
                return
            except Exception as e:
                print(f"Error loading {path}: {e}")
    MODEL_STATUS_TEXT = "Demonstration Preview Mode (Upload 'comment_classifier_pipeline.joblib' to activate full ensemble)"
    print(MODEL_STATUS_TEXT)


_load_artifact()


def _bag_predict_proba(models, X):
    """Average predict_proba across a list of fold-fitted copies of the same base learner."""
    total = None
    for m in models:
        p = m.predict_proba(X)
        total = p if total is None else total + p
    return total / len(models)


def is_model_active() -> bool:
    return MODEL_ARTIFACT is not None


# ==============================================================================
# 3. Prediction
# ==============================================================================

def predict(
    comment_text: str,
    if_1: float = 0,
    if_2: float = 5,
    upvotes: float = 0,
    downvotes: float = 0,
    race: str = "none",
    religion: str = "none",
    gender: str = "none",
) -> dict:
    """Run the full pipeline on one comment and return a structured result dict."""
    input_df = pd.DataFrame([{
        'comment': comment_text,
        'if_1': float(if_1),
        'if_2': float(if_2),
        'upvote': float(upvotes),
        'downvote': float(downvotes),
        'emoticon_1': 0, 'emoticon_2': 0, 'emoticon_3': 0,
        'created_date': pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'race': race,
        'religion': religion,
        'gender': gender,
        'disability': False,
    }])

    df_clean = clean_text_column(input_df)
    df_eng = build_features(df_clean)
    clean_str = df_eng['comment_clean'].iloc[0]

    if MODEL_ARTIFACT is not None:
        w_vec = MODEL_ARTIFACT['tfidf_word'].transform([clean_str])
        c_vec = MODEL_ARTIFACT['tfidf_char'].transform([clean_str])
        p_vec = MODEL_ARTIFACT['tfidf_phrase'].transform([clean_str])

        num_feats = df_eng[MODEL_ARTIFACT['numeric_col_names']].values.astype(np.float32)
        cat_feats = MODEL_ARTIFACT['ordinal_enc'].transform(
            df_eng[MODEL_ARTIFACT['cat_col_names']].fillna('missing').astype(str)
        ).astype(np.int32)

        X_full = hstack([w_vec, c_vec, p_vec, csr_matrix(num_feats), csr_matrix(cat_feats)]).tocsr()
        clean_text_input = [clean_str]

        p_lgb = _bag_predict_proba(MODEL_ARTIFACT['lgb_models'], X_full)
        p_lr_w = _bag_predict_proba(MODEL_ARTIFACT['lr_word_models'], clean_text_input)
        p_lr_c = _bag_predict_proba(MODEL_ARTIFACT['lr_char_models'], clean_text_input)
        p_nbsvm = _bag_predict_proba(MODEL_ARTIFACT['nbsvm_models'], clean_text_input)
        p_lgb_bin = _bag_predict_proba(MODEL_ARTIFACT['lgb_bin_models'], X_full)
        p_svc = _bag_predict_proba(MODEL_ARTIFACT['svc_models'], clean_text_input)
        p_c23 = _bag_predict_proba(MODEL_ARTIFACT['lgb_c23_models'], X_full)

        stacked = np.hstack([p_lgb, p_lr_w, p_lr_c, p_nbsvm, p_lgb_bin, p_svc, p_c23]).astype(np.float32)
        meta_probs = MODEL_ARTIFACT['meta_model'].predict_proba(stacked)[0]

        mults = MODEL_ARTIFACT.get('thresholds', [1.032, 0.951, 0.85, 1.25])
        for i in range(4):
            meta_probs[i] *= mults[i]
        probs = meta_probs / meta_probs.sum()
    else:
        v_count = df_eng['violence_count'].iloc[0]
        n_count = df_eng['neg_count'].iloc[0]
        p_count = df_eng['pos_count'].iloc[0]
        has_id = df_eng['has_identity'].iloc[0]

        if v_count > 0 or 'kill' in clean_str or 'shoot' in clean_str or 'die' in clean_str:
            probs = np.array([0.05, 0.15, 0.20, 0.60])
        elif n_count > 1 or 'hate' in clean_str or 'stupid' in clean_str:
            probs = np.array([0.10, 0.50, 0.30, 0.10]) if not has_id else np.array([0.05, 0.25, 0.60, 0.10])
        elif p_count > 0 or 'good' in clean_str or 'thanks' in clean_str or 'great' in clean_str:
            probs = np.array([0.85, 0.10, 0.03, 0.02])
        else:
            probs = np.array([0.70, 0.18, 0.08, 0.04])

    pred_class_id = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_class_id]
    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}

    words = clean_str.split()
    violence_terms = sorted(set(w for w in words if w in VIOLENCE_WORDS))
    negative_terms = sorted(set(w for w in words if w in NEGATIVE_WORDS))
    positive_terms = sorted(set(w for w in words if w in POSITIVE_WORDS))

    sentiment_score = float(df_eng['sent_balance'].iloc[0])
    caps_ratio = float(df_eng['caps_ratio'].iloc[0]) * 100
    risk_level = (
        "high" if pred_class_id in (2, 3) else
        "moderate" if pred_class_id == 1 else
        "low"
    )

    return {
        "comment_text": comment_text,
        "label": pred_label,
        "label_id": pred_class_id,
        "confidence": float(probs[pred_class_id]),
        "confidences": confidences,
        "risk_level": risk_level,
        "sentiment_score": sentiment_score,
        "caps_ratio": caps_ratio,
        "violence_terms": violence_terms,
        "negative_terms": negative_terms,
        "positive_terms": positive_terms,
        "model_status": MODEL_STATUS_TEXT,
        "model_active": MODEL_ARTIFACT is not None,
    }
