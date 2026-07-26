"""Text-cleaning step shared by the training pipeline and the Gradio app.

Order matters: leet-speak decoding has to happen before non-alphabetic
characters are stripped, otherwise the substituted digits/symbols are
destroyed along with the noise (see README).
"""

import re

import pandas as pd

LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s", "!": "i",
})

ABBREV_MAP = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "gonna": "going to",
    "wanna": "want to",
    "kys": "kill yourself",
    "kms": "kill myself",
    "wtf": "what the fuck",
    "stfu": "shut the fuck up",
    "idk": "i do not know",
    "ngl": "not gonna lie",
}


def expand_abbreviations(text: str) -> str:
    """Replace whole-word abbreviations using ABBREV_MAP."""
    words = text.split()
    return " ".join(ABBREV_MAP.get(w, w) for w in words)


def clean_text(text: str) -> str:
    """Clean a single comment string (scalar equivalent of clean_text_column)."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\w+\.(com|org|net|co|us)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(LEET_MAP)
    text = expand_abbreviations(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "empty_comment"


def clean_text_column(df: pd.DataFrame, column: str = "comment") -> pd.DataFrame:
    """Vectorized clean of a DataFrame's comment column, adding `{column}_clean`."""
    df = df.copy()
    text = df[column].fillna("").astype(str).str.lower()
    text = text.str.replace(r"http\S+|www\.\S+", " ", regex=True)
    text = text.str.replace(r"\w+\.(com|org|net|co|us)", " ", regex=True)
    text = text.str.replace(r"<[^>]+>", " ", regex=True)
    text = text.apply(lambda s: s.translate(LEET_MAP))
    text = text.apply(expand_abbreviations)
    text = text.str.replace(r"[^a-z\s]", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    text = text.replace("", "empty_comment").fillna("missing_comment")
    df[f"{column}_clean"] = text
    return df
