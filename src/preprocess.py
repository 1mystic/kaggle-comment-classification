"""Illustrative reimplementation of the text-cleaning step described in the README.

NOT the original competition notebook: rewritten independently, after the fact,
to demonstrate the approach. Contains no competition data.
"""

import re

import pandas as pd

# Leet-speak substitutions for common filter-evasion patterns.
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s", "!": "i",
})

# TODO: fill in with the actual abbreviation set used, if you want this to be
# a closer reference. Left minimal here since the exact list wasn't specified.
ABBREV_MAP = {
    "u": "you",
    "r": "are",
    "ur": "your",
}


def expand_abbreviations(text: str) -> str:
    """Replace whole-word abbreviations using ABBREV_MAP."""
    words = text.split()
    return " ".join(ABBREV_MAP.get(w, w) for w in words)


def clean_text(text: str) -> str:
    """Clean a single comment string.

    Order matters: leet-decode before stripping non-alphabetic characters,
    otherwise the substituted digits/symbols are lost along with the noise.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(LEET_MAP)
    text = expand_abbreviations(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "empty_comment"


def clean_text_column(df: pd.DataFrame, column: str = "comment") -> pd.DataFrame:
    """Apply clean_text to a DataFrame column, returning a copy with `{column}_clean` added."""
    df = df.copy()
    df[f"{column}_clean"] = df[column].fillna("").astype(str).apply(clean_text)
    return df
