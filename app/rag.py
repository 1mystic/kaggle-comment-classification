"""RAG policy-grounding layer: retrieve relevant policy clauses for a flagged
comment and generate a human-readable, policy-cited explanation.

Retrieval is TF-IDF (scikit-learn, already a dependency) over a small local
policy corpus - deliberately not a neural embedding model, to stay inside
Render's 512MB memory budget alongside the classifier.

Explanation generation calls the Anthropic API (claude-opus-5) when
ANTHROPIC_API_KEY is set; otherwise falls back to a deterministic template
built from the retrieved clause + the lexicon signals engine.py already
computes, so the app works with zero secrets configured.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loads repo-root .env (if present) into os.environ. No-op if the file doesn't
# exist, so this is safe with zero configuration.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

POLICY_DIR = Path(__file__).resolve().parent / "policy"

_CLASS_TO_SECTION_HINT = {
    "Normal": "no action good faith discussion",
    "Offensive": "offensive language personal insults",
    "Hate Speech": "hate speech identity based attacks",
    "Severe/Violent": "severe violent explicit threats",
}


def _load_chunks():
    """Split each policy markdown file into (citation, text) chunks on '## ' headers."""
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        doc_title = path.stem.replace("_", " ").title()
        raw = path.read_text(encoding="utf-8")
        sections = re.split(r"\n(?=## )", raw)
        for section in sections:
            section = section.strip()
            if not section or not section.startswith("## "):
                continue
            heading, _, body = section.partition("\n")
            heading = heading.lstrip("# ").strip()
            citation = f"{doc_title} - {heading}"
            chunks.append({"citation": citation, "text": body.strip()})
    return chunks


_CHUNKS = _load_chunks()
_VECTORIZER = TfidfVectorizer(stop_words="english")
_CHUNK_MATRIX = _VECTORIZER.fit_transform([c["text"] for c in _CHUNKS]) if _CHUNKS else None


def retrieve(comment_text: str, predicted_label: str, top_k: int = 1):
    """Return the top_k most relevant policy chunks for this comment + predicted class."""
    if not _CHUNKS:
        return []
    query = comment_text + " " + _CLASS_TO_SECTION_HINT.get(predicted_label, "")
    query_vec = _VECTORIZER.transform([query])
    sims = cosine_similarity(query_vec, _CHUNK_MATRIX)[0]
    ranked = sorted(range(len(_CHUNKS)), key=lambda i: sims[i], reverse=True)
    return [_CHUNKS[i] for i in ranked[:top_k]]


def _template_explanation(comment_text: str, prediction: dict, clause: dict) -> str:
    label = prediction["label"]
    terms = prediction.get("violence_terms") or prediction.get("negative_terms") or []
    signal_note = f" Flagged terms: {', '.join(terms)}." if terms else ""
    return (
        f"This comment was classified as **{label}** "
        f"({prediction['confidence'] * 100:.1f}% confidence).{signal_note} "
        f"Per {clause['citation']}: {clause['text'][:280].rstrip()}..."
    )


_ANTHROPIC_CLIENT = None
_ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        _ANTHROPIC_CLIENT = anthropic.Anthropic()
        _ANTHROPIC_AVAILABLE = True
except ImportError:
    pass


def _llm_explanation(comment_text: str, prediction: dict, clause: dict) -> str | None:
    if not _ANTHROPIC_AVAILABLE:
        return None
    try:
        response = _ANTHROPIC_CLIENT.messages.create(
            model="claude-opus-5",
            max_tokens=300,
            system=(
                "You write short, neutral, policy-grounded explanations for a content "
                "moderation decision, in the style required by transparency regulations "
                "(e.g. India's IT Rules 2021 Rule 4(8)). Cite the given policy clause by "
                "name. Two to three sentences. No preamble, no markdown headers."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Comment: {comment_text!r}\n"
                    f"Predicted category: {prediction['label']} "
                    f"({prediction['confidence'] * 100:.1f}% confidence)\n"
                    f"Detected signals: violence_terms={prediction['violence_terms']}, "
                    f"negative_terms={prediction['negative_terms']}\n"
                    f"Relevant policy clause - {clause['citation']}:\n{clause['text']}\n\n"
                    "Write the explanation shown to the user who posted this comment."
                ),
            }],
        )
        if response.stop_reason == "refusal":
            return None
        return next((b.text for b in response.content if b.type == "text"), None)
    except Exception:
        return None


def explain(comment_text: str, prediction: dict) -> dict:
    """Retrieve the relevant policy clause and generate an explanation for this decision."""
    clauses = retrieve(comment_text, prediction["label"], top_k=1)
    if not clauses:
        return {
            "policy_citation": None,
            "policy_excerpt": None,
            "explanation": "No policy corpus available.",
            "source": "none",
        }
    clause = clauses[0]
    llm_text = _llm_explanation(comment_text, prediction, clause)
    if llm_text:
        return {
            "policy_citation": clause["citation"],
            "policy_excerpt": clause["text"][:400],
            "explanation": llm_text,
            "source": "llm",
        }
    return {
        "policy_citation": clause["citation"],
        "policy_excerpt": clause["text"][:400],
        "explanation": _template_explanation(comment_text, prediction, clause),
        "source": "template",
    }
