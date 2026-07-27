"""Agent escalation workflow: routes a classified comment to an action state.

A small LangGraph state graph - classify (done upstream by engine.py) -> assess
risk -> route to {auto_action | human_review | auto_clear}. Kept intentionally
small: this is the concrete "agent orchestration" artifact, not a place to
over-engineer a multi-agent system for a single decision.
"""

import re
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, END

import audit
import engine
import rag

AUTO_ACTION_CONFIDENCE_THRESHOLD = 0.60

# --- Identity-targeting safety net -------------------------------------------
# The underlying classifier's strongest feature is the platform's own opaque
# if_1/if_2 signal (see root README) - when that's left at neutral defaults
# (as it will be for arbitrary raw text with no upstream platform score) and
# the comment has no leetspeak or hand-built-lexicon matches, the model can
# under-detect identity-based exclusionary rhetoric. This is a known limitation
# of the trained model itself, not fixable by re-code here (that needs
# retraining). This heuristic is a deliberately simple, explainable guardrail
# sitting ON TOP of the model: if the comment names an identity/nationality
# group AND uses exclusionary language, escalate at least to human review even
# when the model alone would clear it. Not exhaustive - a watchlist, not NLP.
IDENTITY_TERMS = frozenset({
    "indians", "indian", "muslims", "muslim", "christians", "christian",
    "hindus", "hindu", "jews", "jewish", "sikhs", "sikh", "buddhists",
    "immigrants", "immigrant", "foreigners", "foreigner", "refugees", "refugee",
    "blacks", "whites", "asians", "mexicans", "chinese", "arabs", "africans",
    "pakistanis", "americans", "gays", "lesbians", "transgender", "disabled",
})

EXCLUSION_PATTERNS = [
    r"should (be )?(kicked out|deported|banned|removed|expelled)",
    r"get out of (this|our|my) (country|nation|land)",
    r"go back to (your|their) (country|homeland)",
    r"(don'?t|do not) belong here",
    r"no place for (them|him|her|these people)",
    r"have no (right|place) (to be|here)",
]


def _identity_exclusion_flag(comment_text: str) -> bool:
    text = comment_text.lower()
    words = set(re.findall(r"[a-z]+", text))
    has_identity = bool(words & IDENTITY_TERMS)
    has_exclusion = any(re.search(p, text) for p in EXCLUSION_PATTERNS)
    return has_identity and has_exclusion


class ModerationState(TypedDict):
    comment_text: str
    prediction: dict
    explanation: dict
    agent_status: str
    identity_exclusion_flag: bool
    override_reason: str | None


def assess_risk(state: ModerationState) -> ModerationState:
    """Compute the identity-exclusion safety-net flag alongside engine.py's risk fields."""
    return {**state, "identity_exclusion_flag": _identity_exclusion_flag(state["comment_text"])}


def _route_after_assess(state: ModerationState) -> Literal["auto_action", "human_review", "auto_clear"]:
    prediction = state["prediction"]
    label_id = prediction["label_id"]
    confidence = prediction["confidence"]

    if label_id in (2, 3) and confidence >= AUTO_ACTION_CONFIDENCE_THRESHOLD:
        return "auto_action"
    if label_id in (1, 2, 3):
        return "human_review"
    if state["identity_exclusion_flag"]:
        return "human_review"
    return "auto_clear"


def _auto_action_node(state: ModerationState) -> ModerationState:
    return {**state, "agent_status": "auto_action"}


def _human_review_node(state: ModerationState) -> ModerationState:
    override_reason = None
    if state["prediction"]["label_id"] == 0 and state["identity_exclusion_flag"]:
        override_reason = (
            "Escalated by the identity-targeting safety net: the model classified this as Normal, "
            "but the comment names an identity/nationality group alongside exclusionary language."
        )
    return {**state, "agent_status": "human_review", "override_reason": override_reason}


def _auto_clear_node(state: ModerationState) -> ModerationState:
    return {**state, "agent_status": "auto_clear"}


def _build_graph():
    graph = StateGraph(ModerationState)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("auto_action", _auto_action_node)
    graph.add_node("human_review", _human_review_node)
    graph.add_node("auto_clear", _auto_clear_node)

    graph.set_entry_point("assess_risk")
    graph.add_conditional_edges("assess_risk", _route_after_assess, {
        "auto_action": "auto_action",
        "human_review": "human_review",
        "auto_clear": "auto_clear",
    })
    graph.add_edge("auto_action", END)
    graph.add_edge("human_review", END)
    graph.add_edge("auto_clear", END)
    return graph.compile()


_GRAPH = _build_graph()

AGENT_STATUS_LABELS = {
    "auto_action": "Auto-actioned",
    "human_review": "Routed for human review",
    "auto_clear": "Cleared automatically",
    "appeal_reviewed": "Appeal reviewed",
}


def moderate(comment_text: str, prediction: dict, explanation: dict) -> dict:
    """Run the agent graph and persist the resulting decision to the audit log."""
    result = _GRAPH.invoke({
        "comment_text": comment_text,
        "prediction": prediction,
        "explanation": explanation,
        "agent_status": "",
        "identity_exclusion_flag": False,
        "override_reason": None,
    })
    override_reason = result.get("override_reason")
    logged_explanation = explanation.get("explanation")
    if override_reason:
        logged_explanation = f"{logged_explanation}\n\n⚠ {override_reason}"

    decision_id = audit.log_decision(
        comment_text=comment_text,
        label=prediction["label"],
        confidence=prediction["confidence"],
        risk_level=prediction["risk_level"],
        policy_citation=explanation.get("policy_citation"),
        explanation=logged_explanation,
        agent_status=result["agent_status"],
    )
    return {
        "decision_id": decision_id,
        "agent_status": result["agent_status"],
        "agent_status_label": AGENT_STATUS_LABELS[result["agent_status"]],
        "override_reason": override_reason,
    }


def appeal(decision_id: int, appeal_reason: str) -> dict:
    """Re-run classification + RAG on the original comment with appeal context, log a new decision."""
    original = audit.get_decision(decision_id)
    if original is None:
        raise ValueError(f"No decision found with id={decision_id}")

    prediction = engine.predict(original["comment_text"])
    explanation = rag.explain(original["comment_text"], prediction)
    explanation["explanation"] = (
        f"[Appeal reviewed - reason given: \"{appeal_reason}\"] " + explanation["explanation"]
    )

    new_id = audit.log_decision(
        comment_text=original["comment_text"],
        label=prediction["label"],
        confidence=prediction["confidence"],
        risk_level=prediction["risk_level"],
        policy_citation=explanation.get("policy_citation"),
        explanation=explanation.get("explanation"),
        agent_status="appeal_reviewed",
        appeal_of=decision_id,
    )
    return {
        "decision_id": new_id,
        "appeal_of": decision_id,
        "agent_status": "appeal_reviewed",
        "agent_status_label": AGENT_STATUS_LABELS["appeal_reviewed"],
        "prediction": prediction,
        "explanation": explanation,
    }
