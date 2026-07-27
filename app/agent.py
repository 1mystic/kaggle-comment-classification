"""Agent escalation workflow: routes a classified comment to an action state.

A small LangGraph state graph — classify (done upstream by engine.py) -> assess
risk -> route to {auto_action | human_review | auto_clear}. Kept intentionally
small: this is the concrete "agent orchestration" artifact, not a place to
over-engineer a multi-agent system for a single decision.
"""

from typing import Literal, TypedDict

from langgraph.graph import StateGraph, END

import audit
import engine
import rag

AUTO_ACTION_CONFIDENCE_THRESHOLD = 0.60


class ModerationState(TypedDict):
    comment_text: str
    prediction: dict
    explanation: dict
    agent_status: str


def assess_risk(state: ModerationState) -> ModerationState:
    """Pass-through node: risk fields already computed by engine.predict()."""
    return state


def _route_after_assess(state: ModerationState) -> Literal["auto_action", "human_review", "auto_clear"]:
    prediction = state["prediction"]
    label_id = prediction["label_id"]
    confidence = prediction["confidence"]

    if label_id in (2, 3) and confidence >= AUTO_ACTION_CONFIDENCE_THRESHOLD:
        return "auto_action"
    if label_id in (1, 2, 3):
        return "human_review"
    return "auto_clear"


def _auto_action_node(state: ModerationState) -> ModerationState:
    return {**state, "agent_status": "auto_action"}


def _human_review_node(state: ModerationState) -> ModerationState:
    return {**state, "agent_status": "human_review"}


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
    })
    decision_id = audit.log_decision(
        comment_text=comment_text,
        label=prediction["label"],
        confidence=prediction["confidence"],
        risk_level=prediction["risk_level"],
        policy_citation=explanation.get("policy_citation"),
        explanation=explanation.get("explanation"),
        agent_status=result["agent_status"],
    )
    return {
        "decision_id": decision_id,
        "agent_status": result["agent_status"],
        "agent_status_label": AGENT_STATUS_LABELS[result["agent_status"]],
    }


def appeal(decision_id: int, appeal_reason: str) -> dict:
    """Re-run classification + RAG on the original comment with appeal context, log a new decision."""
    original = audit.get_decision(decision_id)
    if original is None:
        raise ValueError(f"No decision found with id={decision_id}")

    prediction = engine.predict(original["comment_text"])
    explanation = rag.explain(original["comment_text"], prediction)
    explanation["explanation"] = (
        f"[Appeal reviewed — reason given: \"{appeal_reason}\"] " + explanation["explanation"]
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
