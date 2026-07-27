"""SQLite audit trail: every moderation decision, logged for compliance reporting.

Uses the stdlib sqlite3 module - zero new dependency, and light enough for
Render's memory budget. Disclosed limitation: Render's free-tier disk is
ephemeral, so this resets on redeploy/restart. Fine for a demo; not a claim
of production data retention.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.environ.get("AUDIT_DB_PATH", str(Path(__file__).resolve().parent / "audit.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    comment_text TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    risk_level TEXT NOT NULL,
    policy_citation TEXT,
    explanation TEXT,
    agent_status TEXT NOT NULL,
    appeal_of INTEGER
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(_SCHEMA)


init_db()


def log_decision(
    comment_text: str,
    label: str,
    confidence: float,
    risk_level: str,
    policy_citation: str | None,
    explanation: str | None,
    agent_status: str,
    appeal_of: int | None = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO decisions
               (created_at, comment_text, label, confidence, risk_level,
                policy_citation, explanation, agent_status, appeal_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                comment_text,
                label,
                confidence,
                risk_level,
                policy_citation,
                explanation,
                agent_status,
                appeal_of,
            ),
        )
        return cur.lastrowid


def get_decision(decision_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
        return dict(row) if row else None


def update_decision(
    decision_id: int,
    label: str,
    confidence: float,
    risk_level: str,
    policy_citation: str | None,
    explanation: str | None,
    agent_status: str,
):
    with _connect() as conn:
        conn.execute(
            """UPDATE decisions
               SET label=?, confidence=?, risk_level=?, policy_citation=?,
                   explanation=?, agent_status=?
               WHERE id=?""",
            (label, confidence, risk_level, policy_citation, explanation, agent_status, decision_id),
        )


def get_recent(limit: int = 25) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"]
        auto_actioned = conn.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE agent_status = 'auto_action'"
        ).fetchone()["n"]
        avg_conf = conn.execute("SELECT AVG(confidence) AS a FROM decisions").fetchone()["a"]
        return {
            "total_decisions": total,
            "pct_auto_actioned": (auto_actioned / total * 100) if total else 0.0,
            "avg_confidence": (avg_conf or 0.0) * 100,
        }


def get_label_distribution() -> dict[str, int]:
    """Count of original (non-appeal) decisions per predicted label."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT label, COUNT(*) AS n FROM decisions WHERE appeal_of IS NULL GROUP BY label"
        ).fetchall()
        return {r["label"]: r["n"] for r in rows}


def get_appeal_counts() -> tuple[int, int]:
    """(total original decisions, count of distinct original decisions that were appealed)."""
    with _connect() as conn:
        total_original = conn.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE appeal_of IS NULL"
        ).fetchone()["n"]
        total_appealed = conn.execute(
            "SELECT COUNT(DISTINCT appeal_of) AS n FROM decisions WHERE appeal_of IS NOT NULL"
        ).fetchone()["n"]
        return total_original, total_appealed


def get_activity_heatmap() -> list[dict]:
    """Return decision counts bucketed by (day_of_week, hour) for the activity heatmap."""
    with _connect() as conn:
        rows = conn.execute("SELECT created_at FROM decisions").fetchall()
    buckets: dict[tuple[int, int], int] = {}
    for row in rows:
        try:
            dt = datetime.fromisoformat(row["created_at"])
        except ValueError:
            continue
        key = (dt.weekday(), dt.hour)
        buckets[key] = buckets.get(key, 0) + 1
    return [{"day": d, "hour": h, "count": c} for (d, h), c in buckets.items()]
