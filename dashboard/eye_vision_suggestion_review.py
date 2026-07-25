"""Practitioner decisions for deterministic Eye & Vision suggestion drafts."""
from datetime import datetime, timezone


VALID_DECISIONS = {"approved", "rejected"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS eye_vision_suggestion_reviews (
            client_email TEXT NOT NULL,
            suggestion_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            reviewer_id TEXT NOT NULL,
            client_safe_wording TEXT NOT NULL DEFAULT '',
            safety_review TEXT NOT NULL DEFAULT '',
            reviewed_at TEXT NOT NULL,
            PRIMARY KEY (client_email, suggestion_id)
        )""")
    cx.commit()


def decisions(cx, email):
    init_table(cx)
    rows = cx.execute("""
        SELECT suggestion_id, decision, reviewer_id, client_safe_wording,
               safety_review, reviewed_at
        FROM eye_vision_suggestion_reviews
        WHERE lower(client_email)=lower(?)
        """, ((email or "").strip(),)).fetchall()
    return {
        row[0]: {
            "decision": row[1], "reviewer_id": row[2],
            "client_safe_wording": row[3], "safety_review": row[4],
            "reviewed_at": row[5],
        }
        for row in rows
    }


def save(cx, email, suggestion_id, decision, reviewer_id,
         client_safe_wording="", safety_review=""):
    decision = (decision or "").strip().lower()
    if decision not in VALID_DECISIONS:
        raise ValueError("decision must be approved or rejected")
    email = (email or "").strip().lower()
    suggestion_id = (suggestion_id or "").strip()
    reviewer_id = (reviewer_id or "").strip()
    wording = (client_safe_wording or "").strip()
    safety = (safety_review or "").strip()
    if not email or not suggestion_id or not reviewer_id:
        raise ValueError("email, suggestion_id, and reviewer_id are required")
    if decision == "approved" and (not wording or not safety):
        raise ValueError(
            "approval requires client-safe wording and a documented safety review")
    init_table(cx)
    cx.execute("""
        INSERT INTO eye_vision_suggestion_reviews (
            client_email, suggestion_id, decision, reviewer_id,
            client_safe_wording, safety_review, reviewed_at
        ) VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(client_email, suggestion_id) DO UPDATE SET
            decision=excluded.decision,
            reviewer_id=excluded.reviewer_id,
            client_safe_wording=excluded.client_safe_wording,
            safety_review=excluded.safety_review,
            reviewed_at=excluded.reviewed_at
        """, (email, suggestion_id, decision, reviewer_id,
              wording, safety, _now()))
    cx.commit()
    return decisions(cx, email)[suggestion_id]


def overlay(contract, stored):
    for candidate in (contract or {}).get("candidates") or []:
        review = (stored or {}).get(candidate.get("suggestion_id"))
        if review:
            candidate["review"] = review
            candidate["status"] = review["decision"]
    return contract
