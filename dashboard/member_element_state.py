"""Per-member TCM five-element state (one row per email).

Written from the member's portal-chat analysis; read by the Glendalf backdrop
to pick the setting for the element they need to nourish (the deficient one).
Caller supplies the sqlite3 connection (same pattern as journal_store).
"""
import json
import sqlite3

_ELEMENTS = ("Wood", "Fire", "Earth", "Metal", "Water")


def init_table(cx):
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS member_element_state (
          email TEXT PRIMARY KEY,
          element_scores TEXT,
          dominant_element TEXT,
          deficient_element TEXT,
          source TEXT,
          updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """
    )


def _scored(element_scores):
    if not isinstance(element_scores, dict):
        return {}
    return {k: element_scores[k] for k in _ELEMENTS
            if isinstance(element_scores.get(k), (int, float))}


def deficient_element(element_scores):
    """Lowest-scoring of the five elements = what to nourish. None if unusable."""
    scored = _scored(element_scores)
    return min(scored, key=scored.get) if scored else None


def dominant_element(element_scores):
    scored = _scored(element_scores)
    return max(scored, key=scored.get) if scored else None


def upsert(cx, email, element_scores, source="portal_chat"):
    email = (email or "").strip().lower()
    if not email:
        return None
    init_table(cx)
    dom = dominant_element(element_scores)
    dfc = deficient_element(element_scores)
    cx.execute(
        """
        INSERT INTO member_element_state
          (email, element_scores, dominant_element, deficient_element, source, updated_at)
        VALUES (?,?,?,?,?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        ON CONFLICT(email) DO UPDATE SET
          element_scores=excluded.element_scores,
          dominant_element=excluded.dominant_element,
          deficient_element=excluded.deficient_element,
          source=excluded.source,
          updated_at=excluded.updated_at
        """,
        (email, json.dumps(element_scores or {}), dom, dfc, source),
    )
    cx.commit()
    return get(cx, email)


def get(cx, email):
    email = (email or "").strip().lower()
    if not email:
        return None
    init_table(cx)
    cx.row_factory = sqlite3.Row
    row = cx.execute(
        "SELECT * FROM member_element_state WHERE email=?", (email,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["element_scores"] = json.loads(d["element_scores"]) if d.get("element_scores") else {}
    except Exception:
        d["element_scores"] = {}
    return d
