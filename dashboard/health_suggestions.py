"""Pending health-record edits surfaced from the portal chat (or entered by a
clinician) for the client to review and approve before they land in their
intake record. Pure sqlite; caller passes the connection (mirrors
dashboard/portal_chat.py and the other dashboard/*.py stores).

Two producers write into this queue:
  - the portal chat hook (app.py, beside portal_chat.record_exchange): best-
    effort LLM extraction of a chat turn into candidate field edits.
  - a clinician, entering a suggested edit directly (source='clinician').

Both feed the same table; `list_pending`/`count_pending` drive the "My Health
Profile" suggestion badge (see health_profile.build_block), and `resolve`
is how a client accepts/edits/dismisses a suggestion.
"""
import json
import sqlite3
from datetime import datetime, timezone

_VALID_STATUSES = ("confirmed", "edited", "dismissed")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS health_suggestions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        source TEXT,
        source_msg_id INTEGER,
        field_id TEXT,
        suggested_value TEXT,
        rationale TEXT,
        status TEXT DEFAULT 'pending',
        created_at TEXT,
        resolved_at TEXT)""")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_hs_email_status "
               "ON health_suggestions(email, status)")
    # Partial UNIQUE index: only pending rows collide, so a re-mention of the
    # same field/value while a suggestion is still pending is a no-op (INSERT
    # OR IGNORE below), but the same field/value can be re-suggested later
    # once the prior row has been resolved (it's no longer in the partial
    # index's row set).
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_hs_pending_dedupe "
               "ON health_suggestions(email, field_id, suggested_value) "
               "WHERE status='pending'")
    cx.commit()


def add_pending(cx, email, field_id, value, rationale, source, source_msg_id=None):
    """Queue one candidate field edit as a pending suggestion. INSERT OR IGNORE
    against the partial unique index so a re-mention of the same (email,
    field_id, value) while a suggestion is still pending does not stack a
    duplicate row. `source` is 'chat' or 'clinician'. `suggested_value` is
    stored as a string (non-str values are JSON-encoded). Returns the new row
    id, or None if the insert was ignored (duplicate) or email/field_id was
    empty."""
    init_table(cx)
    e = _norm(email)
    fid = (field_id or "").strip()
    if not e or not fid:
        return None
    val = value if isinstance(value, str) else json.dumps(value)
    cur = cx.execute(
        "INSERT OR IGNORE INTO health_suggestions "
        "(email, source, source_msg_id, field_id, suggested_value, rationale, "
        " status, created_at) VALUES (?,?,?,?,?,?,'pending',?)",
        (e, source, source_msg_id, fid, val, rationale, _now()))
    cx.commit()
    return cur.lastrowid if cur.rowcount == 1 else None


def list_pending(cx, email):
    """The client's pending suggestions, oldest first."""
    init_table(cx)
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT id, source, field_id, suggested_value, rationale FROM health_suggestions "
        "WHERE email=? AND status='pending' ORDER BY id", (_norm(email),)).fetchall()
    return [{"id": r["id"], "source": r["source"], "field_id": r["field_id"],
             "suggested_value": r["suggested_value"], "rationale": r["rationale"]}
            for r in rows]


def count_pending(cx, email):
    init_table(cx)
    row = cx.execute(
        "SELECT COUNT(*) FROM health_suggestions WHERE email=? AND status='pending'",
        (_norm(email),)).fetchone()
    return row[0] if row else 0


def resolve(cx, sug_id, email, status):
    """Resolve one of the client's own pending suggestions (confirmed/edited/
    dismissed), stamping resolved_at. Scoped to `email` so a client can't
    resolve another client's row. No-op (returns False) if the id doesn't
    belong to that email or `status` isn't one of the valid values."""
    init_table(cx)
    if status not in _VALID_STATUSES:
        return False
    cur = cx.execute(
        "UPDATE health_suggestions SET status=?, resolved_at=? WHERE id=? AND email=?",
        (status, _now(), sug_id, _norm(email)))
    cx.commit()
    return cur.rowcount == 1


def extract_from_turn(client_msg, assistant_msg, extractor=None):
    """Parse one chat turn into candidate editable health-record edits:
    [{"field_id", "value", "rationale"}, ...]. Pure and testable — pass
    `extractor(client_msg, assistant_msg) -> list[dict]` to inject a fake (or
    a real LLM call) for tests; the return is always filtered down to
    `health_profile.EDITABLE_FIELD_IDS` before it comes back, so an
    extractor can never smuggle in an identity/consent field.

    With no `extractor` given this is a documented no-op seam (returns [])
    rather than wiring a live model call — an LLM-backed default belongs in
    app.py alongside `_CONCIERGE_EXTRACT_SYSTEM`, where the Anthropic client
    and streaming context already live; duplicating that here would either
    import app.py (circular) or fork the client setup.
    """
    if extractor is None:
        candidates = []
    else:
        candidates = extractor(client_msg, assistant_msg) or []
    # Lazy import: health_profile imports dashboard.intake, and nothing here
    # is needed at module load time, so import inside the function to avoid
    # any import-cycle risk with modules that import health_suggestions.
    from dashboard import health_profile
    allowed = health_profile.EDITABLE_FIELD_IDS
    out = []
    for c in candidates:
        fid = (c or {}).get("field_id")
        if fid in allowed:
            out.append({"field_id": fid, "value": c.get("value"),
                        "rationale": c.get("rationale")})
    return out
