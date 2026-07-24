"""AI extraction drafts for uploaded client documents.

One draft per document. Everything here is a PROPOSAL: nothing in this module
writes a live clinical store. The live writes happen only at approval, in the
console route, from the payloads this module hands back.

Payload columns are kept separate on purpose (attributes -> person_attributes,
facts -> client_facts, unstructured -> display only) so a display-only lab
value cannot silently acquire a write path later.
"""
import json
from datetime import datetime, timezone

from dashboard import db

_COLS = ("id", "document_id", "email", "status", "narrative_md",
         "attributes_json", "facts_json", "unstructured_json", "model",
         "created_at", "reviewed_at", "reviewed_by")


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS client_document_extractions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL, email TEXT NOT NULL, status TEXT,
        narrative_md TEXT, attributes_json TEXT, facts_json TEXT,
        unstructured_json TEXT, model TEXT, created_at TEXT,
        reviewed_at TEXT, reviewed_by TEXT)""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_cdx_document "
               "ON client_document_extractions(document_id)")
    cx.commit()


def _loads(s):
    try:
        v = json.loads(s or "[]")
        return v if isinstance(v, list) else []
    except (TypeError, ValueError):
        return []


def put_draft(cx, document_id, email, narrative_md, attributes, facts,
              unstructured, model):
    """Write (or replace) the single draft for `document_id`. Returns its id.

    A `confirmed` row is an approved narrative a client may be actively
    reading in their portal, with no history table to recover it from if
    overwritten -- so it is left completely untouched and its existing id
    is returned. `ai_draft` and `rejected` rows are replaced as before;
    re-extraction of a rejected document is the normal re-queue path.
    """
    init_table(cx)
    # The status guard lives INSIDE the DELETE's WHERE clause, not in a
    # preceding SELECT. SELECT-then-DELETE is a classic TOCTOU race: on
    # another connection, confirm() can commit in the gap between the two
    # statements, so the status this code saw is stale by the time the
    # DELETE runs -- and an unconditional DELETE would then wipe out a row
    # that is now confirmed and irrecoverable (no history table). A single
    # statement's WHERE clause is instead evaluated atomically against
    # whatever is committed at the instant the DELETE executes, so there is
    # no window for another connection's commit to land in between the check
    # and the act. Do not "simplify" this back into SELECT-then-DELETE.
    cx.execute(
        "DELETE FROM client_document_extractions "
        "WHERE document_id=? AND status != 'confirmed'", (document_id,))
    existing = cx.execute(
        "SELECT id, status FROM client_document_extractions WHERE document_id=?",
        (document_id,)).fetchone()
    if existing:
        existing_id, existing_status = existing
        if existing_status == "confirmed":
            cx.commit()
            return existing_id
        # A non-confirmed row survived the guarded DELETE above: a concurrent
        # put_draft call for this SAME document_id committed its own
        # replacement row in the gap between our DELETE and this SELECT. Both
        # callers are extracting the same source document, so last-write-wins
        # is fine -- but retrying here would just re-race, and returning this
        # id while quietly dropping our own payload (with no trace) would be
        # dishonest. Make the drop visible and hand back the row that is
        # actually there.
        print(f"put_draft: race on document_id={document_id} -- a concurrent "
              f"writer's '{existing_status}' row (id={existing_id}) already "
              f"exists; this call's payload was NOT written", flush=True)
        cx.commit()
        return existing_id
    try:
        cx.execute(
            "INSERT INTO client_document_extractions"
            "(document_id, email, status, narrative_md, attributes_json,"
            " facts_json, unstructured_json, model, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (document_id, (email or "").strip().lower(), "ai_draft",
             narrative_md or "", json.dumps(attributes or []),
             json.dumps(facts or []), json.dumps(unstructured or []),
             model or "", _now()))
        cx.commit()
    except db.IntegrityError:
        # Lost the UNIQUE(document_id) race: another writer's INSERT landed
        # between our SELECT above and this INSERT. Postgres aborts the
        # transaction on the failed statement, so roll back before issuing
        # any further query on this connection (see household_holds.py for
        # the same shape). Resolve to whatever is there now instead of
        # propagating the exception -- same last-write-wins rationale and
        # visible-drop logging as the branch above.
        cx.rollback()
        row = cx.execute(
            "SELECT id, status FROM client_document_extractions "
            "WHERE document_id=?", (document_id,)).fetchone()
        if row is None:
            raise
        print(f"put_draft: lost UNIQUE race on document_id={document_id} "
              f"during insert -- a concurrent writer's '{row[1]}' row "
              f"(id={row[0]}) already exists; this call's payload was NOT "
              f"written", flush=True)
        return row[0]
    row = cx.execute("SELECT id FROM client_document_extractions "
                     "WHERE document_id=?", (document_id,)).fetchone()
    return row[0]


def get_for_document(cx, document_id):
    init_table(cx)
    r = cx.execute(
        "SELECT id, document_id, email, status, narrative_md, attributes_json,"
        " facts_json, unstructured_json, model, created_at, reviewed_at,"
        " reviewed_by FROM client_document_extractions WHERE document_id=?",
        (document_id,)).fetchone()
    if not r:
        return None
    d = dict(zip(_COLS, r))
    d["attributes"] = _loads(d.pop("attributes_json"))
    d["facts"] = _loads(d.pop("facts_json"))
    d["unstructured"] = _loads(d.pop("unstructured_json"))
    return d


def confirm(cx, extraction_id, narrative_md, reviewed_by):
    """Flip an ai_draft to confirmed. Returns False when it is not an ai_draft
    (already confirmed, or rejected) so approval is idempotent."""
    init_table(cx)
    cur = cx.execute(
        "UPDATE client_document_extractions SET status='confirmed',"
        " narrative_md=?, reviewed_at=?, reviewed_by=? "
        "WHERE id=? AND status='ai_draft'",
        (narrative_md or "", _now(), reviewed_by or "", extraction_id))
    cx.commit()
    return cur.rowcount > 0


def reject(cx, extraction_id, reviewed_by):
    init_table(cx)
    cur = cx.execute(
        "UPDATE client_document_extractions SET status='rejected',"
        " reviewed_at=?, reviewed_by=? WHERE id=? AND status='ai_draft'",
        (_now(), reviewed_by or "", extraction_id))
    cx.commit()
    return cur.rowcount > 0
