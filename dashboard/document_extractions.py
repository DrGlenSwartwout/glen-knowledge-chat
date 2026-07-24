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
    existing = cx.execute(
        "SELECT id, status FROM client_document_extractions "
        "WHERE document_id=?", (document_id,)).fetchone()
    if existing and existing[1] == "confirmed":
        return existing[0]
    cx.execute("DELETE FROM client_document_extractions WHERE document_id=?",
               (document_id,))
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
