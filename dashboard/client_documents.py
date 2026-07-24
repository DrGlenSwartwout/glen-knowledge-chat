"""Client document store — many documents per client, keyed by lowercased email.

Holds uploaded medical records (labs, imaging, specialist letters) for the
portal document-ingestion feature. Persistence only — no HTTP, no AI, no
rendering. See docs/superpowers/specs/2026-07-23-portal-document-ingestion-design.md

The `blob` column is declared BYTEA, not BLOB: runtime pgcompat does NOT
translate BLOB, so a BLOB column fails outright on Postgres (`type "blob" does
not exist`). BYTEA is native on Postgres and round-trips bytes losslessly on
SQLite. See test_bytea_column_round_trips_all_byte_values.
"""
import hashlib
from datetime import datetime, timezone

_COLS = ("id", "email", "filename", "content_type", "byte_size", "sha256",
         "source", "uploaded_at", "extract_status")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS client_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL, filename TEXT, content_type TEXT,
        byte_size INTEGER, sha256 TEXT, blob BYTEA, source TEXT,
        uploaded_at TEXT, extract_status TEXT)""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_cdoc_email ON client_documents(email)")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_cdoc_email_sha "
               "ON client_documents(email, sha256)")
    cx.commit()


def _row(cols, values):
    return dict(zip(cols, values)) if values else None


def put(cx, email, blob, filename, content_type, source):
    """Insert a document. Idempotent on (email, sha256): re-uploading identical
    bytes returns the existing row with deduped=True. Returns {"id", "deduped"}
    or None when email/blob is empty."""
    e = _norm(email)
    if not e or not blob:
        return None
    init_table(cx)
    digest = hashlib.sha256(blob).hexdigest()
    cur = cx.execute(
        "INSERT OR IGNORE INTO client_documents"
        "(email, filename, content_type, byte_size, sha256, blob, source,"
        " uploaded_at, extract_status) VALUES(?,?,?,?,?,?,?,?,?)",
        (e, filename or "", content_type or "", len(blob), digest, blob,
         source or "", _now(), "pending"))
    cx.commit()
    inserted = cur.rowcount > 0
    # Read the id back by its UNIQUE key: cur.lastrowid raises on Postgres.
    row = cx.execute("SELECT id FROM client_documents WHERE email=? AND sha256=?",
                     (e, digest)).fetchone()
    return {"id": row[0], "deduped": not inserted}


def get(cx, doc_id):
    init_table(cx)
    r = cx.execute(
        "SELECT id, email, filename, content_type, byte_size, sha256, source,"
        " uploaded_at, extract_status, blob FROM client_documents WHERE id=?",
        (doc_id,)).fetchone()
    return _row(_COLS + ("blob",), r)


def get_for_email(cx, doc_id, email):
    """Scoped read — the single isolation primitive. Every client-facing route
    resolves through this so a token can only ever reach its owner's document."""
    e = _norm(email)
    if not e:
        return None
    init_table(cx)
    r = cx.execute(
        "SELECT id, email, filename, content_type, byte_size, sha256, source,"
        " uploaded_at, extract_status, blob FROM client_documents"
        " WHERE id=? AND email=?", (doc_id, e)).fetchone()
    return _row(_COLS + ("blob",), r)


def list_for_email(cx, email):
    e = _norm(email)
    if not e:
        return []
    init_table(cx)
    rows = cx.execute(
        "SELECT id, email, filename, content_type, byte_size, sha256, source,"
        " uploaded_at, extract_status FROM client_documents WHERE email=?"
        " ORDER BY id DESC", (e,)).fetchall()
    return [_row(_COLS, r) for r in rows]


def set_extract_status(cx, doc_id, status):
    init_table(cx)
    cx.execute("UPDATE client_documents SET extract_status=? WHERE id=?",
               (status, doc_id))
    cx.commit()


def claim_for_extraction(cx, doc_id):
    """Atomically claim a pending document for extraction. Returns True iff
    THIS call won the claim.

    Production runs multiple web instances against one database, so a
    plain SELECT-then-UPDATE (check the status, then set it) is a race: two
    instances can both see 'pending' in the SELECT and both proceed to
    extract the same document -- duplicate paid Claude calls and racing
    writes into the draft store. The guard must therefore live INSIDE the
    UPDATE's WHERE clause so it is evaluated atomically against whatever is
    committed at the instant the UPDATE executes, not against a stale read
    from a moment earlier. This exact class of bug (SELECT-then-act instead
    of a WHERE-guarded statement) was already found and fixed twice
    elsewhere in this feature (see document_extractions.put_draft and
    portal_identity.py) -- do not "simplify" this back into a SELECT first.

    Note: if a process dies mid-extraction, the document is left stuck in
    'extracting' and needs a manual requeue (set_extract_status back to
    'pending'). That is an accepted limitation here -- no lease/timeout
    system is built for it.
    """
    init_table(cx)
    cur = cx.execute(
        "UPDATE client_documents SET extract_status='extracting' "
        "WHERE id=? AND extract_status='pending'", (doc_id,))
    cx.commit()
    return cur.rowcount > 0


def pending(cx, limit=20):
    init_table(cx)
    rows = cx.execute(
        "SELECT id, email, filename, content_type, byte_size, sha256, source,"
        " uploaded_at, extract_status FROM client_documents"
        " WHERE extract_status='pending' ORDER BY id LIMIT ?", (int(limit),)
    ).fetchall()
    return [_row(_COLS, r) for r in rows]
