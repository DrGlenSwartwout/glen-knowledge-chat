"""Client document store — many documents per client, keyed by lowercased email.

Holds uploaded medical records (labs, imaging, specialist letters) for the
portal document-ingestion feature. Persistence only — no HTTP, no AI, no
rendering. See docs/superpowers/specs/2026-07-23-portal-document-ingestion-design.md

The `blob` column is declared BYTEA, not BLOB: runtime pgcompat does NOT
translate BLOB, so a BLOB column fails outright on Postgres (`type "blob" does
not exist`). BYTEA is native on Postgres and round-trips bytes losslessly on
SQLite. See test_bytea_column_round_trips_all_byte_values.

`client_visible` (0/1) gates whether a document is visible to the CLIENT in
their own portal. Glen's rule: a record a client uploads themselves is their
own file and is visible immediately (source='portal-self'); a record Glen or
staff uploads on the console stays staff-only until Glen explicitly marks it
visible (POST /api/console/client-document/<id>/visibility) -- he may be
holding a third-party record he hasn't reviewed yet. This column is enforced
in the ROUTES (app.py), not here: get_for_email() deliberately still returns
regardless of visibility because the console needs the full record for its
review screen; see list_visible_for_email() for the client-facing read.
"""
import hashlib
from datetime import datetime, timezone

_COLS = ("id", "email", "filename", "content_type", "byte_size", "sha256",
         "source", "uploaded_at", "extract_status", "client_visible")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS client_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL, filename TEXT, content_type TEXT,
        byte_size INTEGER, sha256 TEXT, blob BYTEA, source TEXT,
        uploaded_at TEXT, extract_status TEXT,
        client_visible INTEGER NOT NULL DEFAULT 0)""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_cdoc_email ON client_documents(email)")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_cdoc_email_sha "
               "ON client_documents(email, sha256)")
    # Additive migration for tables created before client_visible existed
    # (idiom matches dashboard/biofield_authoring.py / customers.py /
    # coach_threads.py: unconditional ALTER inside a try/except, caught +
    # ignored when the column already exists; pgcompat translates this to
    # `ALTER TABLE IF EXISTS ... ADD COLUMN IF NOT EXISTS ...` on Postgres so
    # it is a no-op there too).
    #
    # Deliberately NO "NOT NULL DEFAULT 0" here (unlike the CREATE TABLE
    # above): ADD COLUMN ... DEFAULT backfills EVERY existing row to that
    # default immediately, on both SQLite and Postgres -- which is exactly
    # the bug this fixes. With a plain nullable ADD COLUMN, pre-existing
    # rows come back NULL, and NULL is the only signal the backfill below is
    # allowed to touch. A 0 written later by the visibility route
    # (set_client_visible) is a real, explicit value, never NULL -- so once
    # a row has been backfilled (or has always carried an explicit value,
    # e.g. everything inserted through put()), this migration can never
    # touch it again on any subsequent init_table() call, which is called by
    # every store function.
    try:
        cx.execute("ALTER TABLE client_documents ADD COLUMN "
                   "client_visible INTEGER")
    except Exception:
        pass
    # Backfill: ONLY rows left NULL by the ADD COLUMN above (i.e. rows that
    # predate this column entirely) get a value here, by source -- a
    # pre-existing self-uploaded document was always meant to be visible to
    # its owner (put() applies this same rule going forward); anything else
    # (console/unspecified source) becomes staff-only, matching put()'s
    # default. Scoping to `IS NULL` (never `=0`) is what makes an explicit 0
    # written by the visibility route permanent: it is a real value, not
    # NULL, so these UPDATEs never match it again. Once every legacy row is
    # backfilled, both UPDATEs match zero rows on every later call.
    cx.execute("UPDATE client_documents SET client_visible=1 "
               "WHERE source='portal-self' AND client_visible IS NULL")
    cx.execute("UPDATE client_documents SET client_visible=0 "
               "WHERE client_visible IS NULL")
    cx.commit()


def _row(cols, values):
    """dict-ify one row. `client_visible` is normalized to an actual Python
    bool -- never a raw 0/1/None -- so every reader downstream gets a proper
    boolean and NULL (a row this session's init_table() hasn't backfilled
    yet, or a driver returning it that way) reads as hidden (False) rather
    than truthy-by-accident or crashing."""
    if not values:
        return None
    d = dict(zip(cols, values))
    if "client_visible" in d:
        d["client_visible"] = bool(d["client_visible"])
    return d


def put(cx, email, blob, filename, content_type, source):
    """Insert a document. Idempotent on (email, sha256): re-uploading identical
    bytes returns the existing row with deduped=True. Returns {"id", "deduped"}
    or None when email/blob is empty.

    client_visible is set from `source`: a client's own self-upload
    ('portal-self') is visible to them immediately; anything else (console,
    email/fax intake, unspecified) is staff-only until Glen explicitly flips
    it visible."""
    e = _norm(email)
    if not e or not blob:
        return None
    init_table(cx)
    digest = hashlib.sha256(blob).hexdigest()
    visible = 1 if source == "portal-self" else 0
    cur = cx.execute(
        "INSERT OR IGNORE INTO client_documents"
        "(email, filename, content_type, byte_size, sha256, blob, source,"
        " uploaded_at, extract_status, client_visible) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (e, filename or "", content_type or "", len(blob), digest, blob,
         source or "", _now(), "pending", visible))
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
        " uploaded_at, extract_status, client_visible, blob FROM client_documents"
        " WHERE id=?", (doc_id,)).fetchone()
    return _row(_COLS + ("blob",), r)


def get_for_email(cx, doc_id, email):
    """Scoped read — the single isolation primitive. Every client-facing route
    resolves through this so a token can only ever reach its owner's document.

    Deliberately does NOT filter on client_visible: the console review screen
    resolves through this too and needs to see a not-yet-visible document.
    Client-facing routes must enforce visibility themselves (or use
    list_visible_for_email below)."""
    e = _norm(email)
    if not e:
        return None
    init_table(cx)
    r = cx.execute(
        "SELECT id, email, filename, content_type, byte_size, sha256, source,"
        " uploaded_at, extract_status, client_visible, blob FROM client_documents"
        " WHERE id=? AND email=?", (doc_id, e)).fetchone()
    return _row(_COLS + ("blob",), r)


def list_for_email(cx, email):
    """ALL documents for a client regardless of visibility -- the console's
    view. Client-facing routes must use list_visible_for_email instead."""
    e = _norm(email)
    if not e:
        return []
    init_table(cx)
    rows = cx.execute(
        "SELECT id, email, filename, content_type, byte_size, sha256, source,"
        " uploaded_at, extract_status, client_visible FROM client_documents"
        " WHERE email=? ORDER BY id DESC", (e,)).fetchall()
    return [_row(_COLS, r) for r in rows]


def list_visible_for_email(cx, email):
    """Only the client-visible documents for a client -- what the portal's
    own document list must be built from."""
    e = _norm(email)
    if not e:
        return []
    init_table(cx)
    rows = cx.execute(
        "SELECT id, email, filename, content_type, byte_size, sha256, source,"
        " uploaded_at, extract_status, client_visible FROM client_documents"
        " WHERE email=? AND client_visible=1 ORDER BY id DESC", (e,)).fetchall()
    return [_row(_COLS, r) for r in rows]


def set_client_visible(cx, doc_id, visible):
    """Flip a document's client visibility. Console-only action (see
    api_console_client_document_visibility in app.py)."""
    init_table(cx)
    cx.execute("UPDATE client_documents SET client_visible=? WHERE id=?",
               (1 if visible else 0, doc_id))
    cx.commit()


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
        " uploaded_at, extract_status, client_visible FROM client_documents"
        " WHERE extract_status='pending' ORDER BY id LIMIT ?", (int(limit),)
    ).fetchall()
    return [_row(_COLS, r) for r in rows]
