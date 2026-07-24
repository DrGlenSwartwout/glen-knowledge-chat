# Portal Document Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let clients and the console upload medical records, have Claude draft structured clinical attributes plus a client-facing narrative, and land both only after Glen approves them in one console screen.

**Architecture:** Two new runtime-created tables hold the file bytes (`client_documents`, a `BYTEA` column) and the AI draft (`client_document_extractions`). Two thin upload routes share one validator. A worker extracts, drops any item whose `source_quote` is absent from the source, and writes a draft only. A single console approval writes the live stores; a portal tile shows the client their own file and, once approved, the narrative.

**Tech Stack:** Python 3, Flask (`app.py`), the `dashboard/` module convention (pure functions taking a `cx`), SQLite in dev/tests and Postgres in prod through `dashboard/db.py` + `dashboard/pgcompat.py`, the `anthropic` SDK, vanilla JS for the portal tile.

## Global Constraints

These apply to every task. Values are copied verbatim from the spec.

- **Binary columns must be declared `BYTEA`, never `BLOB`.** Runtime `pgcompat.translate_sql` does not translate `BLOB`, so a runtime-created `BLOB` column fails on Postgres with `type "blob" does not exist`. `BYTEA` is native on Postgres and round-trips `bytes` losslessly on SQLite.
- **Never use `cur.lastrowid`.** It raises on Postgres. Read the id back by its UNIQUE key instead.
- **Never set `cx.row_factory` in a store module** without saving and restoring it. A leaked row_factory breaks unrelated readers sharing the connection. These stores use tuple indexing and build dicts explicitly.
- **Bytes never go to `$DATA_DIR`.** That disk is mounted on the web service only; prod is multi-instance. Bytes live in the database.
- **Approval must never write `client_conditions`.** That table is the single eye-condition support-program override (`email TEXT PRIMARY KEY`); writing extracted diagnoses there would clobber Glen's support-program selection.
- **`client_facts` is boolean-only** (`value INTEGER`, `set_fact` coerces `1 if value else 0`). Never route a lab value or medication name there.
- **Size cap: 30 MB.** Over-cap is a 400, never a truncated store.
- **Accepted for extraction:** `application/pdf` and `image/*`. Everything else is stored with `extract_status='skipped-unreadable'`.
- **The portal never shows the client extracted attributes, facts, or labs** — only their own file and, after approval, the narrative.
- **The fabrication guard verifies quotes against the model's `document_text` transcription, never against its narrative.** Checking a model's quotes against its own summary is self-validating and therefore vacuous. A missing transcription must FAIL CLOSED (empty haystack → every item dropped → empty draft), never fail open.
- **Running tests:** run *targeted* test files during development (`python3 -m pytest tests/test_x.py -v`). Do **not** run a bare full suite from a shell carrying real credentials — it sends real email. For the full gate use `bash ci/run-tests.sh`, which ratchets against `tests/known_failures.txt` and fails only on a NEW failure.

---

## File Structure

**Create:**
- `dashboard/client_documents.py` — file persistence. No HTTP, no AI.
- `dashboard/document_extractions.py` — AI draft persistence. No HTTP, no AI.
- `dashboard/document_extract.py` — the Claude call + fabrication guard. No HTTP, no persistence beyond calling the two stores.
- `static/js/portal-documents.js` — the My Records tile.
- `static/js/console-documents.js` — the console review section renderer.
- Tests: `tests/test_client_documents_store.py`, `tests/test_document_extractions_store.py`, `tests/test_document_upload_routes.py`, `tests/test_document_extract.py`, `tests/test_document_approve.py`, `tests/test_portal_documents_api.py`

**Modify:**
- `app.py` — upload routes, portal read routes, approve/reject route, console section.
- `static/client-portal.html` — mount div + script tag for the tile.

Each of the three new `dashboard/` modules has one responsibility and can be tested without Flask, matching the established convention (`client_photos.py`, `canonical_tags.py`).

---

### Task 1: `client_documents` store

**Files:**
- Create: `dashboard/client_documents.py`
- Test: `tests/test_client_documents_store.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `init_table(cx) -> None`
  - `put(cx, email, blob, filename, content_type, source) -> {"id": int, "deduped": bool}`
  - `get(cx, doc_id) -> dict | None` (includes `blob`)
  - `get_for_email(cx, doc_id, email) -> dict | None` (includes `blob`)
  - `list_for_email(cx, email) -> [dict]` (no `blob`)
  - `set_extract_status(cx, doc_id, status) -> None`
  - `pending(cx, limit=20) -> [dict]` (no `blob`)

Row dicts have keys: `id, email, filename, content_type, byte_size, sha256, source, uploaded_at, extract_status` (plus `blob` where noted).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_client_documents_store.py
import sqlite3
from dashboard import client_documents as cd


def _cx():
    cx = sqlite3.connect(":memory:")
    cd.init_table(cx)
    return cx


def test_bytea_column_round_trips_all_byte_values():
    """Pins the BYTEA choice: BLOB would fail on Postgres, and BYTEA must still
    carry every byte value (including NUL) losslessly on SQLite."""
    cx = _cx()
    blob = bytes(range(256)) * 40
    r = cd.put(cx, "c@x.com", blob, "scan.pdf", "application/pdf", "console")
    got = cd.get(cx, r["id"])
    assert got["blob"] == blob
    assert got["byte_size"] == len(blob)


def test_put_then_get_round_trip_fields():
    cx = _cx()
    r = cd.put(cx, "C@X.com", b"hello", "labs.pdf", "application/pdf", "portal-self")
    assert r["deduped"] is False
    got = cd.get(cx, r["id"])
    assert got["email"] == "c@x.com"          # lowercased
    assert got["filename"] == "labs.pdf"
    assert got["content_type"] == "application/pdf"
    assert got["source"] == "portal-self"
    assert got["extract_status"] == "pending"
    assert got["uploaded_at"]


def test_identical_bytes_for_same_email_dedup_to_one_row():
    cx = _cx()
    a = cd.put(cx, "c@x.com", b"same", "a.pdf", "application/pdf", "console")
    b = cd.put(cx, "c@x.com", b"same", "b.pdf", "application/pdf", "portal-self")
    assert b["deduped"] is True
    assert a["id"] == b["id"]
    assert len(cd.list_for_email(cx, "c@x.com")) == 1


def test_same_bytes_different_email_are_separate_rows():
    cx = _cx()
    a = cd.put(cx, "a@x.com", b"same", "a.pdf", "application/pdf", "console")
    b = cd.put(cx, "b@x.com", b"same", "a.pdf", "application/pdf", "console")
    assert a["id"] != b["id"]


def test_list_for_email_excludes_blob():
    cx = _cx()
    cd.put(cx, "c@x.com", b"bytes", "a.pdf", "application/pdf", "console")
    rows = cd.list_for_email(cx, "c@x.com")
    assert len(rows) == 1
    assert "blob" not in rows[0]


def test_get_for_email_is_the_scoping_primitive():
    cx = _cx()
    r = cd.put(cx, "owner@x.com", b"bytes", "a.pdf", "application/pdf", "console")
    assert cd.get_for_email(cx, r["id"], "owner@x.com") is not None
    assert cd.get_for_email(cx, r["id"], "other@x.com") is None


def test_put_rejects_empty_email_or_blob():
    cx = _cx()
    assert cd.put(cx, "", b"x", "a.pdf", "application/pdf", "console") is None
    assert cd.put(cx, "c@x.com", b"", "a.pdf", "application/pdf", "console") is None


def test_set_extract_status_and_pending():
    cx = _cx()
    r = cd.put(cx, "c@x.com", b"one", "a.pdf", "application/pdf", "console")
    assert [p["id"] for p in cd.pending(cx)] == [r["id"]]
    cd.set_extract_status(cx, r["id"], "drafted")
    assert cd.pending(cx) == []
    assert cd.get(cx, r["id"])["extract_status"] == "drafted"


def test_pending_excludes_unreadable():
    cx = _cx()
    r = cd.put(cx, "c@x.com", b"one", "a.zip", "application/zip", "console")
    cd.set_extract_status(cx, r["id"], "skipped-unreadable")
    assert cd.pending(cx) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_client_documents_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.client_documents'`

- [ ] **Step 3: Write the implementation**

```python
# dashboard/client_documents.py
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


def pending(cx, limit=20):
    init_table(cx)
    rows = cx.execute(
        "SELECT id, email, filename, content_type, byte_size, sha256, source,"
        " uploaded_at, extract_status FROM client_documents"
        " WHERE extract_status='pending' ORDER BY id LIMIT ?", (int(limit),)
    ).fetchall()
    return [_row(_COLS, r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_client_documents_store.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/client_documents.py tests/test_client_documents_store.py
git commit -m "feat(documents): client_documents store with BYTEA blob column"
```

---

### Task 2: `client_document_extractions` draft store

**Files:**
- Create: `dashboard/document_extractions.py`
- Test: `tests/test_document_extractions_store.py`

**Interfaces:**
- Consumes: nothing from Task 1 at the code level (linked only by `document_id`).
- Produces:
  - `init_table(cx) -> None`
  - `put_draft(cx, document_id, email, narrative_md, attributes, facts, unstructured, model) -> int` (the extraction id)
  - `get_for_document(cx, document_id) -> dict | None`
  - `confirm(cx, extraction_id, narrative_md, reviewed_by) -> bool`
  - `reject(cx, extraction_id, reviewed_by) -> bool`

`attributes`, `facts`, `unstructured` are lists of dicts, stored as JSON and returned already decoded under the keys `attributes`, `facts`, `unstructured`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_document_extractions_store.py
import sqlite3
from dashboard import document_extractions as dx


def _cx():
    cx = sqlite3.connect(":memory:")
    dx.init_table(cx)
    return cx


def test_put_draft_round_trips_decoded_payloads():
    cx = _cx()
    eid = dx.put_draft(
        cx, 7, "c@x.com", "You had a panel done.",
        attributes=[{"field": "conditions", "value": "Glaucoma", "source_quote": "dx: glaucoma"}],
        facts=[{"fact_key": "on_areds2", "value": True, "source_quote": "taking AREDS2"}],
        unstructured=[{"label": "HbA1c", "value": "6.4", "source_quote": "HbA1c 6.4"}],
        model="claude-opus-4-8")
    got = dx.get_for_document(cx, 7)
    assert got["id"] == eid
    assert got["status"] == "ai_draft"
    assert got["email"] == "c@x.com"
    assert got["narrative_md"] == "You had a panel done."
    assert got["model"] == "claude-opus-4-8"
    assert got["attributes"][0]["value"] == "Glaucoma"
    assert got["facts"][0]["fact_key"] == "on_areds2"
    assert got["unstructured"][0]["label"] == "HbA1c"
    assert got["reviewed_at"] is None


def test_get_for_document_returns_none_when_absent():
    assert dx.get_for_document(_cx(), 999) is None


def test_confirm_sets_status_narrative_and_reviewer():
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft text", [], [], [], "m")
    assert dx.confirm(cx, eid, "edited text", "glen") is True
    got = dx.get_for_document(cx, 7)
    assert got["status"] == "confirmed"
    assert got["narrative_md"] == "edited text"
    assert got["reviewed_by"] == "glen"
    assert got["reviewed_at"]


def test_confirm_is_idempotent_second_call_is_a_noop():
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    assert dx.confirm(cx, eid, "first", "glen") is True
    assert dx.confirm(cx, eid, "second", "glen") is False
    assert dx.get_for_document(cx, 7)["narrative_md"] == "first"


def test_reject_sets_status_and_blocks_later_confirm():
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    assert dx.reject(cx, eid, "glen") is True
    assert dx.get_for_document(cx, 7)["status"] == "rejected"
    assert dx.confirm(cx, eid, "x", "glen") is False


def test_put_draft_replaces_a_prior_draft_for_the_same_document():
    """Re-extraction must not leave two competing drafts on one document."""
    cx = _cx()
    dx.put_draft(cx, 7, "c@x.com", "old", [], [], [], "m")
    dx.put_draft(cx, 7, "c@x.com", "new", [], [], [], "m")
    got = dx.get_for_document(cx, 7)
    assert got["narrative_md"] == "new"
    rows = cx.execute("SELECT COUNT(*) FROM client_document_extractions "
                      "WHERE document_id=7").fetchone()
    assert rows[0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_document_extractions_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.document_extractions'`

- [ ] **Step 3: Write the implementation**

```python
# dashboard/document_extractions.py
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
    """Write (or replace) the single draft for `document_id`. Returns its id."""
    init_table(cx)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_document_extractions_store.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/document_extractions.py tests/test_document_extractions_store.py
git commit -m "feat(documents): extraction draft store with idempotent confirm"
```

---

### Task 3: Upload routes — shared validator, portal + console

**Files:**
- Modify: `app.py` (add after the existing `/api/portal/<token>/photo` routes, near line 20585)
- Test: `tests/test_document_upload_routes.py`

**Interfaces:**
- Consumes: `client_documents.put` from Task 1.
- Produces:
  - `_accept_document_upload(cx, email, f, source) -> (dict, int)` — returns a JSON-ready body and an HTTP status.
  - `POST /api/portal/<token>/documents`
  - `POST /api/console/client-document`
  - Module constants `_DOC_MAX = 30 * 1024 * 1024`, `_DOC_EXTRACTABLE = ("application/pdf",)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_document_upload_routes.py
import importlib, io, sqlite3, sys
from pathlib import Path


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONSOLE_SECRET", "test-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _token(appmod, email):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    cx.commit(); cx.close()
    return tok


def _upload(client, url, data=b"%PDF-1.4 fake", name="labs.pdf",
            ctype="application/pdf"):
    return client.post(url, data={"file": (io.BytesIO(data), name, ctype)},
                       content_type="multipart/form-data")


def test_portal_upload_stores_against_the_token_owner(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "owner@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents")
    assert r.status_code == 200 and r.get_json()["ok"] is True
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    rows = cd.list_for_email(cx, "owner@x.com")
    assert len(rows) == 1
    assert rows[0]["source"] == "portal-self"
    assert rows[0]["extract_status"] == "pending"


def test_portal_upload_cannot_reach_another_clients_documents(tmp_path, monkeypatch):
    """Token scoping: a token writes ONLY its own owner's email."""
    appmod = _app(tmp_path, monkeypatch)
    _token(appmod, "other@x.com")
    tok_a = _token(appmod, "a@x.com")
    _upload(appmod.app.test_client(), f"/api/portal/{tok_a}/documents")
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    assert len(cd.list_for_email(cx, "a@x.com")) == 1
    assert cd.list_for_email(cx, "other@x.com") == []


def test_portal_upload_unknown_token_404s(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    r = _upload(appmod.app.test_client(), "/api/portal/nope/documents")
    assert r.status_code == 404


def test_upload_rejects_empty_file(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents", data=b"")
    assert r.status_code == 400


def test_upload_rejects_over_cap(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    big = b"x" * (30 * 1024 * 1024 + 1)
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents", data=big)
    assert r.status_code == 400
    assert "too large" in r.get_json()["error"]


def test_unreadable_type_is_stored_but_marked_skipped(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents",
                data=b"PK zip", name="records.zip", ctype="application/zip")
    assert r.status_code == 200
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    rows = cd.list_for_email(cx, "c@x.com")
    assert rows[0]["extract_status"] == "skipped-unreadable"


def test_image_type_is_extractable(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents",
            data=b"\xff\xd8\xff", name="record.jpg", ctype="image/jpeg")
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cd.list_for_email(cx, "c@x.com")[0]["extract_status"] == "pending"


def test_reupload_of_identical_bytes_is_deduped(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    c = appmod.app.test_client()
    _upload(c, f"/api/portal/{tok}/documents")
    r = _upload(c, f"/api/portal/{tok}/documents")
    assert r.status_code == 200 and r.get_json()["deduped"] is True
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    assert len(cd.list_for_email(cx, "c@x.com")) == 1


def test_console_upload_requires_the_console_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-document",
               data={"email": "c@x.com",
                     "file": (io.BytesIO(b"%PDF"), "a.pdf", "application/pdf")},
               content_type="multipart/form-data")
    assert r.status_code == 401


def test_console_upload_with_key_stores_with_console_source(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-document?key=test-secret",
               data={"email": "C@x.com",
                     "file": (io.BytesIO(b"%PDF"), "a.pdf", "application/pdf")},
               content_type="multipart/form-data")
    assert r.status_code == 200
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cd.list_for_email(cx, "c@x.com")[0]["source"] == "console"


def test_console_upload_requires_an_email(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-document?key=test-secret",
               data={"file": (io.BytesIO(b"%PDF"), "a.pdf", "application/pdf")},
               content_type="multipart/form-data")
    assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_document_upload_routes.py -v`
Expected: FAIL — all upload requests 404 (routes do not exist yet)

- [ ] **Step 3: Write the implementation**

Insert into `app.py` immediately after `api_portal_photo_serve` (the function ending around line 20583):

```python
_DOC_MAX = 30 * 1024 * 1024
_DOC_EXTRACTABLE = ("application/pdf",)


def _doc_extract_status(ctype):
    """PDFs and images go to the extractor; everything else is stored as-is.
    Glen gets a wide range of file types — the rule is store everything,
    extract what is readable."""
    c = (ctype or "").lower()
    if c in _DOC_EXTRACTABLE or c.startswith("image/"):
        return "pending"
    return "skipped-unreadable"


def _accept_document_upload(cx, email, f, source):
    """Validate + store one uploaded document. Shared by the portal and console
    routes so the two can never drift apart. Returns (body, status)."""
    from dashboard import client_documents as _cd
    blob = f.read() if f else b""
    if not blob:
        return {"ok": False, "error": "no file uploaded"}, 400
    if len(blob) > _DOC_MAX:
        return {"ok": False, "error": "file too large (max 30 MB)"}, 400
    ctype = (getattr(f, "mimetype", "") or "").lower()
    res = _cd.put(cx, email, blob, getattr(f, "filename", "") or "", ctype, source)
    if not res:
        return {"ok": False, "error": "could not store file"}, 400
    if not res["deduped"]:
        _cd.set_extract_status(cx, res["id"], _doc_extract_status(ctype))
    return {"ok": True, "id": res["id"], "deduped": res["deduped"]}, 200


@app.route("/api/portal/<token>/documents", methods=["POST"])
def api_portal_document_upload(token):
    """Client self-uploads a medical record. Token-scoped: writes ONLY the
    token owner's email, exactly as the /photo upload does."""
    from dashboard import client_portal as _cp
    with _db_lock, db.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
        email = (portal.get("email") or "").strip().lower() if portal else ""
        if not email:
            return jsonify({"ok": False, "error": "not found"}), 404
        body, status = _accept_document_upload(
            cx, email, request.files.get("file"), "portal-self")
    return jsonify(body), status


@app.route("/api/console/client-document", methods=["POST"])
def api_console_client_document_upload():
    """Console-side upload for records that arrive by email or fax."""
    if CONSOLE_SECRET:
        key = _present_console_key()
        if key != CONSOLE_SECRET and not _owner_token_ok(key):
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
    email = (request.form.get("email") or "").strip().lower()
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400
    with _db_lock, db.connect(LOG_DB) as cx:
        body, status = _accept_document_upload(
            cx, email, request.files.get("file"), "console")
    return jsonify(body), status
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_document_upload_routes.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_document_upload_routes.py
git commit -m "feat(documents): portal + console upload routes on one shared validator"
```

---

### Task 4: Extraction pipeline with the fabrication guard

**Files:**
- Create: `dashboard/document_extract.py`
- Test: `tests/test_document_extract.py`

**Interfaces:**
- Consumes: `client_documents.get/set_extract_status/pending` (Task 1), `document_extractions.put_draft` (Task 2), `canonical_tags.resolve`.
- Produces:
  - `verify_quotes(items, source_text) -> (kept, dropped)`
  - `extract_document(cx, doc_id, call_model=None, source_text=None) -> dict | None`
  - `run_pending(cx, limit=5, call_model=None) -> int`
  - `_MODEL` (module constant, overridable via `DOC_EXTRACT_MODEL`)

`call_model` is injected in tests. In production it defaults to the real Anthropic call.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_document_extract.py
import sqlite3
from dashboard import client_documents as cd
from dashboard import document_extractions as dx
from dashboard import document_extract as de


def _cx():
    cx = sqlite3.connect(":memory:")
    cd.init_table(cx); dx.init_table(cx)
    return cx


def _doc(cx, email="c@x.com", blob=b"%PDF fake"):
    return cd.put(cx, email, blob, "labs.pdf", "application/pdf", "console")["id"]


SOURCE = "Patient reports taking AREDS2 daily. Assessment: glaucoma. HbA1c 6.4."


def _fake_model(payload):
    def call(blob, content_type):
        return payload
    return call


def test_verify_quotes_keeps_grounded_items_and_drops_invented_ones():
    items = [{"value": "Glaucoma", "source_quote": "Assessment: glaucoma"},
             {"value": "Diabetes", "source_quote": "Assessment: diabetes"}]
    kept, dropped = de.verify_quotes(items, SOURCE)
    assert [k["value"] for k in kept] == ["Glaucoma"]
    assert [d["value"] for d in dropped] == ["Diabetes"]


def test_verify_quotes_drops_items_with_no_quote_at_all():
    kept, dropped = de.verify_quotes([{"value": "Glaucoma"}], SOURCE)
    assert kept == [] and len(dropped) == 1


def test_verify_quotes_is_case_and_whitespace_insensitive():
    kept, _ = de.verify_quotes(
        [{"value": "X", "source_quote": "  ASSESSMENT:   GLAUCOMA "}], SOURCE)
    assert len(kept) == 1


def test_extract_writes_a_draft_and_marks_the_document(monkeypatch):
    cx = _cx()
    doc_id = _doc(cx)
    payload = {
        "narrative_md": "Your panel showed a few things.",
        "attributes": [{"field": "conditions", "value": "glaucoma",
                        "source_quote": "Assessment: glaucoma"}],
        "facts": [{"fact_key": "on_areds2", "value": True,
                   "source_quote": "taking AREDS2 daily"}],
        "unstructured": [{"label": "HbA1c", "value": "6.4",
                          "source_quote": "HbA1c 6.4"}],
    }
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                              source_text=SOURCE)
    assert out["dropped"] == 0
    draft = dx.get_for_document(cx, doc_id)
    assert draft["status"] == "ai_draft"
    assert draft["narrative_md"] == "Your panel showed a few things."
    assert draft["facts"][0]["fact_key"] == "on_areds2"
    assert draft["unstructured"][0]["label"] == "HbA1c"
    assert cd.get(cx, doc_id)["extract_status"] == "drafted"


def test_extract_drops_an_ungrounded_attribute_from_the_draft():
    """The fabrication guard: an invented diagnosis never reaches Glen's review."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [
                   {"field": "conditions", "value": "glaucoma",
                    "source_quote": "Assessment: glaucoma"},
                   {"field": "conditions", "value": "lupus",
                    "source_quote": "Assessment: lupus"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                              source_text=SOURCE)
    assert out["dropped"] == 1
    vals = [a["value"] for a in dx.get_for_document(cx, doc_id)["attributes"]]
    assert "lupus" not in [v.lower() for v in vals]


def test_production_path_verifies_against_the_models_transcription():
    """No source_text injected — the real call path. Quotes are checked against
    the model's `document_text`, NOT its narrative."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"document_text": SOURCE, "narrative_md": "n",
               "facts": [], "unstructured": [],
               "attributes": [
                   {"field": "conditions", "value": "glaucoma",
                    "source_quote": "Assessment: glaucoma"},
                   {"field": "conditions", "value": "lupus",
                    "source_quote": "Assessment: lupus"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload))
    assert out["dropped"] == 1
    vals = [a["value"] for a in dx.get_for_document(cx, doc_id)["attributes"]]
    assert [v.lower() for v in vals] == ["glaucoma"]


def test_narrative_alone_cannot_validate_a_quote():
    """The guard must not be self-validating: a diagnosis invented into the
    narrative, absent from the transcription, is still dropped."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"document_text": "Routine visit. Nothing remarkable.",
               "narrative_md": "Assessment: lupus was noted.",
               "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions", "value": "lupus",
                               "source_quote": "Assessment: lupus"}]}
    de.extract_document(cx, doc_id, call_model=_fake_model(payload))
    assert dx.get_for_document(cx, doc_id)["attributes"] == []


def test_missing_transcription_fails_closed():
    """A model that omits document_text yields an EMPTY draft, never an
    unchecked one."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions", "value": "glaucoma",
                               "source_quote": "Assessment: glaucoma"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload))
    assert out["dropped"] == 1
    assert dx.get_for_document(cx, doc_id)["attributes"] == []


def test_extract_canonicalizes_attribute_values_before_drafting():
    """Glen reviews the canonical form he will actually be approving."""
    cx = _cx()
    from dashboard import canonical_tags as ct
    ct.init_tables(cx)
    cx.execute("INSERT INTO canonical_vocab(field, alias_norm, canonical) "
               "VALUES(?,?,?)", ("conditions", "glaucoma", "Glaucoma (POAG)"))
    cx.commit()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions", "value": "glaucoma",
                               "source_quote": "Assessment: glaucoma"}]}
    de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                        source_text=SOURCE)
    assert dx.get_for_document(cx, doc_id)["attributes"][0]["value"] == "Glaucoma (POAG)"


def test_extract_drops_attributes_with_an_out_of_vocabulary_field():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [{"field": "not_a_field", "value": "x",
                               "source_quote": "Assessment: glaucoma"}]}
    de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                        source_text=SOURCE)
    assert dx.get_for_document(cx, doc_id)["attributes"] == []


def test_extract_marks_failed_and_writes_no_draft_when_the_model_raises():
    cx = _cx()
    doc_id = _doc(cx)

    def boom(blob, content_type):
        raise RuntimeError("api down")

    assert de.extract_document(cx, doc_id, call_model=boom, source_text=SOURCE) is None
    assert cd.get(cx, doc_id)["extract_status"] == "failed"
    assert dx.get_for_document(cx, doc_id) is None


def test_extract_marks_failed_on_unparseable_model_output():
    cx = _cx()
    doc_id = _doc(cx)
    assert de.extract_document(cx, doc_id, call_model=_fake_model("not a dict"),
                               source_text=SOURCE) is None
    assert cd.get(cx, doc_id)["extract_status"] == "failed"


def test_run_pending_processes_only_pending_documents():
    cx = _cx()
    a = _doc(cx, blob=b"one")
    b = _doc(cx, blob=b"two")
    cd.set_extract_status(cx, b, "drafted")
    payload = {"narrative_md": "n", "attributes": [], "facts": [],
               "unstructured": []}
    assert de.run_pending(cx, call_model=_fake_model(payload)) == 1
    assert cd.get(cx, a)["extract_status"] == "drafted"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_document_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.document_extract'`

- [ ] **Step 3: Write the implementation**

```python
# dashboard/document_extract.py
"""Extract structured clinical proposals + a client narrative from an uploaded
document, and write them as a DRAFT.

This module writes NOTHING to a live clinical store. That is the whole point of
the review gate: the console approval route is the only thing that writes
person_attributes / client_facts.

The fabrication guard is structural. Every extracted item must carry a
`source_quote` that actually appears in the document; anything else is dropped
before Glen ever sees it. Prompted output alone is not trusted.
"""
import json
import os
import re

_MODEL = os.environ.get("DOC_EXTRACT_MODEL", "claude-opus-4-8")

_PROMPT = (
    "You are reading a patient's medical record. Extract ONLY what the document "
    "actually states. Never infer, never generalize, never add a diagnosis that "
    "is not written down.\n\n"
    "Return STRICT JSON with these keys:\n"
    '  "document_text": a VERBATIM transcription of all text in the document, '
    "exactly as written. This is what every source_quote is checked against, so "
    "a quote that is not present here is discarded.\n"
    '  "narrative_md": a warm, plain-language summary for the patient '
    "(2-4 short paragraphs, markdown, no headings). Explain what the document "
    "says in everyday words. Do not give advice or recommend treatment.\n"
    '  "attributes": [{"field": one of "tags"|"conditions"|"terrain_concerns"'
    '|"body_systems"|"challenges"|"goals", "value": str, "source_quote": str}]\n'
    '  "facts": [{"fact_key": str, "value": true|false, "source_quote": str}]\n'
    '  "unstructured": [{"label": str, "value": str, "source_quote": str}] '
    "for lab results with numeric values and medications.\n\n"
    "EVERY item MUST include a source_quote copied VERBATIM from the document. "
    "An item without a verbatim quote will be discarded. No markdown fences, no "
    "prose outside the JSON."
)


def _norm_text(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def verify_quotes(items, source_text):
    """Split `items` into (kept, dropped) by whether each item's source_quote
    actually occurs in `source_text`. Whitespace- and case-insensitive, so the
    model is not punished for reflowing a line."""
    hay = _norm_text(source_text)
    kept, dropped = [], []
    for it in items or []:
        q = _norm_text((it or {}).get("source_quote"))
        (kept if (q and q in hay) else dropped).append(it)
    return kept, dropped


def _default_call_model(blob, content_type):
    """The real Anthropic call. Lazy-imported so tests that inject call_model
    never pull the SDK."""
    import base64
    import anthropic
    cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    b64 = base64.standard_b64encode(blob).decode("ascii")
    if (content_type or "").lower() == "application/pdf":
        doc = {"type": "document",
               "source": {"type": "base64", "media_type": "application/pdf",
                          "data": b64}}
    else:
        doc = {"type": "image",
               "source": {"type": "base64", "media_type": content_type,
                          "data": b64}}
    resp = cli.messages.create(
        model=_MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": [doc, {"type": "text",
                                                     "text": _PROMPT}]}])
    text = resp.content[0].text.strip()
    if text.startswith("```"):                     # tolerate accidental fences
        text = text.split("```", 2)[1]
        if text.startswith("json\n"):
            text = text[5:]
    return json.loads(text)


def _source_text_for(payload):
    """The haystack the guard checks quotes against: the model's VERBATIM
    transcription of the document.

    Deliberately NOT the narrative. Checking the model's quotes against the
    model's own summary would make the guard vacuous — an invented diagnosis
    mentioned in the narrative would validate itself. Verifying against a
    separate transcription field means a fabricated quote must also be
    fabricated into the transcription, a meaningfully higher bar.

    Missing transcription returns "" and the guard FAILS CLOSED: with an empty
    haystack every quote fails and every item is dropped, so a model that omits
    the field yields an empty draft rather than an unchecked one.
    """
    return payload.get("document_text") or ""


def extract_document(cx, doc_id, call_model=None, source_text=None):
    """Run extraction for one document and write its draft. Returns a small
    summary dict, or None when extraction failed (document marked 'failed')."""
    from dashboard import client_documents as _cd
    from dashboard import document_extractions as _dx
    from dashboard import canonical_tags as _ct

    doc = _cd.get(cx, doc_id)
    if not doc:
        return None
    call = call_model or _default_call_model
    try:
        payload = call(doc["blob"], doc["content_type"])
        if not isinstance(payload, dict):
            raise ValueError("model did not return an object")
    except Exception as e:                          # noqa: BLE001 - any failure
        print(f"[documents] extract failed for {doc_id}: {e!r}", flush=True)
        _cd.set_extract_status(cx, doc_id, "failed")
        return None

    hay = source_text if source_text is not None else _source_text_for(payload)
    attrs, d1 = verify_quotes(payload.get("attributes"), hay)
    facts, d2 = verify_quotes(payload.get("facts"), hay)
    unstruct, d3 = verify_quotes(payload.get("unstructured"), hay)

    # Drop out-of-vocabulary fields, then canonicalize so Glen reviews the exact
    # value that would be written.
    _ct.init_tables(cx)
    clean_attrs = []
    for a in attrs:
        field = (a.get("field") or "").strip()
        if field not in _ct.ALL_FIELDS:
            d1.append(a)
            continue
        value = _ct.resolve(cx, field, a.get("value"))
        if not value:
            d1.append(a)
            continue
        clean_attrs.append({"field": field, "value": value,
                            "source_quote": a.get("source_quote", "")})

    _dx.put_draft(cx, doc_id, doc["email"], payload.get("narrative_md") or "",
                  clean_attrs, facts, unstruct, _MODEL)
    _cd.set_extract_status(cx, doc_id, "drafted")
    dropped = len(d1) + len(d2) + len(d3)
    if dropped:
        print(f"[documents] dropped {dropped} ungrounded item(s) for {doc_id}",
              flush=True)
    return {"document_id": doc_id, "kept": len(clean_attrs) + len(facts)
            + len(unstruct), "dropped": dropped}


def run_pending(cx, limit=5, call_model=None):
    """Process up to `limit` pending documents. Returns how many drafted."""
    from dashboard import client_documents as _cd
    done = 0
    for doc in _cd.pending(cx, limit=limit):
        if extract_document(cx, doc["id"], call_model=call_model):
            done += 1
    return done
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_document_extract.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/document_extract.py tests/test_document_extract.py
git commit -m "feat(documents): extraction pipeline with source-quote fabrication guard"
```

---

### Task 5: Approve / reject — the only writer of live stores

**Files:**
- Modify: `app.py` (add after the console upload route from Task 3)
- Test: `tests/test_document_approve.py`

**Interfaces:**
- Consumes: `document_extractions.get_for_document/confirm/reject` (Task 2), `canonical_tags.set_attr`, `client_facts.set_fact`.
- Produces:
  - `POST /api/console/client-document/<int:doc_id>/approve` — JSON body `{"narrative_md": str, "attributes": [int], "facts": [int], "reviewed_by": str}`, where the two index lists are the positions Glen left checked.
  - `POST /api/console/client-document/<int:doc_id>/reject`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_document_approve.py
import importlib, sqlite3, sys
from pathlib import Path
from dashboard import client_documents as cd
from dashboard import document_extractions as dx


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONSOLE_SECRET", "test-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed(appmod, email="c@x.com"):
    cx = sqlite3.connect(appmod.LOG_DB)
    doc_id = cd.put(cx, email, b"%PDF", "labs.pdf", "application/pdf", "console")["id"]
    dx.put_draft(cx, doc_id, email, "Draft narrative.",
                 attributes=[{"field": "conditions", "value": "Glaucoma",
                              "source_quote": "q"},
                             {"field": "body_systems", "value": "Liver",
                              "source_quote": "q"}],
                 facts=[{"fact_key": "on_areds2", "value": True,
                         "source_quote": "q"}],
                 unstructured=[{"label": "HbA1c", "value": "6.4",
                                "source_quote": "q"}],
                 model="m")
    cx.commit(); cx.close()
    return doc_id


def _approve(appmod, doc_id, **body):
    payload = {"narrative_md": "Final narrative.", "attributes": [0, 1],
               "facts": [0], "reviewed_by": "glen"}
    payload.update(body)
    return appmod.app.test_client().post(
        f"/api/console/client-document/{doc_id}/approve?key=test-secret",
        json=payload)


def test_approve_writes_checked_attributes_with_document_provenance(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    assert _approve(appmod, doc_id).status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB)
    rows = cx.execute("SELECT field, value, source FROM person_attributes "
                      "WHERE email='c@x.com' ORDER BY field").fetchall()
    assert ("body_systems", "Liver", f"document:{doc_id}") in rows
    assert ("conditions", "Glaucoma", f"document:{doc_id}") in rows


def test_approve_writes_checked_boolean_facts(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    _approve(appmod, doc_id)
    from dashboard import client_facts as cf
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    assert cf.get_facts(cx, "c@x.com")["on_areds2"] is True


def test_approve_does_not_write_client_conditions(tmp_path, monkeypatch):
    """Guards the single eye-condition support-program override from being
    clobbered by extracted diagnoses."""
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    from dashboard import client_conditions as cc
    cx = sqlite3.connect(appmod.LOG_DB)
    cc.init_table(cx)
    cc.set(cx, "c@x.com", "glaucoma-support", "operator")
    cx.commit(); cx.close()
    _approve(appmod, doc_id)
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cc.get(cx, "c@x.com") == "glaucoma-support"


def test_unchecked_items_are_not_written(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    _approve(appmod, doc_id, attributes=[0], facts=[])
    cx = sqlite3.connect(appmod.LOG_DB)
    fields = [r[0] for r in cx.execute(
        "SELECT field FROM person_attributes WHERE email='c@x.com'").fetchall()]
    assert fields == ["conditions"]
    n = cx.execute("SELECT COUNT(*) FROM client_facts "
                   "WHERE email='c@x.com'").fetchone()[0]
    assert n == 0


def test_approve_saves_the_edited_narrative_and_confirms(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    _approve(appmod, doc_id, narrative_md="Glen's edited words.")
    cx = sqlite3.connect(appmod.LOG_DB)
    got = dx.get_for_document(cx, doc_id)
    assert got["status"] == "confirmed"
    assert got["narrative_md"] == "Glen's edited words."
    assert got["reviewed_by"] == "glen"


def test_second_approve_is_an_idempotent_noop(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    assert _approve(appmod, doc_id).status_code == 200
    r2 = _approve(appmod, doc_id, narrative_md="second")
    assert r2.status_code == 200 and r2.get_json()["already"] is True
    cx = sqlite3.connect(appmod.LOG_DB)
    assert dx.get_for_document(cx, doc_id)["narrative_md"] == "Final narrative."


def test_reject_discards_the_draft_and_keeps_the_file(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = appmod.app.test_client().post(
        f"/api/console/client-document/{doc_id}/reject?key=test-secret",
        json={"reviewed_by": "glen"})
    assert r.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB)
    assert dx.get_for_document(cx, doc_id)["status"] == "rejected"
    assert cd.get(cx, doc_id) is not None
    n = cx.execute("SELECT COUNT(*) FROM person_attributes").fetchone()[0]
    assert n == 0


def test_approve_requires_the_console_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = appmod.app.test_client().post(
        f"/api/console/client-document/{doc_id}/approve", json={})
    assert r.status_code == 401


def test_approve_404s_for_a_document_with_no_draft(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert _approve(appmod, 4242).status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_document_approve.py -v`
Expected: FAIL — approve/reject requests 404 (routes do not exist yet)

- [ ] **Step 3: Write the implementation**

Append to `app.py` after `api_console_client_document_upload`:

```python
def _console_guard():
    """None when the caller is authorized, else a (body, status) tuple."""
    if CONSOLE_SECRET:
        key = _present_console_key()
        if key != CONSOLE_SECRET and not _owner_token_ok(key):
            return {"ok": False, "error": "Unauthorized"}, 401
    return None


@app.route("/api/console/client-document/<int:doc_id>/approve", methods=["POST"])
def api_console_client_document_approve(doc_id):
    """The ONE gate. Writes the checked proposals to the live stores and
    publishes the narrative to the client's portal, in a single request.

    Deliberately does NOT write client_conditions: that table is the single
    eye-condition support-program override and has its own console control.
    """
    guard = _console_guard()
    if guard:
        return jsonify(guard[0]), guard[1]
    from dashboard import document_extractions as _dx
    from dashboard import canonical_tags as _ct
    from dashboard import client_facts as _cf
    body = request.get_json(silent=True) or {}
    keep_attrs = set(body.get("attributes") or [])
    keep_facts = set(body.get("facts") or [])
    reviewed_by = (body.get("reviewed_by") or "console").strip()
    with _db_lock, db.connect(LOG_DB) as cx:
        draft = _dx.get_for_document(cx, doc_id)
        if not draft:
            return jsonify({"ok": False, "error": "not found"}), 404
        narrative = body.get("narrative_md")
        if narrative is None:
            narrative = draft["narrative_md"]
        # confirm() only flips an ai_draft, so a repeat approval writes nothing.
        if not _dx.confirm(cx, draft["id"], narrative, reviewed_by):
            return jsonify({"ok": True, "already": True,
                            "status": draft["status"]}), 200
        written = {"attributes": 0, "facts": 0}
        for i, a in enumerate(draft["attributes"]):
            if i not in keep_attrs:
                continue
            if _ct.set_attr(cx, draft["email"], a.get("field"), a.get("value"),
                            source=f"document:{doc_id}"):
                written["attributes"] += 1
        for i, f in enumerate(draft["facts"]):
            if i not in keep_facts:
                continue
            _cf.set_fact(cx, draft["email"], f.get("fact_key"),
                         bool(f.get("value")))
            written["facts"] += 1
    return jsonify({"ok": True, "already": False, "written": written}), 200


@app.route("/api/console/client-document/<int:doc_id>/reject", methods=["POST"])
def api_console_client_document_reject(doc_id):
    """Discard the AI's reading. The uploaded file itself is kept."""
    guard = _console_guard()
    if guard:
        return jsonify(guard[0]), guard[1]
    from dashboard import document_extractions as _dx
    body = request.get_json(silent=True) or {}
    with _db_lock, db.connect(LOG_DB) as cx:
        draft = _dx.get_for_document(cx, doc_id)
        if not draft:
            return jsonify({"ok": False, "error": "not found"}), 404
        ok = _dx.reject(cx, draft["id"],
                        (body.get("reviewed_by") or "console").strip())
    return jsonify({"ok": True, "changed": ok}), 200
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_document_approve.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_document_approve.py
git commit -m "feat(documents): single console approval writes attributes, facts, narrative"
```

---

### Task 6: Portal read API — list + raw-file download

**Files:**
- Modify: `app.py` (add after the portal upload route from Task 3)
- Test: `tests/test_portal_documents_api.py`

**Interfaces:**
- Consumes: `client_documents.list_for_email/get_for_email` (Task 1), `document_extractions.get_for_document` (Task 2).
- Produces:
  - `GET /api/portal/<token>/documents` → `{"enabled": bool, "items": [{id, filename, uploaded_at, status, file_url, narrative_md}]}`
  - `GET /api/portal/<token>/documents/<int:doc_id>/file` → raw bytes or 404

`status` is `"under_review"` until the draft is confirmed, then `"ready"`. `narrative_md` is `""` unless confirmed.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_portal_documents_api.py
import importlib, sqlite3, sys
from pathlib import Path
from dashboard import client_documents as cd
from dashboard import document_extractions as dx


def _app(tmp_path, monkeypatch, hub="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PORTAL_HUB_ENABLED", hub)
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed(appmod, email, blob=b"%PDF bytes", confirm=False):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    doc_id = cd.put(cx, email, blob, "labs.pdf", "application/pdf", "console")["id"]
    eid = dx.put_draft(cx, doc_id, email, "Your summary.", [], [], [], "m")
    if confirm:
        dx.confirm(cx, eid, "Your summary.", "glen")
    cx.commit(); cx.close()
    return tok, doc_id


def test_before_approval_status_is_under_review_and_no_narrative(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed(appmod, "c@x.com")
    body = appmod.app.test_client().get(f"/api/portal/{tok}/documents").get_json()
    assert body["enabled"] is True
    it = body["items"][0]
    assert it["status"] == "under_review"
    assert it["narrative_md"] == ""
    assert it["file_url"] == f"/api/portal/{tok}/documents/{doc_id}/file"


def test_after_approval_narrative_is_visible(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, _ = _seed(appmod, "c@x.com", confirm=True)
    it = appmod.app.test_client().get(
        f"/api/portal/{tok}/documents").get_json()["items"][0]
    assert it["status"] == "ready"
    assert it["narrative_md"] == "Your summary."


def test_payload_never_exposes_attributes_or_facts(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed(appmod, "c@x.com", confirm=True)
    cx = sqlite3.connect(appmod.LOG_DB)
    dx.put_draft(cx, doc_id, "c@x.com", "n",
                 [{"field": "conditions", "value": "Glaucoma", "source_quote": "q"}],
                 [{"fact_key": "on_areds2", "value": True, "source_quote": "q"}],
                 [{"label": "HbA1c", "value": "6.4", "source_quote": "q"}], "m")
    cx.commit(); cx.close()
    raw = appmod.app.test_client().get(f"/api/portal/{tok}/documents").get_data(as_text=True)
    assert "Glaucoma" not in raw and "on_areds2" not in raw and "HbA1c" not in raw


def test_owner_can_download_their_own_raw_file(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed(appmod, "c@x.com", blob=b"%PDF-1.4 real bytes")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert r.status_code == 200
    assert r.data == b"%PDF-1.4 real bytes"
    assert r.headers["Cache-Control"] == "private, no-store"


def test_another_token_cannot_download_the_file(tmp_path, monkeypatch):
    """Cross-token isolation — the test that matters most."""
    appmod = _app(tmp_path, monkeypatch)
    _, doc_id = _seed(appmod, "owner@x.com")
    other_tok, _ = _seed(appmod, "other@x.com", blob=b"other bytes")
    r = appmod.app.test_client().get(
        f"/api/portal/{other_tok}/documents/{doc_id}/file")
    assert r.status_code == 404


def test_another_token_does_not_see_the_document_in_its_list(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _seed(appmod, "owner@x.com")
    other_tok, _ = _seed(appmod, "other@x.com", blob=b"other bytes")
    items = appmod.app.test_client().get(
        f"/api/portal/{other_tok}/documents").get_json()["items"]
    assert len(items) == 1


def test_enabled_false_when_hub_flag_off(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, hub="0")
    tok, _ = _seed(appmod, "c@x.com")
    assert appmod.app.test_client().get(
        f"/api/portal/{tok}/documents").get_json()["enabled"] is False


def test_unknown_token_404s(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert appmod.app.test_client().get("/api/portal/nope/documents").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_portal_documents_api.py -v`
Expected: FAIL — the GET routes 404

- [ ] **Step 3: Write the implementation**

Append to `app.py` after `api_portal_document_upload`:

```python
@app.route("/api/portal/<token>/documents", methods=["GET"])
def api_portal_documents(token):
    """The token owner's uploaded records. `enabled` mirrors the hub flag so the
    My Records tile stays dark until the flag flips.

    The client sees their own file and — once Glen has approved it — the
    narrative. Extracted attributes, facts, and labs are NEVER included.
    """
    from dashboard import client_portal as _cp
    from dashboard import client_documents as _cd
    from dashboard import document_extractions as _dx
    with db.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"error": "not found"}), 404
        email = (portal.get("email") or "").strip().lower()
        docs = _cd.list_for_email(cx, email) if email else []
        items = []
        for d in docs:
            draft = _dx.get_for_document(cx, d["id"])
            ready = bool(draft and draft["status"] == "confirmed")
            items.append({
                "id": d["id"],
                "filename": d["filename"],
                "uploaded_at": d["uploaded_at"],
                "status": "ready" if ready else "under_review",
                "file_url": f"/api/portal/{token}/documents/{d['id']}/file",
                "narrative_md": draft["narrative_md"] if ready else "",
            })
    return jsonify({"enabled": _PORTAL_HUB_ENABLED, "items": items})


@app.route("/api/portal/<token>/documents/<int:doc_id>/file", methods=["GET"])
def api_portal_document_file(token, doc_id):
    """Stream the token owner's OWN document. Resolved through get_for_email so
    a token can never fetch another client's file."""
    from dashboard import client_portal as _cp
    from dashboard import client_documents as _cd
    with db.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
        email = (portal.get("email") or "").strip().lower() if portal else ""
        doc = _cd.get_for_email(cx, doc_id, email) if email else None
    if not doc:
        return Response("", status=404)
    resp = Response(doc["blob"],
                    mimetype=doc["content_type"] or "application/octet-stream")
    resp.headers["Cache-Control"] = "private, no-store"
    resp.headers["Content-Disposition"] = (
        f'inline; filename="{(doc["filename"] or "document").replace(chr(34), "")}"')
    return resp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_portal_documents_api.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_portal_documents_api.py
git commit -m "feat(documents): token-scoped portal list + raw-file download"
```

---

### Task 7: Portal "My Records" tile

**Files:**
- Create: `static/js/portal-documents.js`
- Modify: `static/client-portal.html` (mount div beside `#portal-library-mount`, line ~549; script tag beside `portal-library.js`, line ~552)
- Test: `tests/test_portal_documents_tile.js`

**Interfaces:**
- Consumes: `GET /api/portal/<token>/documents` from Task 6.
- Produces: `renderDocuments(items) -> string` (exported for the node test, mirroring `portal-library.js`).

**Note on the JS test:** `ci/run-tests.sh` runs pytest only, so `.js` tests are not CI-gated. Run this one by hand with `node`. The Python API tests in Task 6 are what gate the behavior.

- [ ] **Step 1: Write the failing test**

```javascript
// tests/test_portal_documents_tile.js
// Run: node tests/test_portal_documents_tile.js
const assert = require('assert');
const { renderDocuments } = require('../static/js/portal-documents.js');

// empty -> no tile at all
assert.strictEqual(renderDocuments([]), '');
assert.strictEqual(renderDocuments(null), '');

// under review -> shows the file link and the review line, no narrative
const pending = renderDocuments([{
  id: 1, filename: 'labs.pdf', uploaded_at: '2026-07-23T00:00:00Z',
  status: 'under_review', file_url: '/api/portal/t/documents/1/file',
  narrative_md: ''
}]);
assert.ok(pending.includes('My Records'));
assert.ok(pending.includes('/api/portal/t/documents/1/file'));
assert.ok(pending.includes('Received — under review'));

// ready -> shows the narrative
const ready = renderDocuments([{
  id: 2, filename: 'panel.pdf', uploaded_at: '2026-07-23T00:00:00Z',
  status: 'ready', file_url: '/api/portal/t/documents/2/file',
  narrative_md: 'Your panel looked at three things.'
}]);
assert.ok(ready.includes('Your panel looked at three things.'));
assert.ok(!ready.includes('Received — under review'));

// filenames are escaped, never injected
const evil = renderDocuments([{
  id: 3, filename: '<img src=x onerror=alert(1)>', uploaded_at: '',
  status: 'under_review', file_url: '/f', narrative_md: ''
}]);
assert.ok(!evil.includes('<img src=x'));
assert.ok(evil.includes('&lt;img'));

console.log('ok - portal documents tile');
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node tests/test_portal_documents_tile.js`
Expected: FAIL — `Cannot find module '../static/js/portal-documents.js'`

- [ ] **Step 3: Write the implementation**

```javascript
// static/js/portal-documents.js
// My Records tile: the client's own uploaded medical records, plus the
// plain-language narrative once Glen has reviewed it.
// Consumes GET /api/portal/<token>/documents ->
//   {enabled, items:[{id,filename,uploaded_at,status,file_url,narrative_md}]}
// The payload deliberately carries no extracted attributes, facts, or labs.
function escapeHtmlDoc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}

function renderDocuments(items) {
  if (!items || !items.length) return '';
  const rows = items.map(function (it) {
    const body = it.status === 'ready'
      ? '<p class="doc-narrative">' + escapeHtmlDoc(it.narrative_md) + '</p>'
      : '<p class="doc-pending">Received — under review</p>';
    return '<li class="doc-item">' +
      '<a class="doc-file" href="' + escapeHtmlDoc(it.file_url) +
        '" target="_blank" rel="noopener">' + escapeHtmlDoc(it.filename) + '</a>' +
      body +
    '</li>';
  }).join('');
  return '<section class="portal-documents"><h2>My Records</h2>' +
         '<ul class="doc-list">' + rows + '</ul></section>';
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { renderDocuments: renderDocuments };
}

// Browser: fetch + mount. Token is the last path segment of /portal/<token>.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('portal-documents-mount');
    if (!mount) return;
    var m = location.pathname.match(/\/portal\/([^\/]+)/);
    if (!m) return;
    fetch('/api/portal/' + m[1] + '/documents')
      .then(function (r) { return r.ok ? r.json() : {enabled: false, items: []}; })
      .then(function (d) { mount.innerHTML = d.enabled ? renderDocuments(d.items) : ''; })
      .catch(function () {});
  });
}
```

In `static/client-portal.html`, add the mount immediately after the existing `<div id="portal-library-mount"></div>`:

```html
  <!-- My Records tile. Sibling of #app, filled by static/js/portal-documents.js's
       own fetch + mount, same pattern as the library tile above. -->
  <div id="portal-documents-mount"></div>
```

and the script immediately after the existing `portal-library.js` tag:

```html
<script src="/static/js/portal-documents.js"></script>
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node tests/test_portal_documents_tile.js`
Expected: `ok - portal documents tile`

- [ ] **Step 5: Commit**

```bash
git add static/js/portal-documents.js static/client-portal.html tests/test_portal_documents_tile.js
git commit -m "feat(documents): My Records portal tile"
```

---

### Task 8: Console review section

**Files:**
- Modify: `app.py` — add `GET /api/console/client-documents` returning the review payload for one client.
- Modify: `static/client-portal.html` — none. Console UI lives in the console page.
- Test: extend `tests/test_document_approve.py`

**Interfaces:**
- Consumes: `client_documents.list_for_email` (Task 1), `document_extractions.get_for_document` (Task 2).
- Produces: `GET /api/console/client-documents?email=<email>` → `{"ok": true, "items": [{id, filename, uploaded_at, extract_status, file_url, draft}]}` where `draft` is the full extraction dict (attributes, facts, unstructured, narrative, each with `source_quote`) or `null`.

This is the payload the console screen renders: the raw file, the checkboxes with their quotes, and the editable narrative. It is a separate task from Task 5 because a reviewer could reasonably accept the approval semantics while rejecting the review payload's shape.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_document_approve.py`:

```python
def test_console_review_payload_carries_quotes_and_file_url(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = appmod.app.test_client().get(
        "/api/console/client-documents?email=c@x.com&key=test-secret")
    assert r.status_code == 200
    it = r.get_json()["items"][0]
    assert it["id"] == doc_id
    assert it["filename"] == "labs.pdf"
    assert it["file_url"] == f"/admin/client-document?id={doc_id}"
    d = it["draft"]
    assert d["narrative_md"] == "Draft narrative."
    assert d["attributes"][0]["source_quote"] == "q"
    assert d["facts"][0]["fact_key"] == "on_areds2"
    assert d["unstructured"][0]["label"] == "HbA1c"


def test_console_review_payload_draft_is_null_when_not_extracted(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB)
    cd.put(cx, "c@x.com", b"%PDF", "raw.pdf", "application/pdf", "console")
    cx.commit(); cx.close()
    it = appmod.app.test_client().get(
        "/api/console/client-documents?email=c@x.com&key=test-secret"
    ).get_json()["items"][0]
    assert it["draft"] is None


def test_console_review_requires_the_console_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    r = appmod.app.test_client().get("/api/console/client-documents?email=c@x.com")
    assert r.status_code == 401


def test_console_document_file_serves_bytes_with_the_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = appmod.app.test_client().get(f"/admin/client-document?id={doc_id}&key=test-secret")
    assert r.status_code == 200 and r.data == b"%PDF"


def test_console_document_file_requires_the_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    assert appmod.app.test_client().get(
        f"/admin/client-document?id={doc_id}").status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_document_approve.py -k console_review -v`
Expected: FAIL — the GET routes 404

- [ ] **Step 3: Write the implementation**

Append to `app.py` after `api_console_client_document_reject`:

```python
@app.route("/api/console/client-documents", methods=["GET"])
def api_console_client_documents():
    """Review payload for one client's documents: the file, the AI's proposals
    with their source quotes, and the editable narrative. This is what the
    console Documents section renders."""
    guard = _console_guard()
    if guard:
        return jsonify(guard[0]), guard[1]
    from dashboard import client_documents as _cd
    from dashboard import document_extractions as _dx
    email = (request.args.get("email") or "").strip().lower()
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400
    with db.connect(LOG_DB) as cx:
        docs = _cd.list_for_email(cx, email)
        items = []
        for d in docs:
            items.append({
                "id": d["id"], "filename": d["filename"],
                "uploaded_at": d["uploaded_at"], "source": d["source"],
                "extract_status": d["extract_status"],
                "file_url": f"/admin/client-document?id={d['id']}",
                "draft": _dx.get_for_document(cx, d["id"]),
            })
    return jsonify({"ok": True, "items": items})


@app.route("/admin/client-document", methods=["GET"])
def admin_client_document_file():
    """Console-gated raw document viewer (PHI). Serves the bytes for review."""
    guard = _console_guard()
    if guard:
        return jsonify(guard[0]), guard[1]
    from dashboard import client_documents as _cd
    try:
        doc_id = int(request.args.get("id") or 0)
    except ValueError:
        return jsonify({"error": "bad id"}), 400
    with db.connect(LOG_DB) as cx:
        doc = _cd.get(cx, doc_id) if doc_id else None
    if not doc:
        return jsonify({"error": "not found"}), 404
    resp = Response(doc["blob"],
                    mimetype=doc["content_type"] or "application/octet-stream")
    resp.headers["Cache-Control"] = "private, no-store"
    return resp
```

- [ ] **Step 4: Run the full task test file**

Run: `python3 -m pytest tests/test_document_approve.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_document_approve.py
git commit -m "feat(documents): console review payload + gated raw-document viewer"
```

---

### Task 8b: Console Documents review section (the screen Glen actually uses)

**Files:**
- Modify: `static/console-client.html` — add a `Documents` `<details>` section beside the existing `sec-recs` (line ~53) and a `renderDocuments` function in the inline `<script>`.
- Test: `tests/test_console_documents_render.js`

**Interfaces:**
- Consumes: `GET /api/console/client-documents?email=` (Task 8), `POST /api/console/client-document/<id>/approve` and `/reject` (Task 5).
- Produces: `renderDocumentsHtml(items, consoleKey) -> string`, defined in the inline script and also exported for the node test via a guarded `module.exports` in a small companion file.

**Why this task exists:** without it nothing can be approved, so no facts are ever written and no client ever sees a narrative. The endpoints from Tasks 5 and 8 are inert until this ships.

**Page conventions to follow** (read `static/console-client.html` before writing):
- Sections are `<details id="sec-X"><summary>Name <span class="head-note" id="h-X"></span></summary><div class="sec-body" id="b-X"></div></details>`.
- `esc(s)` escapes text, `escAttr(s)` escapes attribute values, `key()` returns the console key, `hdr()` returns `{"X-Console-Key": key()}`, `qsEmail()` returns the current client's email.
- Section renderers write `innerHTML` into their `#b-X` host.

- [ ] **Step 1: Write the failing test**

```javascript
// tests/test_console_documents_render.js
// Run: node tests/test_console_documents_render.js
const assert = require('assert');
const { renderDocumentsHtml } = require('../static/js/console-documents.js');

const ITEM = {
  id: 5, filename: 'labs.pdf', uploaded_at: '2026-07-23T00:00:00Z',
  source: 'console', extract_status: 'drafted',
  file_url: '/admin/client-document?id=5',
  draft: {
    id: 9, status: 'ai_draft', narrative_md: 'Draft narrative.',
    attributes: [{ field: 'conditions', value: 'Glaucoma', source_quote: 'Assessment: glaucoma' }],
    facts: [{ fact_key: 'on_areds2', value: true, source_quote: 'taking AREDS2' }],
    unstructured: [{ label: 'HbA1c', value: '6.4', source_quote: 'HbA1c 6.4' }]
  }
};

const html = renderDocumentsHtml([ITEM], 'k');

// the raw file is reachable, with the console key attached
assert.ok(html.includes('/admin/client-document?id=5'));
// proposals render as PRE-CHECKED boxes carrying their index
assert.ok(/type=['"]checkbox['"][^>]*checked/.test(html));
assert.ok(html.includes('data-kind="attributes"'));
assert.ok(html.includes('data-idx="0"'));
// every proposal shows its source quote so an invention is visible at a glance
assert.ok(html.includes('Assessment: glaucoma'));
assert.ok(html.includes('taking AREDS2'));
// labs are shown but marked as not stored structurally
assert.ok(html.includes('HbA1c'));
assert.ok(/not stored/i.test(html));
// the narrative is editable
assert.ok(html.includes('<textarea'));
assert.ok(html.includes('Draft narrative.'));
// both actions are present
assert.ok(/Approve/.test(html) && /Reject/.test(html));

// a confirmed draft shows as reviewed, with no approve button
const done = renderDocumentsHtml([Object.assign({}, ITEM, {
  draft: Object.assign({}, ITEM.draft, { status: 'confirmed' })
})], 'k');
assert.ok(/Approved/i.test(done));
assert.ok(!/>Approve</.test(done));

// a document with no draft yet says so and offers no checkboxes
const raw = renderDocumentsHtml([{
  id: 6, filename: 'raw.pdf', uploaded_at: '', source: 'console',
  extract_status: 'pending', file_url: '/admin/client-document?id=6', draft: null
}], 'k');
assert.ok(/awaiting extraction/i.test(raw));
assert.ok(!raw.includes('type="checkbox"'));

// empty state
assert.strictEqual(renderDocumentsHtml([], 'k'), '<p class="muted">No documents.</p>');

// filenames and quotes are escaped, never injected
const evil = renderDocumentsHtml([{
  id: 7, filename: '<img src=x onerror=alert(1)>', uploaded_at: '', source: 'console',
  extract_status: 'drafted', file_url: '/f', draft: {
    id: 1, status: 'ai_draft', narrative_md: '', attributes: [], facts: [],
    unstructured: []
  }
}], 'k');
assert.ok(!evil.includes('<img src=x'));
assert.ok(evil.includes('&lt;img'));

console.log('ok - console documents review render');
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node tests/test_console_documents_render.js`
Expected: FAIL — `Cannot find module '../static/js/console-documents.js'`

- [ ] **Step 3: Write the implementation**

Create `static/js/console-documents.js`:

```javascript
// static/js/console-documents.js
// Console Documents review section: the ONE screen where Glen turns an AI draft
// into live clinical data. Renders the raw file, every proposal beside the
// verbatim quote it came from, and the editable narrative.
//
// Loaded as a plain script on the console page (it defines globals) and also
// exported for the node render test.
function cdEsc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}

function cdProposalRow(kind, idx, label, quote) {
  return '<li class="cd-prop">' +
    '<label><input type="checkbox" checked data-kind="' + kind + '" ' +
      'data-idx="' + idx + '"> ' + cdEsc(label) + '</label>' +
    '<blockquote class="cd-quote">' + cdEsc(quote) + '</blockquote>' +
  '</li>';
}

function renderDocumentsHtml(items, consoleKey) {
  if (!items || !items.length) return '<p class="muted">No documents.</p>';
  return items.map(function (it) {
    var head = '<h3>' + cdEsc(it.filename) + ' ' +
      '<span class="pill">' + cdEsc(it.source) + '</span> ' +
      '<a href="' + cdEsc(it.file_url) + '&key=' +
        encodeURIComponent(consoleKey || '') +
        '" target="_blank" rel="noopener">open file</a></h3>';

    var d = it.draft;
    if (!d) {
      return '<section class="cd-doc" data-doc="' + it.id + '">' + head +
        '<p class="muted">Awaiting extraction (' +
        cdEsc(it.extract_status) + ').</p></section>';
    }
    if (d.status !== 'ai_draft') {
      return '<section class="cd-doc" data-doc="' + it.id + '">' + head +
        '<p class="muted">' +
        (d.status === 'confirmed' ? 'Approved' : 'Rejected') +
        (d.reviewed_by ? ' by ' + cdEsc(d.reviewed_by) : '') +
        '.</p></section>';
    }

    var attrs = (d.attributes || []).map(function (a, i) {
      return cdProposalRow('attributes', i, a.field + ': ' + a.value,
                           a.source_quote);
    }).join('');
    var facts = (d.facts || []).map(function (f, i) {
      return cdProposalRow('facts', i,
                           f.fact_key + ' = ' + (f.value ? 'yes' : 'no'),
                           f.source_quote);
    }).join('');
    var labs = (d.unstructured || []).map(function (u) {
      return '<li>' + cdEsc(u.label) + ': ' + cdEsc(u.value) +
        '<blockquote class="cd-quote">' + cdEsc(u.source_quote) +
        '</blockquote></li>';
    }).join('');

    return '<section class="cd-doc" data-doc="' + it.id + '">' + head +
      (attrs ? '<h4>Proposed attributes</h4><ul class="cd-props">' + attrs + '</ul>' : '') +
      (facts ? '<h4>Proposed facts</h4><ul class="cd-props">' + facts + '</ul>' : '') +
      (labs ? '<h4>Labs and medications <span class="muted">(not stored ' +
              'structurally — reference only)</span></h4><ul class="cd-labs">' +
              labs + '</ul>' : '') +
      '<h4>Client narrative</h4>' +
      '<textarea class="cd-narrative" rows="8">' + cdEsc(d.narrative_md) +
        '</textarea>' +
      '<p class="cd-actions">' +
        '<button class="cd-approve" data-doc="' + it.id + '">Approve</button> ' +
        '<button class="cd-reject" data-doc="' + it.id + '">Reject</button>' +
      '</p>' +
    '</section>';
  }).join('');
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { renderDocumentsHtml: renderDocumentsHtml };
}
```

In `static/console-client.html`, add the section after the `sec-recs` `<details>` (line ~53):

```html
    <details id="sec-docs"><summary>Documents <span class="head-note" id="h-docs"></span></summary>
      <div class="sec-body" id="b-docs"></div></details>
```

add the script tag beside the other page scripts (after line 31's `op-nav.js`):

```html
  <script src="/static/js/console-documents.js"></script>
```

and add to the inline `<script>`, wiring fetch, render, and the two actions:

```javascript
function loadDocuments(){
  var email=qsEmail(); if(!email) return;
  fetch("/api/console/client-documents?email="+encodeURIComponent(email),{headers:hdr()})
    .then(function(r){ return r.ok?r.json():{items:[]}; })
    .then(function(d){
      var items=d.items||[];
      document.getElementById("h-docs").textContent=items.length?String(items.length):"";
      document.getElementById("b-docs").innerHTML=renderDocumentsHtml(items,key());
      wireDocumentActions();
    }).catch(function(){});
}

function checkedIdx(section,kind){
  return Array.prototype.slice.call(
    section.querySelectorAll('input[data-kind="'+kind+'"]:checked')
  ).map(function(el){ return parseInt(el.getAttribute("data-idx"),10); });
}

function wireDocumentActions(){
  var host=document.getElementById("b-docs");
  host.querySelectorAll(".cd-approve").forEach(function(btn){
    btn.addEventListener("click",function(){
      var sec=btn.closest(".cd-doc"), id=btn.getAttribute("data-doc");
      btn.disabled=true;
      fetch("/api/console/client-document/"+id+"/approve",{
        method:"POST",
        headers:Object.assign({"Content-Type":"application/json"},hdr()),
        body:JSON.stringify({
          narrative_md: sec.querySelector(".cd-narrative").value,
          attributes: checkedIdx(sec,"attributes"),
          facts: checkedIdx(sec,"facts"),
          reviewed_by: "console"
        })
      }).then(function(){ loadDocuments(); })
        .catch(function(){ btn.disabled=false; });
    });
  });
  host.querySelectorAll(".cd-reject").forEach(function(btn){
    btn.addEventListener("click",function(){
      var id=btn.getAttribute("data-doc");
      btn.disabled=true;
      fetch("/api/console/client-document/"+id+"/reject",{
        method:"POST",
        headers:Object.assign({"Content-Type":"application/json"},hdr()),
        body:JSON.stringify({reviewed_by:"console"})
      }).then(function(){ loadDocuments(); })
        .catch(function(){ btn.disabled=false; });
    });
  });
}
```

Call `loadDocuments();` from the page's existing `load(email)` function, alongside the other section loaders.

- [ ] **Step 4: Run the test to verify it passes**

Run: `node tests/test_console_documents_render.js`
Expected: `ok - console documents review render`

- [ ] **Step 5: Verify the section renders in a real browser**

Start the app locally and open `/console/client?email=<a seeded client>&key=<console key>`. Confirm: the Documents section appears, the file link opens the PDF, checkboxes are pre-checked with quotes beneath them, editing the narrative and clicking Approve flips the section to "Approved", and a reload still shows it approved. A green node test is not evidence that the page renders — check the real page.

- [ ] **Step 6: Commit**

```bash
git add static/js/console-documents.js static/console-client.html tests/test_console_documents_render.js
git commit -m "feat(documents): console Documents review section"
```

---

### Task 9: Full-suite gate

**Files:** none changed unless the gate reports a new failure.

- [ ] **Step 1: Run the whole gated suite**

Run: `bash ci/run-tests.sh`
Expected: PASS. The script ratchets against `tests/known_failures.txt` and fails only on a NEW failure. It sets fake `PINECONE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `CONSOLE_SECRET` and unsets `DOPPLER_TOKEN`, which is what keeps a full run from touching real services.

- [ ] **Step 2: If a NEW failure appears, fix it**

Read the named test, fix the cause in the new code, and re-run. Do not add the test to `known_failures.txt` — the baseline only tightens.

Two failure modes to expect from this feature specifically:
- A test module that imports `app` failing at collection because a required env key is missing. Add the dummy key to that module's `_app()` helper, as the helpers in this plan already do.
- An existing test asserting a fixed count of routes or of `person_attributes` rows. Update the assertion to account for the new routes.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "test(documents): keep the CI ratchet green"
```

---

## Deferred to their own projects

Recorded here so nobody implements them by accident:

- **CTI-2** — wiring `canonical_tags.person_attributes` into the readers that feed analysis and remedy matching. Until it ships, approved *attributes* are written correctly with `source='document:<id>'` provenance but do not yet influence matching. Approved *boolean facts* do work today.
- **Multi-photo Body Map + alignment editor** — sub-project #2, unrelated code.
- **`client_photos.init_table` `BLOB` → `BYTEA`** — the same latent defect this plan avoids. It works today only because the table already exists, migrated; it would fail to create on a fresh Postgres database.
