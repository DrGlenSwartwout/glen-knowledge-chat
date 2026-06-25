# Email-Sourcing Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-collect supplier pricing/sourcing data from Glen's email into a `supplier_quotes` review queue; the console "Sourcing inbox" lets him approve a quote → writes an `ingredient_sources` row (via E2's `create_source`). Review-only; never auto-write.

**Architecture:** A daily Render cron reads drglenswartwout@gmail.com via IMAP (`GMAIL_DRGLEN_APP_PASSWORD`), heuristically + LLM-classifies supplier-quote emails, Haiku-extracts structured fields, and INSERT-OR-IGNOREs into `supplier_quotes` (idempotent by gmail message-id). The console lists pending quotes with a fuzzy ingredient/supplier match the operator adjusts, then approves (→ `create_source`) or dismisses.

**Tech Stack:** Python 3 / Flask, SQLite (`chat_log.db`), `imaplib` (IMAP), `anthropic` Haiku (structured extraction), vanilla-JS console. Tests: pytest.

## Global Constraints

- **Review-only:** quotes stage → operator approves → THEN a source is written. No auto-write to `ingredient_sources`.
- **Idempotent** by `gmail_msg_id` (partial unique index + INSERT OR IGNORE); re-scanning the inbox never duplicates a quote.
- `approve_quote` writes via E2 `dashboard.ingredient_catalog.create_source` (ADDS a new source — never overwrites an existing one, so no override conflict). Requires a matched `ingredient_id` (else `ValueError` → operator matches/creates the ingredient first).
- Extraction is a **separable, mockable** function `extract_quote(body, client=None)`; IMAP read is separable too — so the staging/matching/approve logic is unit-tested without network/LLM.
- Reuse: `02 Skills/email-bounce-scan.py` IMAP pattern (`imaplib.IMAP4_SSL("imap.gmail.com")` + app-password login); `dashboard/inbox_ai.py` `_client()` + Haiku `messages.create` (model `claude-haiku-4-5-20251001`) with **tool-use structured output** (force a tool call — never free-text JSON); E2 `create_source`/`create_ingredient`; the console tab + `api()` patterns.
- Endpoints: `@require_console_key`, `ok`/`fail`, `ValueError`→`fail(str(e),status=400)` before generic. Route tests use the Pinecone `pytest.skip` pattern.
- Module name `dashboard/sourcing.py` confirmed collision-free.

---

### Task 1: `dashboard/sourcing.py` — schema, reads, match, approve, dismiss

**Files:**
- Create: `dashboard/sourcing.py`
- Test: `tests/test_sourcing.py`

**Interfaces:**
- Produces: `init_sourcing_schema(cx)`; `stage_quotes(cx, rows) -> int`; `list_quotes(status=None, limit=200, db_path=None)`; `get_quote(qid, db_path=None)`; `update_quote_match(qid, fields, db_path=None)`; `match_quote(cx, qid)`; `approve_quote(qid, db_path=None) -> int`; `dismiss_quote(qid, db_path=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sourcing.py
import sqlite3
import pytest
from dashboard.ingredient_catalog import init_ingredients_schema, list_sources_for_ingredient
from dashboard import sourcing as sc


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        sc.init_sourcing_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name) VALUES (1,'Curcumin')")
        cx.execute("INSERT INTO suppliers (id,company) VALUES (7,'Pharmako')")
        cx.commit()
    return db


_Q = {"gmail_msg_id": "m1", "from_email": "sales@pharmako.com", "subject": "HydroCurc quote",
      "supplier_name": "Pharmako", "ingredient_name": "Curcumin", "price": 334.0, "price_unit": "kg",
      "currency": "USD", "moq": 25.0, "moq_unit": "kg", "lead_time_days": 9, "confidence": 0.9}


def test_stage_idempotent(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        assert sc.stage_quotes(cx, [_Q]) == 1
        assert sc.stage_quotes(cx, [_Q]) == 0          # same gmail_msg_id → no dup
        cx.commit()
    assert len(sc.list_quotes(db_path=db)) == 1


def test_match_and_approve_creates_source(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        sc.stage_quotes(cx, [_Q]); cx.commit()
    qid = sc.list_quotes(db_path=db)[0]["id"]
    with sqlite3.connect(db) as cx:
        sc.match_quote(cx, qid); cx.commit()           # fuzzy: name → ids
    q = sc.get_quote(qid, db_path=db)
    assert q["ingredient_id"] == 1 and q["supplier_id"] == 7
    sid = sc.approve_quote(qid, db_path=db)             # → create_source on ingredient 1
    srcs = list_sources_for_ingredient(1, db_path=db)
    assert len(srcs) == 1 and srcs[0]["id"] == sid and srcs[0]["price_per_unit"] == 334.0
    assert srcs[0]["minimum_order"] == 25.0 and srcs[0]["lead_time_days"] == 9
    assert sc.get_quote(qid, db_path=db)["status"] == "applied"
    with pytest.raises(ValueError):
        sc.approve_quote(qid, db_path=db)              # already applied


def test_approve_requires_ingredient(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        sc.stage_quotes(cx, [{**_Q, "ingredient_name": "Unknownium"}]); cx.commit()
    qid = sc.list_quotes(db_path=db)[0]["id"]
    with sqlite3.connect(db) as cx:
        sc.match_quote(cx, qid); cx.commit()           # no ingredient match → ingredient_id stays NULL
    with pytest.raises(ValueError):
        sc.approve_quote(qid, db_path=db)              # can't apply without a matched ingredient
    sc.dismiss_quote(qid, db_path=db)
    assert sc.get_quote(qid, db_path=db)["status"] == "dismissed"
```

- [ ] **Step 2: Run to verify it fails** — FAIL (module missing).

- [ ] **Step 3: Implement `dashboard/sourcing.py`**

```python
"""Supplier-quote review queue (email-sourcing collector)."""
import os
import sqlite3
from pathlib import Path
from typing import Optional

from dashboard.ingredient_catalog import create_source


def _default_db_path() -> str:
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    cx = sqlite3.connect(db_path or _default_db_path())
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA foreign_keys=ON")
    return cx


def init_sourcing_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS supplier_quotes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          gmail_msg_id TEXT, received_at TEXT, from_email TEXT, subject TEXT, raw_snippet TEXT,
          supplier_name TEXT, ingredient_name TEXT,
          price REAL, price_unit TEXT, currency TEXT,
          moq REAL, moq_unit TEXT, lead_time_days INTEGER, confidence REAL,
          supplier_id INTEGER REFERENCES suppliers(id),
          ingredient_id INTEGER REFERENCES ingredients(id),
          status TEXT DEFAULT 'pending', applied_source_id INTEGER REFERENCES ingredient_sources(id),
          has_attachments INTEGER DEFAULT 0, extras TEXT, notes TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_quotes_msg ON supplier_quotes(gmail_msg_id) WHERE gmail_msg_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_quotes_status ON supplier_quotes(status)")
    cx.commit()


_STAGE_COLS = ["gmail_msg_id", "received_at", "from_email", "subject", "raw_snippet",
               "supplier_name", "ingredient_name", "price", "price_unit", "currency",
               "moq", "moq_unit", "lead_time_days", "confidence", "has_attachments"]


def stage_quotes(cx, rows) -> int:
    cx.row_factory = sqlite3.Row
    n = 0
    for r in rows or []:
        vals = [r.get(c) for c in _STAGE_COLS]
        cur = cx.execute(
            f"INSERT OR IGNORE INTO supplier_quotes ({','.join(_STAGE_COLS)}) "
            f"VALUES ({','.join('?' for _ in _STAGE_COLS)})", vals)
        n += cur.rowcount
    return n


def list_quotes(status=None, limit=200, db_path=None):
    with _connect(db_path) as cx:
        if status:
            rows = cx.execute("SELECT * FROM supplier_quotes WHERE status=? ORDER BY id DESC LIMIT ?",
                              (status, int(limit))).fetchall()
        else:
            rows = cx.execute("SELECT * FROM supplier_quotes ORDER BY (status='pending') DESC, id DESC LIMIT ?",
                              (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def get_quote(qid, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("SELECT * FROM supplier_quotes WHERE id=?", (qid,)).fetchone()
    return dict(r) if r else None


_MATCH_EDITABLE = {"ingredient_id", "supplier_id", "supplier_name", "ingredient_name",
                   "price", "price_unit", "currency", "moq", "moq_unit", "lead_time_days", "notes"}


def update_quote_match(qid, fields, db_path=None) -> None:
    cols = {k: v for k, v in (fields or {}).items() if k in _MATCH_EDITABLE}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE supplier_quotes SET {sets} WHERE id=?", (*cols.values(), qid))
        cx.commit()


def match_quote(cx, qid) -> None:
    """Best-effort fuzzy match: set ingredient_id/supplier_id from the extracted names (exact-ish, single hit)."""
    cx.row_factory = sqlite3.Row
    q = cx.execute("SELECT * FROM supplier_quotes WHERE id=?", (qid,)).fetchone()
    if not q:
        return
    def _one(table, col, name):
        if not name:
            return None
        hits = cx.execute(f"SELECT id FROM {table} WHERE lower({col}) = lower(?) LIMIT 2", (name.strip(),)).fetchall()
        if len(hits) == 1:
            return hits[0]["id"]
        like = cx.execute(f"SELECT id FROM {table} WHERE {col} LIKE ? LIMIT 2", (f"%{name.strip()}%",)).fetchall()
        return like[0]["id"] if len(like) == 1 else None
    iid = _one("ingredients", "name", q["ingredient_name"])
    sid = _one("suppliers", "company", q["supplier_name"])
    cx.execute("UPDATE supplier_quotes SET ingredient_id=COALESCE(?,ingredient_id), supplier_id=COALESCE(?,supplier_id), updated_at=datetime('now') WHERE id=?",
               (iid, sid, qid))


def approve_quote(qid, db_path=None) -> int:
    q = get_quote(qid, db_path=db_path)
    if not q:
        raise ValueError(f"no quote {qid}")
    if q["status"] != "pending":
        raise ValueError(f"quote {qid} is already {q['status']}")
    if not q["ingredient_id"]:
        raise ValueError("match an ingredient before approving")
    src_fields = {
        "supplier_id": q["supplier_id"], "supplier_name": q["supplier_name"],
        "price_per_unit": q["price"], "unit_size": 1, "unit_type": q["price_unit"],
        "minimum_order": q["moq"], "minimum_order_unit": q["moq_unit"],
        "lead_time_days": q["lead_time_days"],
    }
    src_fields = {k: v for k, v in src_fields.items() if v is not None}
    sid = create_source(q["ingredient_id"], src_fields, db_path=db_path)
    with _connect(db_path) as cx:
        cx.execute("UPDATE supplier_quotes SET status='applied', applied_source_id=?, updated_at=datetime('now') WHERE id=?",
                   (sid, qid))
        cx.commit()
    return sid


def dismiss_quote(qid, db_path=None) -> None:
    with _connect(db_path) as cx:
        if not cx.execute("SELECT 1 FROM supplier_quotes WHERE id=?", (qid,)).fetchone():
            raise ValueError(f"no quote {qid}")
        cx.execute("UPDATE supplier_quotes SET status='dismissed', updated_at=datetime('now') WHERE id=?", (qid,))
        cx.commit()
```

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_sourcing.py -q` → 3 pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sourcing.py tests/test_sourcing.py
git commit -m "feat(sourcing): supplier_quotes review queue + approve→create_source"
```

---

### Task 2: `scripts/scan_supplier_quotes.py` — IMAP read + Haiku extract + stage

**Files:**
- Create: `scripts/scan_supplier_quotes.py`
- Test: `tests/test_scan_supplier_quotes.py`

**Interfaces:**
- Produces: `looks_like_quote(subject, body) -> bool` (cheap pre-filter); `extract_quote(subject, body, client=None) -> dict|None` (Haiku tool-use; `client` injectable for tests); `scan(write=False, days=14, db_path=None, imap=None, client=None) -> dict` (the cron entry; `imap`/`client` injectable).

- [ ] **Step 1: Write the failing test** (mock IMAP + LLM — only the testable seams)

```python
# tests/test_scan_supplier_quotes.py
import sqlite3
from dashboard import sourcing as sc
from scripts.scan_supplier_quotes import looks_like_quote, extract_quote, _to_stage_row


def test_looks_like_quote():
    assert looks_like_quote("Re: HydroCurc quote", "Price is $334/kg, MOQ 25 kg, lead time 7-10 days")
    assert not looks_like_quote("Lunch?", "are you free thursday")


class _FakeClient:
    def __init__(self, payload): self._p = payload
    @property
    def messages(self): return self
    def create(self, **kw):
        class _Block:  # mimic a tool_use content block
            type = "tool_use"; name = "record_quote"
        b = _Block(); b.input = self._p
        class _Msg: content = [b]
        return _Msg()


def test_extract_quote_tooluse():
    payload = {"is_supplier_quote": True, "supplier_name": "Pharmako", "ingredient_name": "HydroCurc",
               "price": 334, "price_unit": "kg", "currency": "USD", "moq": 25, "moq_unit": "kg",
               "lead_time_days": 9, "confidence": 0.9}
    q = extract_quote("HydroCurc quote", "…", client=_FakeClient(payload))
    assert q["is_supplier_quote"] and q["price"] == 334 and q["ingredient_name"] == "HydroCurc"
    # a non-quote returns None
    assert extract_quote("hi", "…", client=_FakeClient({"is_supplier_quote": False})) is None


def test_stage_row_idempotent(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        from dashboard.ingredient_catalog import init_ingredients_schema
        init_ingredients_schema(cx); sc.init_sourcing_schema(cx); cx.commit()
        row = _to_stage_row("msg-1", "sales@x.com", "Quote",
                            {"supplier_name": "X", "ingredient_name": "Y", "price": 10, "price_unit": "kg", "confidence": 0.8})
        assert sc.stage_quotes(cx, [row]) == 1
        assert sc.stage_quotes(cx, [row]) == 0
        cx.commit()
```

- [ ] **Step 2: Run to verify it fails** — FAIL (module missing).

- [ ] **Step 3: Implement `scripts/scan_supplier_quotes.py`**

Full code for the testable seams; the IMAP + client glue follows the cited prior art.

```python
#!/usr/bin/env python3
"""Scan Glen's inbox for supplier quotes → stage into supplier_quotes (review queue).
IMAP read (reuses the email-bounce-scan pattern) + Haiku tool-use extraction. Dry-run default."""
import argparse, imaplib, email, os, re, sqlite3, sys
from email.header import decode_header
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import sourcing as sc  # noqa: E402

_MODEL = "claude-haiku-4-5-20251001"
_KW = re.compile(r"\b(quote|price|\$|/kg|/lb|per kg|moq|minimum order|lead time|cost|coa|c of a)\b", re.I)

_TOOL = {
    "name": "record_quote",
    "description": "Record a supplier price quote extracted from a sourcing email.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_supplier_quote": {"type": "boolean"},
            "supplier_name": {"type": "string"}, "ingredient_name": {"type": "string"},
            "price": {"type": "number"}, "price_unit": {"type": "string"},
            "currency": {"type": "string"},
            "moq": {"type": "number"}, "moq_unit": {"type": "string"},
            "lead_time_days": {"type": "integer"}, "confidence": {"type": "number"},
        },
        "required": ["is_supplier_quote"],
    },
}


def looks_like_quote(subject: str, body: str) -> bool:
    return bool(_KW.search((subject or "") + " " + (body or "")))


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def extract_quote(subject, body, client=None):
    client = client or _client()
    msg = client.messages.create(
        model=_MODEL, max_tokens=600, tools=[_TOOL], tool_choice={"type": "tool", "name": "record_quote"},
        messages=[{"role": "user", "content":
                   f"Extract the supplier price quote from this email. If it is not a supplier price "
                   f"quote, set is_supplier_quote=false.\n\nSubject: {subject}\n\n{body[:6000]}"}])
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            data = block.input
            return data if data.get("is_supplier_quote") else None
    return None


def _to_stage_row(msg_id, from_email, subject, data, has_attachments=0):
    return {"gmail_msg_id": msg_id, "from_email": from_email, "subject": subject,
            "raw_snippet": (subject or "")[:200], "has_attachments": has_attachments,
            "supplier_name": data.get("supplier_name"), "ingredient_name": data.get("ingredient_name"),
            "price": data.get("price"), "price_unit": data.get("price_unit"), "currency": data.get("currency"),
            "moq": data.get("moq"), "moq_unit": data.get("moq_unit"),
            "lead_time_days": data.get("lead_time_days"), "confidence": data.get("confidence")}


def _decode(s):
    if not s:
        return ""
    out = []
    for part, enc in decode_header(s):
        out.append(part.decode(enc or "utf-8", "replace") if isinstance(part, bytes) else part)
    return "".join(out)


def _body_text(m):
    if m.is_multipart():
        for p in m.walk():
            if p.get_content_type() == "text/plain":
                try:
                    return p.get_payload(decode=True).decode(p.get_content_charset() or "utf-8", "replace")
                except Exception:
                    pass
        return ""
    try:
        return m.get_payload(decode=True).decode(m.get_content_charset() or "utf-8", "replace")
    except Exception:
        return ""


def scan(write=False, days=14, db_path=None, imap=None, client=None) -> dict:
    user = os.environ.get("GMAIL_DRGLEN_USER", "drglenswartwout@gmail.com")
    pw = os.environ.get("GMAIL_DRGLEN_APP_PASSWORD", "")
    own = imap is None
    if own:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(user, pw)
    staged = scanned = 0
    try:
        imap.select("INBOX")
        import datetime
        since = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%d-%b-%Y")
        typ, data = imap.search(None, "SINCE", since)
        ids = data[0].split() if data and data[0] else []
        cx = sqlite3.connect(db_path or sc._default_db_path()); cx.row_factory = sqlite3.Row
        try:
            sc.init_sourcing_schema(cx)
            existing = {r["gmail_msg_id"] for r in cx.execute("SELECT gmail_msg_id FROM supplier_quotes WHERE gmail_msg_id IS NOT NULL")}
            for num in ids:
                typ, md = imap.fetch(num, "(RFC822)")
                m = email.message_from_bytes(md[0][1])
                msg_id = m.get("Message-ID") or num.decode()
                if msg_id in existing:
                    continue
                subject = _decode(m.get("Subject")); body = _body_text(m)
                scanned += 1
                if not looks_like_quote(subject, body):
                    continue
                data2 = extract_quote(subject, body, client=client)
                if not data2:
                    continue
                row = _to_stage_row(msg_id, _decode(m.get("From")), subject, data2,
                                    has_attachments=1 if m.is_multipart() and any(p.get_filename() for p in m.walk()) else 0)
                n = sc.stage_quotes(cx, [row]); staged += n
            if write:
                cx.commit()
            else:
                cx.rollback()
        finally:
            cx.close()
    finally:
        if own:
            imap.logout()
    return {"scanned": scanned, "staged": staged, "mode": "write" if write else "dry_run"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--days", type=int, default=14)
    print(scan(write=ap.parse_args().write, days=ap.parse_args().days))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_scan_supplier_quotes.py tests/test_sourcing.py -q` → green (the IMAP `scan()` path is exercised only via injected fakes if at all; the seams `looks_like_quote`/`extract_quote`/`_to_stage_row` are unit-tested).

- [ ] **Step 5: Commit**

```bash
git add scripts/scan_supplier_quotes.py tests/test_scan_supplier_quotes.py
git commit -m "feat(sourcing): inbox scanner — IMAP + Haiku tool-use extraction + idempotent stage"
```

---

### Task 3: `/api/sourcing/*` endpoints

**Files:**
- Modify: `app.py`
- Test: `tests/test_admin_sourcing_api.py`

**Interfaces:**
- Produces: `GET /api/sourcing/quotes`; `PATCH /api/sourcing/quotes/<int:qid>` (match-adjust); `POST /api/sourcing/quotes/<int:qid>/approve`; `POST /api/sourcing/quotes/<int:qid>/dismiss`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_admin_sourcing_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.sourcing import init_sourcing_schema, stage_quotes
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_sourcing_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name) VALUES (1,'Curcumin')")
        stage_quotes(cx, [{"gmail_msg_id": "m1", "ingredient_name": "Curcumin",
                           "price": 334.0, "price_unit": "kg", "moq": 25.0, "confidence": 0.9}])
        cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()


def test_sourcing_flow(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    q = c.get("/api/sourcing/quotes").get_json()["data"]
    qid = q[0]["id"]
    assert c.patch(f"/api/sourcing/quotes/{qid}", json={"ingredient_id": 1}).status_code == 200
    r = c.post(f"/api/sourcing/quotes/{qid}/approve")
    assert r.status_code == 200 and r.get_json()["data"]["source_id"]
```

- [ ] **Step 2: Run to verify it fails** — FAIL/SKIP. Proceed.

- [ ] **Step 3: Add endpoints in `app.py`** (beside the other `/api/*` blocks)

```python
from dashboard import sourcing as _sourcing


@app.route("/api/sourcing/quotes", methods=["GET"])
@require_console_key
def api_sourcing_quotes():
    try:
        return ok(_sourcing.list_quotes(request.args.get("status"), int(request.args.get("limit", 200))))
    except Exception as e:
        return fail(e)


@app.route("/api/sourcing/quotes/<int:qid>", methods=["PATCH"])
@require_console_key
def api_sourcing_match(qid):
    try:
        _sourcing.update_quote_match(qid, request.get_json(silent=True) or {})
        return ok(_sourcing.get_quote(qid))
    except Exception as e:
        return fail(e)


@app.route("/api/sourcing/quotes/<int:qid>/approve", methods=["POST"])
@require_console_key
def api_sourcing_approve(qid):
    try:
        return ok({"source_id": _sourcing.approve_quote(qid)})
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)


@app.route("/api/sourcing/quotes/<int:qid>/dismiss", methods=["POST"])
@require_console_key
def api_sourcing_dismiss(qid):
    try:
        _sourcing.dismiss_quote(qid)
        return ok({"id": qid})
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)
```

Add a schema-init wiring for `init_sourcing_schema` alongside the other `_init_*_tables()` calls (open `LOG_DB`, init, try/finally close).

- [ ] **Step 4: Run tests** — PASS or SKIP on Pinecone. Smoke `python3 -c "import app"`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_admin_sourcing_api.py
git commit -m "feat(sourcing): /api/sourcing/* list/match/approve/dismiss endpoints"
```

---

### Task 4: Console "Sourcing inbox" tab + cron wiring

**Files:**
- Modify: `static/admin-ingredients.html`, `static/console-search-index.json`
- Create: `render-cron-sourcing.md` (a short ops note; the cron is wired in the Render dashboard)

- [ ] **Step 1: Read** the existing tabs (Inventory/Production/Reorder) + the `api()`/search-to-pick/`escapeHtml` patterns.

- [ ] **Step 2: Add `"sourcing"` to the `labels` array + a "Sourcing inbox" tab:** a list of quotes (`GET /api/sourcing/quotes`, pending first) showing supplier · ingredient · $price/unit · MOQ · lead-time · confidence · status; clicking a row opens a detail with the extracted fields editable (→ `PATCH /api/sourcing/quotes/<id>` on change) + an ingredient **search-to-pick** (sets `ingredient_id`) + a supplier search-to-pick + **Approve** (`POST .../approve`) and **Dismiss** (`POST .../dismiss`) buttons. Approve disabled until an `ingredient_id` is set; show the `ValueError` toast if approve fails. Reuse the real `api()` (returns `j.data`, throws), `escapeHtml`, the index-array picker (NO `JSON.stringify` in onclick). Add the `display:none` initial CSS for the detail panel.

- [ ] **Step 3: Register in `static/console-search-index.json`:** `{ "title": "Sourcing Inbox (supplier quotes)", "page": "Products", "url": "/admin/ingredients", "keywords": ["sourcing","quote","supplier","price","email","inbox","moq"] }`.

- [ ] **Step 4: Write `render-cron-sourcing.md`** — the ops note: a daily Render cron `glen-sourcing-scan` running `python scripts/scan_supplier_quotes.py --write` with `GMAIL_DRGLEN_APP_PASSWORD` + `ANTHROPIC_API_KEY` from the env (mirrors `glen-qbo-reconcile`). Dry-run first (`--write` omitted) to review counts.

- [ ] **Step 5: Verify** HTML parses; no `JSON.stringify` in onclick; ids consistent; existing tabs untouched. Commit.

```bash
git add static/admin-ingredients.html static/console-search-index.json render-cron-sourcing.md
git commit -m "feat(sourcing): Sourcing-inbox console tab + cron ops note"
```

---

## Self-Review
- **Spec coverage:** supplier_quotes + reads/match/approve/dismiss (T1); IMAP+Haiku scanner idempotent stage (T2); endpoints (T3); console + cron (T4). Review-only (approve-gated) ✓. Idempotent by gmail_msg_id ✓. Approve → E2 `create_source` (additive, no overwrite) ✓. Extraction mockable + tool-use structured ✓.
- **Placeholders:** full code for T1/T3 + the testable seams of T2; T4 adapts to existing console patterns.
- **Type consistency:** `stage_quotes`/`list_quotes`/`get_quote`/`update_quote_match`/`match_quote`/`approve_quote`/`dismiss_quote` used identically across module/scanner/endpoints/tests; `approve_quote` → `create_source` (E2) with mapped fields; `extract_quote(subject,body,client=None)` injectable.
- **Reviewer note (T2):** the IMAP `scan()` glue is integration code exercised in prod, not unit-tested; the unit tests cover `looks_like_quote`/`extract_quote`(mock client)/`_to_stage_row`+stage idempotency. The Haiku call MUST be tool-use (forced tool_choice) — never parse free-text JSON (the journal lesson).

## Build approach
Subagent-driven-development, this branch (off merged main), one PR, whole-branch review. Order T1 (foundation: the approve→create_source bridge + idempotency) → T2 (scanner) → T3 (endpoints) → T4 (console + cron). After merge + a dry-run scan, review staged quotes, then enable the cron. The HydroCurc/Pharmako email is the live end-to-end test.
