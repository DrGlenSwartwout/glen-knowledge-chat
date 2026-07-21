# Client-Centered Console Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a per-client console hub at `/console/client?email=…` that folds together a client's clinical tags, tests, invoices, and comms, plus a live "current process" strip of sequence-status buttons linking to where each action happens.

**Architecture:** One new pure module `dashboard/client_360.py` holds all read/assemble logic (a generalized process-strip resolver, a clinical-tags-by-email reader, and a `bundle()` assembler that calls the existing per-client readers). `app.py` gets one JSON endpoint `/api/console/client-360` and one static-page route `/console/client`. A new `static/console-client.html` renders the payload. Deep-links from CRM / handoffs / nav point at the hub. No schema migration (Option A): the process strip ties its Recommendation stage to an order heuristically (same email, latest non-cancelled order), exactly as the existing biofield pipeline already does.

**Tech Stack:** Python 3 / Flask monolith (`app.py`), `dashboard/*.py` modules, SQLite (`LOG_DB` = `chat_log.db`; read-only synced `e4l.db`), pytest, vanilla-JS static HTML pages with `op-nav.js`.

## Global Constraints

- **Email is the universal join key** — lowercased. There is no single client id. Every reader keys off `lower(email)`.
- **Balances are always derived, never stored** — use `order_payments.balance(cx, order_id)`; never read a stored balance column.
- **No new order↔recommendation-source schema link.** Do not add columns to `orders`. Source is detected from existing recommendation records only.
- **The page must never error when data is missing** — every section renders an empty headline; the clinical-tags reader degrades to `{"active": [], "suggested": []}` when `e4l.db` or the `client_clinical_tags` table is unavailable.
- **Auth pattern** — API endpoints gate with `actor = _bos_actor(); if actor is None: return jsonify({"ok": False, "error": "unauthorized"}), 401`. Page routes are served like `bos_crm_page()`.
- **Money is cents server-side**, formatted to dollars in the page/payload only where a `_dollars` field is explicitly produced.
- **CI has a known_failures ratchet (~6450 tests).** New tests must pass; do not introduce new failures.
- Follow existing console-test patterns in `tests/test_console_*.py`.

---

### Task 1: Generalized process-strip resolver

**Files:**
- Create: `dashboard/client_360.py`
- Test: `tests/test_client_360_process.py`

**Interfaces:**
- Produces: `client_360.process_strip(cx, email) -> {"source": str|None, "order_id": int|None, "stages": list[dict]}` where `cx` is a `LOG_DB` connection with `row_factory = sqlite3.Row`. Each stage dict is `{"key": str, "label": str, "done": bool, "action": dict, ...}`. Stage keys in order: `recommendation, invoice, sent, paid, fulfilled`. `action.kind` ∈ `{"link","dispatch","none"}`; `link` carries `target` (a URL with no console key and no `?email`), `dispatch` carries `target` (an action key) + `order_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client_360_process.py
import sqlite3
import pytest
from dashboard import client_360


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, "
               "status TEXT, pay_status TEXT, invoice_sent_at TEXT)")
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY, email TEXT, scan_date TEXT)")
    cx.execute("CREATE TABLE ff_match_drafts (email TEXT, scan_date TEXT, status TEXT)")
    cx.execute("CREATE TABLE intake_responses (email TEXT PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE inquiries (id INTEGER PRIMARY KEY, client_email TEXT)")
    return cx


def _stage(res, key):
    return next(s for s in res["stages"] if s["key"] == key)


def test_no_data_all_pending_no_source():
    res = client_360.process_strip(_cx(), "nobody@example.com")
    assert res["source"] is None
    assert res["order_id"] is None
    assert [s["key"] for s in res["stages"]] == ["recommendation", "invoice", "sent", "paid", "fulfilled"]
    assert all(s["done"] is False for s in res["stages"])
    assert _stage(res, "recommendation")["action"]["kind"] == "none"


def test_source_priority_biofield_over_scan():
    cx = _cx()
    cx.execute("INSERT INTO biofield_reveals (id, email, scan_date) VALUES (1, 'a@b.com', '2026-07-01')")
    cx.execute("INSERT INTO ff_match_drafts (email, scan_date, status) VALUES ('a@b.com', '2026-07-01', 'draft')")
    res = client_360.process_strip(cx, "A@B.com")
    assert res["source"] == "biofield"
    rec = _stage(res, "recommendation")
    assert rec["done"] is True
    assert rec["action"] == {"kind": "link", "target": "/console/biofield-portal"}


def test_scan_source_when_no_biofield():
    cx = _cx()
    cx.execute("INSERT INTO ff_match_drafts (email, scan_date, status) VALUES ('a@b.com', '2026-07-01', 'draft')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] == "scan"
    assert _stage(res, "recommendation")["action"]["target"] == "/console/ff-drafts"


def test_intake_source():
    cx = _cx()
    cx.execute("INSERT INTO intake_responses (email, status) VALUES ('a@b.com', 'submitted')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] == "intake"


def test_intake_draft_is_not_a_source():
    cx = _cx()
    cx.execute("INSERT INTO intake_responses (email, status) VALUES ('a@b.com', 'draft')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] is None


def test_chat_source():
    cx = _cx()
    cx.execute("INSERT INTO inquiries (id, client_email) VALUES (1, 'a@b.com')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] == "chat"


def test_money_stages_from_latest_order():
    cx = _cx()
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at) "
               "VALUES ('a@b.com', 'cancelled', 'unpaid', NULL)")  # ignored
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at) "
               "VALUES ('a@b.com', 'confirmed', 'unpaid', '2026-07-10')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["order_id"] == 2
    assert _stage(res, "invoice")["done"] is True
    assert _stage(res, "sent")["done"] is True          # invoice_sent_at present
    assert _stage(res, "paid")["done"] is False
    assert _stage(res, "fulfilled")["done"] is False


def test_unsent_invoice_offers_send_dispatch():
    cx = _cx()
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at) "
               "VALUES ('a@b.com', 'confirmed', 'unpaid', NULL)")
    res = client_360.process_strip(cx, "a@b.com")
    sent = _stage(res, "sent")
    assert sent["done"] is False
    assert sent["action"] == {"kind": "dispatch", "target": "orders.send_invoice", "order_id": 1}


def test_paid_and_fulfilled():
    cx = _cx()
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at) "
               "VALUES ('a@b.com', 'shipped', 'paid', '2026-07-10')")
    res = client_360.process_strip(cx, "a@b.com")
    assert _stage(res, "paid")["done"] is True
    assert _stage(res, "fulfilled")["done"] is True


def test_missing_recommendation_tables_do_not_raise():
    # Bare orders-only db (no biofield_reveals/ff_match_drafts/etc.)
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, status TEXT, "
               "pay_status TEXT, invoice_sent_at TEXT)")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_client_360_process.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.client_360'` (or `AttributeError: process_strip`).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/client_360.py
"""Per-client "client-360" hub: read + assemble everything the console knows
about one client (by email). Pure functions take open sqlite connections so
they are testable offline. No writes."""
import sqlite3

_SOURCE_ACTION = {
    "biofield": {"kind": "link", "target": "/console/biofield-portal"},
    "scan":     {"kind": "link", "target": "/console/ff-drafts"},
    "intake":   {"kind": "link", "target": "/console/crm"},
    "chat":     {"kind": "link", "target": "/console/crm"},
}
_SOURCE_LABEL = {"biofield": "Biofield", "scan": "Scan",
                 "intake": "Intake", "chat": "Chat"}
_FULFILLED = ("shipped", "delivered", "done", "fulfilled")


def _exists(cx, sql, params):
    """True if the query returns a row; False if the table is absent."""
    try:
        return cx.execute(sql, params).fetchone() is not None
    except sqlite3.OperationalError:
        return False


def _detect_source(cx, email):
    """Recommendation source for a client, in priority order. None if no
    concrete recommendation record exists."""
    if _exists(cx, "SELECT 1 FROM biofield_reveals WHERE lower(email)=? LIMIT 1", (email,)):
        return "biofield"
    if _exists(cx, "SELECT 1 FROM ff_match_drafts WHERE lower(email)=? LIMIT 1", (email,)):
        return "scan"
    if _exists(cx, "SELECT 1 FROM intake_responses WHERE lower(email)=? AND status='submitted' LIMIT 1", (email,)):
        return "intake"
    if _exists(cx, "SELECT 1 FROM inquiries WHERE lower(client_email)=? LIMIT 1", (email,)):
        return "chat"
    return None


def process_strip(cx, email):
    """The client's CURRENT in-flight cycle as sequence-status stages.
    cx: LOG_DB connection (row_factory=sqlite3.Row). Read-only."""
    e = (email or "").strip().lower()
    source = _detect_source(cx, e)
    order = cx.execute(
        "SELECT id, COALESCE(status,'') status, COALESCE(pay_status,'') pay, "
        "COALESCE(invoice_sent_at,'') sent FROM orders "
        "WHERE lower(COALESCE(email,''))=? AND COALESCE(status,'')<>'cancelled' "
        "ORDER BY id DESC LIMIT 1", (e,)).fetchone()
    oid = order["id"] if order else None
    status = order["status"] if order else ""
    pay = order["pay"] if order else ""
    sent = order["sent"] if order else ""

    rec_action = _SOURCE_ACTION.get(source, {"kind": "none"}) if source else {"kind": "none"}
    stages = [
        {"key": "recommendation",
         "label": _SOURCE_LABEL.get(source, "Recommendation"),
         "done": source is not None, "source": source, "action": rec_action},
        {"key": "invoice", "label": "Invoice", "done": order is not None,
         "action": {"kind": "link", "target": "/console/orders"} if order else {"kind": "none"}},
        {"key": "sent", "label": "Sent", "done": bool(sent),
         "action": ({"kind": "dispatch", "target": "orders.send_invoice", "order_id": oid}
                    if order and not sent else {"kind": "link", "target": "/console/orders"})},
        {"key": "paid", "label": "Paid", "done": pay == "paid",
         "action": {"kind": "link", "target": "/console/orders"} if order else {"kind": "none"}},
        {"key": "fulfilled", "label": "Fulfilled", "done": status in _FULFILLED,
         "action": {"kind": "link", "target": "/console/orders"} if order else {"kind": "none"}},
    ]
    return {"source": source, "order_id": oid, "stages": stages}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_client_360_process.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add dashboard/client_360.py tests/test_client_360_process.py
git commit -m "feat(console): generalized client process-strip resolver"
```

---

### Task 2: Clinical-tags-by-email reader (graceful degrade)

**Files:**
- Modify: `dashboard/client_360.py`
- Test: `tests/test_client_360_tags.py`

**Interfaces:**
- Consumes: `dashboard.biofield_e4l._db_path`, `dashboard.biofield_e4l._connect_ro`, `dashboard.clinical_tags_console.client_tags`.
- Produces: `client_360.client_tags_for_email(email, *, e4l_path=None) -> {"active": list, "suggested": list}`. Resolves `client_id` from `e4l_clients` by email, then reads `client_clinical_tags`. Returns empty lists when the db file, the `e4l_clients` row, or the `client_clinical_tags` table is missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client_360_tags.py
import os
import sqlite3
import pytest
from dashboard import client_360


def _make_e4l(path, *, with_tags=True):
    cx = sqlite3.connect(path)
    cx.execute("CREATE TABLE e4l_clients (client_id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    cx.execute("INSERT INTO e4l_clients VALUES (7, 'Jane Doe', 'jane@example.com')")
    if with_tags:
        cx.execute("CREATE TABLE client_clinical_tags (client_id INTEGER, axis TEXT, tag TEXT, "
                   "status TEXT, confidence REAL, source TEXT, evidence TEXT, confirmed_by TEXT)")
        cx.execute("INSERT INTO client_clinical_tags VALUES (7,'A','system:gut','active',0.9,'auto','x','glen')")
        cx.execute("INSERT INTO client_clinical_tags VALUES (7,'B','element:water','suggested',0.5,'infer','y',NULL)")
    cx.commit()
    cx.close()


def test_reads_active_and_suggested(tmp_path):
    p = str(tmp_path / "e4l.db")
    _make_e4l(p)
    out = client_360.client_tags_for_email("JANE@example.com", e4l_path=p)
    assert [t["tag"] for t in out["active"]] == ["system:gut"]
    assert [t["tag"] for t in out["suggested"]] == ["element:water"]


def test_missing_tags_table_degrades_empty(tmp_path):
    p = str(tmp_path / "e4l.db")
    _make_e4l(p, with_tags=False)
    out = client_360.client_tags_for_email("jane@example.com", e4l_path=p)
    assert out == {"active": [], "suggested": []}


def test_missing_db_file_degrades_empty(tmp_path):
    out = client_360.client_tags_for_email("jane@example.com", e4l_path=str(tmp_path / "nope.db"))
    assert out == {"active": [], "suggested": []}


def test_unknown_email_degrades_empty(tmp_path):
    p = str(tmp_path / "e4l.db")
    _make_e4l(p)
    out = client_360.client_tags_for_email("stranger@example.com", e4l_path=p)
    assert out == {"active": [], "suggested": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_client_360_tags.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.client_360' has no attribute 'client_tags_for_email'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/client_360.py`:

```python
def client_tags_for_email(email, *, e4l_path=None):
    """Clinical tags for a client from the synced e4l.db. Degrades to empty
    lists when the db/table/row is unavailable — never raises."""
    from dashboard import biofield_e4l, clinical_tags_console
    empty = {"active": [], "suggested": []}
    path = biofield_e4l._db_path(e4l_path)
    cx = biofield_e4l._connect_ro(path)
    if cx is None:
        return empty
    try:
        row = cx.execute("SELECT client_id FROM e4l_clients WHERE lower(email)=lower(?)",
                         ((email or "").strip(),)).fetchone()
        if not row:
            return empty
        data = clinical_tags_console.client_tags(cx, row["client_id"])
        return {"active": data.get("active", []), "suggested": data.get("suggested", [])}
    except sqlite3.Error:
        return empty
    finally:
        cx.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_client_360_tags.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add dashboard/client_360.py tests/test_client_360_tags.py
git commit -m "feat(console): clinical-tags-by-email reader with graceful degrade"
```

---

### Task 3: `bundle()` assembler

**Files:**
- Modify: `dashboard/client_360.py`
- Test: `tests/test_client_360_bundle.py`

**Interfaces:**
- Consumes: `client_360.process_strip`, `client_360.client_tags_for_email`; existing readers `dashboard.client_scans.scans_for`, `dashboard.biofield_reveals.list_for_email`, `dashboard.order_payments.balance`, `dashboard.fmp_orders.client_order_history`, `dashboard.recent_comms.recent_comms`.
- Produces: `client_360.bundle(cx, email, *, e4l_path=None) -> dict` with keys `person, clinical, tests, invoices, comms, process`. Shapes:
  - `person`: `{name, email, phone, location, profession, order_count, last_order_date}` (empty strings/0 when absent).
  - `clinical`: `{active: [...], suggested: [...]}`.
  - `tests`: `[{date, type}]` newest-first; `type` ∈ `{"biofield","scan"}` (biofield wins on a shared date).
  - `invoices`: `{total_paid_cents, open_balance_cents, orders: [{id, date, status, total_cents, paid_cents, balance_cents, edit_url}], fmp: [...]}`.
  - `comms`: `[{date, topic, source}]` newest-first; `source` ∈ `{"inquiry","query","feedback","intake"}`.
  - `process`: the `process_strip` dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client_360_bundle.py
import sqlite3
import pytest
from dashboard import client_360


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT, name TEXT, phone TEXT, "
               "city TEXT, state TEXT, island TEXT, profession TEXT, order_count INTEGER, last_order_date TEXT)")
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, status TEXT, "
               "pay_status TEXT, invoice_sent_at TEXT, total_cents INTEGER, created_at TEXT)")
    cx.execute("CREATE TABLE order_payments (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, "
               "kind TEXT, amount_cents INTEGER, status TEXT DEFAULT 'active')")
    cx.execute("CREATE TABLE client_scans (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, scan_date TEXT, scan_id TEXT)")
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, scan_date TEXT, "
               "interpretation_json TEXT DEFAULT '{}', remedies_json TEXT DEFAULT '[]', first_approved INTEGER DEFAULT 0, "
               "created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '', dropped TEXT DEFAULT '[]', "
               "layers_json TEXT DEFAULT '[]', notified_at TEXT, requested_at TEXT, token_hash TEXT, "
               "approved_at TEXT, approved_by TEXT)")
    cx.execute("CREATE TABLE inquiries (id INTEGER PRIMARY KEY AUTOINCREMENT, client_email TEXT, "
               "main_challenge TEXT, main_goal TEXT, created_at TEXT)")
    return cx


def test_bundle_shape_and_invoice_math(tmp_path):
    cx = _cx()
    cx.execute("INSERT INTO people (id,email,name,phone,city,state,island,profession,order_count,last_order_date) "
               "VALUES (1,'a@b.com','Al Bee','808','Hilo','HI','Big Island','yoga',2,'2026-07-10')")
    cx.execute("INSERT INTO orders (email,status,pay_status,invoice_sent_at,total_cents,created_at) "
               "VALUES ('a@b.com','confirmed','unpaid','2026-07-10',10000,'2026-07-09')")
    cx.execute("INSERT INTO order_payments (order_id,kind,amount_cents,status) VALUES (1,'payment',4000,'active')")
    cx.execute("INSERT INTO client_scans (email,scan_date,scan_id) VALUES ('a@b.com','2026-07-01','s1')")
    cx.execute("INSERT INTO biofield_reveals (email,scan_date) VALUES ('a@b.com','2026-07-05')")
    cx.execute("INSERT INTO inquiries (client_email,main_challenge,main_goal,created_at) "
               "VALUES ('a@b.com','fatigue','energy','2026-07-08 12:00:00')")
    b = client_360.bundle(cx, "a@b.com", e4l_path=str(tmp_path / "missing.db"))

    assert b["person"]["name"] == "Al Bee"
    assert b["person"]["location"] == "Hilo, HI"
    assert b["clinical"] == {"active": [], "suggested": []}   # no e4l db -> empty
    # tests: newest first, biofield 07-05 before scan 07-01
    assert [(t["date"], t["type"]) for t in b["tests"]] == [("2026-07-05", "biofield"), ("2026-07-01", "scan")]
    # invoices: 100.00 total, 40.00 paid, 60.00 balance
    assert b["invoices"]["total_paid_cents"] == 4000
    assert b["invoices"]["open_balance_cents"] == 6000
    o = b["invoices"]["orders"][0]
    assert (o["total_cents"], o["paid_cents"], o["balance_cents"]) == (10000, 4000, 6000)
    assert o["edit_url"] == "/orders/new?edit_order=1"
    # comms include the inquiry
    assert any(c["source"] == "inquiry" and c["topic"] for c in b["comms"])
    # process present
    assert b["process"]["source"] == "biofield"


def test_bundle_empty_client(tmp_path):
    cx = _cx()
    b = client_360.bundle(cx, "nobody@x.com", e4l_path=str(tmp_path / "missing.db"))
    assert b["person"]["name"] == ""
    assert b["tests"] == []
    assert b["invoices"] == {"total_paid_cents": 0, "open_balance_cents": 0, "orders": [], "fmp": []}
    assert b["comms"] == []
    assert b["process"]["source"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_client_360_bundle.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.client_360' has no attribute 'bundle'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/client_360.py`:

```python
def _person(cx, email):
    empty = {"name": "", "email": email, "phone": "", "location": "",
             "profession": "", "order_count": 0, "last_order_date": ""}
    try:
        r = cx.execute(
            "SELECT name, email, COALESCE(phone,'') phone, COALESCE(city,'') city, "
            "COALESCE(state,'') state, COALESCE(island,'') island, "
            "COALESCE(profession,'') profession, COALESCE(order_count,0) oc, "
            "COALESCE(last_order_date,'') lod FROM people WHERE lower(email)=? LIMIT 1",
            (email,)).fetchone()
    except sqlite3.OperationalError:
        return empty
    if not r:
        return empty
    loc = ", ".join(p for p in (r["city"], r["state"]) if p) or (r["island"] or "")
    return {"name": r["name"] or "", "email": r["email"] or email, "phone": r["phone"],
            "location": loc, "profession": r["profession"],
            "order_count": r["oc"], "last_order_date": r["lod"]}


def _tests(cx, email):
    from dashboard import client_scans, biofield_reveals
    by_date = {}
    try:
        for s in client_scans.scans_for(cx, email):
            by_date[s["scan_date"]] = "scan"
    except sqlite3.OperationalError:
        pass
    try:
        for rv in biofield_reveals.list_for_email(cx, email):
            by_date[rv["scan_date"]] = "biofield"   # biofield wins on a shared date
    except sqlite3.OperationalError:
        pass
    return [{"date": d, "type": t}
            for d, t in sorted(by_date.items(), key=lambda kv: kv[0], reverse=True)]


def _invoices(cx, email):
    from dashboard import order_payments, fmp_orders
    out = {"total_paid_cents": 0, "open_balance_cents": 0, "orders": [], "fmp": []}
    try:
        rows = cx.execute(
            "SELECT id, COALESCE(status,'') status, COALESCE(created_at,'') created_at, "
            "COALESCE(total_cents,0) total FROM orders "
            "WHERE lower(COALESCE(email,''))=? AND COALESCE(status,'')<>'cancelled' "
            "ORDER BY id DESC", (email,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    for r in rows:
        bal = order_payments.balance(cx, r["id"])
        out["orders"].append({
            "id": r["id"], "date": r["created_at"], "status": r["status"],
            "total_cents": bal["invoice_cents"], "paid_cents": bal["paid_cents"],
            "balance_cents": bal["balance_cents"],
            "edit_url": f"/orders/new?edit_order={r['id']}"})
        out["total_paid_cents"] += bal["paid_cents"]
        if bal["balance_cents"] > 0:
            out["open_balance_cents"] += bal["balance_cents"]
    try:
        out["fmp"] = fmp_orders.client_order_history(cx, email=email)
    except Exception:
        out["fmp"] = []
    return out


def _comms(cx, email):
    from dashboard import recent_comms
    try:
        rc = recent_comms.recent_comms(cx, email, days_window=3650)
    except Exception:
        return []
    out = []
    for q in rc.get("recent_inquiries", []):
        topic = q.get("main_challenge") or q.get("main_goal") or "inquiry"
        out.append({"date": q.get("created_at") or "", "topic": topic, "source": "inquiry"})
    for q in rc.get("recent_queries", []):
        out.append({"date": q.get("ts") or "", "topic": q.get("question") or "", "source": "query"})
    for f in rc.get("recent_feedback", []):
        topic = ", ".join(f.get("topics") or []) or f.get("summary") or "feedback"
        out.append({"date": f.get("received_at") or "", "topic": topic, "source": "feedback"})
    if rc.get("intake_summary"):
        out.append({"date": "", "topic": "Intake on file", "source": "intake"})
    out.sort(key=lambda c: c["date"], reverse=True)
    return out


def bundle(cx, email, *, e4l_path=None):
    """Assemble the full client-360 payload. cx: LOG_DB connection
    (row_factory=sqlite3.Row). Read-only; never raises on missing data."""
    e = (email or "").strip().lower()
    return {
        "person": _person(cx, e),
        "clinical": client_tags_for_email(e, e4l_path=e4l_path),
        "tests": _tests(cx, e),
        "invoices": _invoices(cx, e),
        "comms": _comms(cx, e),
        "process": process_strip(cx, e),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_client_360_bundle.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add dashboard/client_360.py tests/test_client_360_bundle.py
git commit -m "feat(console): client-360 bundle assembler"
```

---

### Task 4: `/api/console/client-360` endpoint

**Files:**
- Modify: `app.py` (add a route near the other per-client console endpoints, e.g. just after `console_client_invoice` at ~`app.py:39340`)
- Test: `tests/test_console_client_360_api.py`

**Interfaces:**
- Consumes: `dashboard.client_360.bundle`, the existing `_bos_actor()` gate, `_db_lock`, `LOG_DB`.
- Produces: `GET /api/console/client-360?email=<email>` → `{"ok": True, ...bundle}` (200) or `{"ok": False, "error": "unauthorized"}` (401). With no email → `{"ok": True, "person": {...empty}, ...}` for an empty client is acceptable; simplest is to still run `bundle` with the empty email.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_client_360_api.py
import json
import pytest
import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_client_360_returns_bundle_shape(client, monkeypatch):
    # Gate open: _bos_actor returns a truthy actor (independent of rbac internals).
    monkeypatch.setattr(app_module, "_bos_actor", lambda: object())
    fake = {"person": {"name": "Test"}, "clinical": {"active": [], "suggested": []},
            "tests": [], "invoices": {"total_paid_cents": 0, "open_balance_cents": 0,
                                      "orders": [], "fmp": []},
            "comms": [], "process": {"source": None, "order_id": None, "stages": []}}
    monkeypatch.setattr(app_module.client_360, "bundle", lambda cx, email, **k: fake)
    r = client.get("/api/console/client-360?email=a@b.com")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["person"]["name"] == "Test"
    assert "invoices" in data and "process" in data


def test_client_360_requires_auth(client, monkeypatch):
    # Gate closed: _bos_actor returns None.
    monkeypatch.setattr(app_module, "_bos_actor", lambda: None)
    r = client.get("/api/console/client-360?email=a@b.com")
    assert r.status_code == 401
    assert r.get_json()["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_console_client_360_api.py -q`
Expected: FAIL — 404 (route missing) or `AttributeError` on `app_module.client_360`.

- [ ] **Step 3: Write minimal implementation**

First ensure the module is imported at the top of `app.py` alongside the other `from dashboard import …` lines (grep for an existing `from dashboard import` block; add `client_360`). If a bare `import` style is used, add `from dashboard import client_360`.

Then add the route (place after `console_client_invoice`):

```python
@app.route("/api/console/client-360", methods=["GET"])
def console_client_360():
    """The client-centered hub payload: person + clinical tags + tests +
    invoices + comms + current-process strip, all for one email. Read-only."""
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    email = (request.args.get("email") or "").strip().lower()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        data = client_360.bundle(cx, email)
    return jsonify({"ok": True, **data})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_console_client_360_api.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add app.py tests/test_console_client_360_api.py
git commit -m "feat(console): /api/console/client-360 endpoint"
```

---

### Task 5: `/console/client` page + route

**Files:**
- Create: `static/console-client.html`
- Modify: `app.py` (add the `/console/client` page route near `bos_crm_page` at ~`app.py:42443`)
- Test: `tests/test_console_client_page.py`

**Interfaces:**
- Consumes: `GET /api/console/client-360`, `GET /api/people?q=` (email autocomplete), `op-nav.js`.
- Produces: route `GET /console/client` serving `console-client.html` with no-store headers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_client_page.py
import app as app_module


def test_console_client_page_served():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.get("/console/client")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "client-360" in body            # the page fetches the bundle endpoint
    assert "op-nav.js" in body
    assert "no-store" in r.headers.get("Cache-Control", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_console_client_page.py -q`
Expected: FAIL — 404 (route missing).

- [ ] **Step 3a: Create the page**

Create `static/console-client.html`. It reads `?email=` (or `?pq=`), calls `/api/console/client-360`, and renders the header, folded sections (`<details>`), and the process strip. Uses the standard console key helper (localStorage `console_key`, `X-Console-Key` header, `?key=` seed) — copy the small key/gate script from `static/console-biofield-portal.html` for consistency.

```html
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Client — Console</title>
<style>
  body{font:15px/1.5 system-ui,-apple-system,sans-serif;margin:0;color:#1a2b26;background:#f6f8f7}
  .wrap{max-width:860px;margin:0 auto;padding:20px}
  h1{margin:.2em 0}
  .sub{color:#5a6b66;font-size:13px}
  details{background:#fff;border:1px solid #e0e6e3;border-radius:10px;margin:10px 0;padding:0 14px}
  summary{cursor:pointer;padding:12px 0;font-weight:600;list-style:none;display:flex;justify-content:space-between}
  summary::-webkit-details-marker{display:none}
  .head-note{color:#5a6b66;font-weight:400}
  .sec-body{padding:0 0 12px}
  table{width:100%;border-collapse:collapse;font-size:14px}
  td,th{padding:6px 8px;border-bottom:1px solid #eef2f0;text-align:left}
  .strip{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
  .stage{flex:1;min-width:120px;border:1px solid #e0e6e3;border-radius:10px;padding:10px;background:#fff;text-align:center}
  .stage.done{background:#eaf5ef;border-color:#2f6f5e}
  .stage .dot{font-size:18px}
  .stage a,.stage button{display:inline-block;margin-top:6px;font-size:12px}
  .pill{background:#eef2f0;border-radius:20px;padding:2px 8px;font-size:12px}
  .empty{color:#8a9a95}
  #picker input{padding:8px;width:260px}
</style>
</head>
<body>
<div class="wrap">
  <script src="/static/op-nav.js" data-active="people" data-sub="client"></script>
  <div id="gate"></div>
  <div id="picker" style="display:none">
    <h1>Open a client</h1>
    <input id="q" placeholder="name or email…" autocomplete="off">
    <ul id="results"></ul>
  </div>
  <div id="hub" style="display:none">
    <h1 id="c-name">—</h1>
    <p class="sub" id="c-meta"></p>

    <div class="strip" id="strip"></div>

    <details id="sec-clinical"><summary>Clinical <span class="head-note" id="h-clinical"></span></summary>
      <div class="sec-body" id="b-clinical"></div></details>
    <details id="sec-tests"><summary>Tests <span class="head-note" id="h-tests"></span></summary>
      <div class="sec-body" id="b-tests"></div></details>
    <details id="sec-invoices"><summary>Invoices <span class="head-note" id="h-invoices"></span></summary>
      <div class="sec-body" id="b-invoices"></div></details>
    <details id="sec-comms"><summary>Comms <span class="head-note" id="h-comms"></span></summary>
      <div class="sec-body" id="b-comms"></div></details>
  </div>
</div>
<script>
function key(){ return localStorage.getItem("console_key") || ""; }
(function(){ var k=new URLSearchParams(location.search).get("key"); if(k) localStorage.setItem("console_key",k); })();
function hdr(){ return {"X-Console-Key": key()}; }
function qsEmail(){ var p=new URLSearchParams(location.search); return (p.get("email")||p.get("pq")||"").trim(); }
function money(c){ return "$"+((c||0)/100).toFixed(2); }
function esc(s){ var d=document.createElement("div"); d.textContent=(s==null?"":s); return d.innerHTML; }

var SRC_ACTION_LABEL={biofield:"Open composer",scan:"Open FF draft",intake:"Open CRM",chat:"Open CRM"};

function renderStrip(p){
  var el=document.getElementById("strip"); el.innerHTML="";
  var email=qsEmail();
  p.stages.forEach(function(s){
    var d=document.createElement("div"); d.className="stage"+(s.done?" done":"");
    var btn="";
    if(s.action && s.action.kind==="link"){
      var href=s.action.target+"?key="+encodeURIComponent(key());
      if(s.key==="recommendation" && p.source) href+="&email="+encodeURIComponent(email);
      btn="<a href='"+href+"'>"+(SRC_ACTION_LABEL[s.source]||"Open")+"</a>";
    } else if(s.action && s.action.kind==="dispatch"){
      btn="<a href='/console/orders?key="+encodeURIComponent(key())+"'>Go</a>";
    }
    d.innerHTML="<div class=dot>"+(s.done?"✓":"○")+"</div><div>"+esc(s.label)+"</div>"+btn;
    el.appendChild(d);
  });
}

function renderClinical(c){
  document.getElementById("h-clinical").textContent =
    c.active.length+" active · "+c.suggested.length+" suggested";
  var rows=c.active.concat(c.suggested);
  document.getElementById("b-clinical").innerHTML = rows.length
    ? "<table>"+rows.map(function(t){return "<tr><td><code>"+esc(t.tag)+"</code></td><td>"+esc(t.axis)+"</td></tr>";}).join("")+"</table>"
    : "<p class=empty>No tags yet.</p>";
}
function renderTests(ts){
  document.getElementById("h-tests").textContent = ts.length
    ? ts.length+" tests · latest "+ts[0].date : "no tests";
  document.getElementById("b-tests").innerHTML = ts.length
    ? "<table>"+ts.map(function(t){return "<tr><td>"+esc(t.date)+"</td><td><span class=pill>"+esc(t.type)+"</span></td></tr>";}).join("")+"</table>"
    : "<p class=empty>No tests.</p>";
}
function renderInvoices(inv){
  document.getElementById("h-invoices").textContent =
    money(inv.total_paid_cents)+" paid · "+money(inv.open_balance_cents)+" open";
  var rows=inv.orders.map(function(o){
    return "<tr><td>"+esc((o.date||"").slice(0,10))+"</td><td>"+esc(o.status)+"</td><td>"+money(o.total_cents)+
      "</td><td>"+money(o.paid_cents)+"</td><td>"+money(o.balance_cents)+
      "</td><td><a href='"+o.edit_url+"?key="+encodeURIComponent(key())+"'>edit</a></td></tr>";});
  var fmp=(inv.fmp||[]).map(function(o){
    return "<tr><td>"+esc((o.date||"").slice(0,10))+"</td><td>FMP</td><td>"+
      (o.total!=null?money(Math.round(o.total*100)):"")+"</td><td></td><td>"+
      (o.outstanding!=null?money(Math.round(o.outstanding*100)):"")+"</td><td></td></tr>";});
  document.getElementById("b-invoices").innerHTML =
    (rows.length||fmp.length)
    ? "<table><tr><th>Date</th><th>Status</th><th>Total</th><th>Paid</th><th>Balance</th><th></th></tr>"+rows.join("")+fmp.join("")+"</table>"
    : "<p class=empty>No invoices.</p>";
}
function renderComms(cs){
  document.getElementById("h-comms").textContent = cs.length
    ? "last contact "+((cs[0].date||"").slice(0,10)||"—") : "no comms";
  document.getElementById("b-comms").innerHTML = cs.length
    ? "<table>"+cs.map(function(c){return "<tr><td>"+esc((c.date||"").slice(0,10))+"</td><td>"+esc(c.topic)+"</td><td><span class=pill>"+esc(c.source)+"</span></td></tr>";}).join("")+"</table>"
    : "<p class=empty>No comms.</p>";
}

function load(email){
  fetch("/api/console/client-360?email="+encodeURIComponent(email),{headers:hdr()})
    .then(function(r){return r.json();})
    .then(function(d){
      if(!d.ok){ document.getElementById("gate").innerHTML="<p class=empty>Enter a console key via ?key= to view.</p>"; return; }
      document.getElementById("picker").style.display="none";
      document.getElementById("hub").style.display="block";
      document.getElementById("c-name").textContent = d.person.name || email;
      document.getElementById("c-meta").textContent =
        [email, d.person.phone, d.person.location, d.person.profession].filter(Boolean).join(" · ");
      renderStrip(d.process); renderClinical(d.clinical); renderTests(d.tests);
      renderInvoices(d.invoices); renderComms(d.comms);
    });
}

function initPicker(){
  document.getElementById("picker").style.display="block";
  var q=document.getElementById("q"), out=document.getElementById("results");
  q.addEventListener("input",function(){
    if(q.value.trim().length<2){ out.innerHTML=""; return; }
    fetch("/api/people?q="+encodeURIComponent(q.value.trim())+"&limit=8",{headers:hdr()})
      .then(function(r){return r.json();})
      .then(function(d){
        var people=(d.people||d.results||d||[]);
        out.innerHTML=people.map(function(p){
          return "<li><a href='/console/client?email="+encodeURIComponent(p.email)+"&key="+encodeURIComponent(key())+"'>"+esc(p.name||p.email)+" — "+esc(p.email)+"</a></li>";}).join("");
      });
  });
}

var email=qsEmail();
if(email) load(email); else initPicker();
</script>
</body>
</html>
```

- [ ] **Step 3b: Add the route**

In `app.py`, near `bos_crm_page` (~`app.py:42443`):

```python
@app.route("/console/client")
def console_client_page():
    resp = send_from_directory(STATIC, "console-client.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_console_client_page.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Manual smoke (optional but recommended)**

Run the app locally, open `/console/client?email=<a real client email>&key=<console key>`, confirm the header, four folded sections, and the process strip render, and that the process-strip links carry the key.

- [ ] **Step 6: Commit**

```bash
cd ~/deploy-chat && git add static/console-client.html app.py tests/test_console_client_page.py
git commit -m "feat(console): client-centered hub page + /console/client route"
```

---

### Task 6: Navigation entry + deep-links into the hub

**Files:**
- Modify: `static/op-nav.js` (add `client` to the `people` sub-array, line ~55)
- Modify: `static/console-crm.html` (add an "Open client hub" link when a contact is loaded)
- Modify: `static/console-handoffs.html` (add a per-card link to `/console/client?email=`)
- Test: `tests/test_console_client_nav.py`

**Interfaces:**
- Consumes: nothing new. Pure wiring so operators can reach `/console/client?email=`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_client_nav.py
import pathlib

STATIC = pathlib.Path(__file__).resolve().parents[1] / "static"


def test_op_nav_has_client_subentry():
    js = (STATIC / "op-nav.js").read_text()
    assert '/console/client' in js
    assert 'id:"client"' in js or "id:'client'" in js


def test_crm_links_to_hub():
    html = (STATIC / "console-crm.html").read_text()
    assert "/console/client" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_console_client_nav.py -q`
Expected: FAIL — `/console/client` not present in op-nav.js / console-crm.html.

- [ ] **Step 3a: Add the nav sub-entry**

In `static/op-nav.js`, the `people` sub-array (line ~55) currently begins:

```javascript
      people:        [ {id:"crm",label:"CRM",href:"/console/crm"+qs}, {id:"members",label:"Members",href:"/console/members"+qs},
```

Insert a `client` entry as the first item so the hub leads the People sub-nav:

```javascript
      people:        [ {id:"client",label:"Client",href:"/console/client"+qs}, {id:"crm",label:"CRM",href:"/console/crm"+qs}, {id:"members",label:"Members",href:"/console/members"+qs},
```

- [ ] **Step 3b: Deep-link from CRM**

In `static/console-crm.html`, where the selected contact's email is known (the autocomplete-selected contact), add a link near the contact header. Locate the element that shows the chosen contact and add (adapt the variable name to the page's existing contact-email variable, e.g. `selectedEmail`):

```javascript
// after a contact email is chosen and shown:
var hub = document.getElementById("client-hub-link");
if (hub) {
  hub.href = "/console/client?email=" + encodeURIComponent(selectedEmail) + "&key=" + encodeURIComponent(key());
  hub.style.display = "inline";
}
```

And add the anchor in the contact panel markup:

```html
<a id="client-hub-link" style="display:none" class="btn">Open client hub &rarr;</a>
```

If the CRM page has no single contact-email variable to hook, add the anchor once and set its href from the autocomplete `onselect` handler that already exists (grep `/api/people` in the file to find it).

- [ ] **Step 3c: Deep-link from the handoffs pipeline card**

In `static/console-handoffs.html`, each pipeline card renders for a client email. Add a link in the card template pointing at the hub (find where the card's email is available, e.g. `c.email`):

```javascript
"<a href='/console/client?email=" + encodeURIComponent(c.email) + "&key=" + encodeURIComponent(key()) + "'>Client hub</a>"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_console_client_nav.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Full-module test + commit**

Run the console suite to confirm nothing regressed:

Run: `cd ~/deploy-chat && python -m pytest tests/test_console_client_360_api.py tests/test_console_client_page.py tests/test_console_client_nav.py tests/test_client_360_process.py tests/test_client_360_tags.py tests/test_client_360_bundle.py -q`
Expected: PASS (all).

```bash
cd ~/deploy-chat && git add static/op-nav.js static/console-crm.html static/console-handoffs.html tests/test_console_client_nav.py
git commit -m "feat(console): nav entry + CRM/handoffs deep-links into client hub"
```

---

## Post-implementation checkpoints (not code tasks)

- **Verify the synced prod `e4l.db` carries `client_clinical_tags`.** On prod, hit `/api/console/client-360?email=<known-tagged client>` and confirm the `clinical` block is populated. If it is empty for a client known to have tags, the e4l manifest sync omits the table — follow-up: add `client_clinical_tags` to the sync push (out of scope here; the page already degrades gracefully).
- **Confirm `/api/people` response shape** matches the picker's `d.people || d.results || d` fallback; tighten the picker if the real key differs.
- **CI:** run the full suite once (`python -m pytest -q`) before opening the PR to confirm no new known-failure entries are needed.
