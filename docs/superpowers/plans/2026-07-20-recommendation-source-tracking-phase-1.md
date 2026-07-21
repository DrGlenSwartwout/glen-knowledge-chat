# Recommendation Source Tracking — Phase 1 (the spine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the recommendation-provenance spine — a source registry, an append-only `recommendation_events` log, ingest from biofield reveals + paid orders, per-product aggregates with a client hide flag — and surface it on the operator client-360 hub, proving ingest→aggregate→display end-to-end.

**Architecture:** Two new pure modules — `dashboard/recommendation_sources.py` (a static extensible registry) and `dashboard/recommendation_events.py` (table + idempotent `record_event` + `ingest_biofield`/`ingest_purchased` + `product_sources` aggregate + hide flag). The existing `dashboard/client_360.py` gains a multi-badge process strip and a `recommendations` block in its bundle; the `/api/console/client-360` endpoint lazily ingests the client on read; `static/console-client.html` renders a Recommendations section. No client-facing UI, no order-creation changes, no marketing integrations.

**Tech Stack:** Python 3 / Flask (`app.py`), `dashboard/*.py`, SQLite (`LOG_DB` = `chat_log.db`), pytest, vanilla-JS static page.

## Global Constraints

- **Canonical `product_key` = the storefront `slug`.** All three data sources normalize to it: biofield reveal remedies carry `slug` (name-resolved at write), order lines carry `slug`. Events with a falsy slug are SKIPPED (never fabricate a key from a display name).
- **Ingest ONLY biofield + purchased in Phase 1.** `biofield` = each remedy in each `biofield_reveals` reveal (recommendation-generated ✓). `purchased` = each line of each PAID order (`pay_status=='paid'`). Do NOT ingest `scan`/`ff_match_drafts` (its counting rule is client-engagement, not match-generation) or intake/chat (no product data yet).
- **Idempotent ingest.** `recommendation_events` has a UNIQUE index `(client_email, product_key, source_key, origin_ref)`; all inserts use `INSERT OR IGNORE`. `origin_ref` = `scan_date` for biofield, `order_id` for purchased (both non-null, so dedup works). Re-running ingest writes nothing new.
- **Email is the lowercased join key** everywhere.
- **Append-only.** Never UPDATE/DELETE events. "Hide" is a separate `recommendation_hidden` row; data is kept.
- **Read-only `bundle()` stays read-only** — lazy ingest happens in the ENDPOINT before `bundle`, not inside it. After ingest, the endpoint resets `cx.row_factory = sqlite3.Row` before `bundle` (a reader may mutate it — same lesson as the earlier fmp row_factory fix).
- **Back-compat:** `process_strip` keeps its top-level `source` (= first detected) so existing tests/pages don't break; it ADDS a `sources` list.
- **CI has a known_failures ratchet; never run the bare full suite (it sends live email).** Run the named feature test files.

---

### Task 1: Source registry module

**Files:**
- Create: `dashboard/recommendation_sources.py`
- Test: `tests/test_recommendation_sources.py`

**Interfaces:**
- Produces: `recommendation_sources.RECOMMENDATION_SOURCES` (dict `key -> {label, icon, kind}`), `get_source(key) -> dict|None`, `known_source(key) -> bool`. `kind` ∈ `{"clinical","engagement"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recommendation_sources.py
from dashboard import recommendation_sources as rs


def test_registry_has_the_ten_sources_with_shapes():
    for key in ["biofield", "intake", "scan", "chat", "self",
                "email", "newsletter", "ads", "social", "purchased"]:
        s = rs.get_source(key)
        assert s is not None, key
        assert s["label"] and s["icon"]
        assert s["kind"] in ("clinical", "engagement")


def test_clinical_vs_engagement_kinds():
    assert rs.get_source("biofield")["kind"] == "clinical"
    assert rs.get_source("intake")["kind"] == "clinical"
    assert rs.get_source("scan")["kind"] == "engagement"
    assert rs.get_source("purchased")["kind"] == "engagement"


def test_known_source():
    assert rs.known_source("biofield") is True
    assert rs.known_source("nope") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_sources.py -q`
Expected: FAIL — `ModuleNotFoundError: dashboard.recommendation_sources`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/recommendation_sources.py
"""Extensible registry of recommendation sources. Each source: a display label,
an icon (shown in the portal with a count at its center), and a kind:
  clinical   -> the recommendation being generated is the counted action
  engagement -> a client action (click/add/order) is the counted action
Adding a source later is a dict entry, not a schema change."""

RECOMMENDATION_SOURCES = {
    "biofield":   {"label": "Biofield",   "icon": "📡", "kind": "clinical"},
    "intake":     {"label": "Intake",     "icon": "📝", "kind": "clinical"},
    "scan":       {"label": "Scan",       "icon": "🔬", "kind": "engagement"},
    "chat":       {"label": "Chat",       "icon": "💬", "kind": "engagement"},
    "self":       {"label": "Self",       "icon": "🛒", "kind": "engagement"},
    "email":      {"label": "Email",      "icon": "✉️", "kind": "engagement"},
    "newsletter": {"label": "Newsletter", "icon": "📰", "kind": "engagement"},
    "ads":        {"label": "Ads",        "icon": "📣", "kind": "engagement"},
    "social":     {"label": "Social",     "icon": "📱", "kind": "engagement"},
    "purchased":  {"label": "Purchased",  "icon": "✅", "kind": "engagement"},
}


def get_source(key):
    return RECOMMENDATION_SOURCES.get(key)


def known_source(key):
    return key in RECOMMENDATION_SOURCES
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_sources.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/recommendation_sources.py tests/test_recommendation_sources.py
git commit -m "feat(rec): recommendation source registry"
```

---

### Task 2: Events table + idempotent record_event

**Files:**
- Create: `dashboard/recommendation_events.py`
- Test: `tests/test_recommendation_events_record.py`

**Interfaces:**
- Produces: `recommendation_events.init_recommendation_events(cx)`; `record_event(cx, email, product_key, source_key, *, occurred_at, origin_ref) -> bool` (True iff a new row was inserted); `list_events(cx, email) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recommendation_events_record.py
import sqlite3
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx)
    return cx


def test_record_inserts_then_dedups():
    cx = _cx()
    assert re.record_event(cx, "A@B.com", "neuro-magnesium", "biofield",
                           occurred_at="2026-07-01", origin_ref="2026-07-01") is True
    # same (email, product, source, origin_ref) -> ignored
    assert re.record_event(cx, "a@b.com", "neuro-magnesium", "biofield",
                           occurred_at="2026-07-01", origin_ref="2026-07-01") is False
    # different origin_ref -> new event
    assert re.record_event(cx, "a@b.com", "neuro-magnesium", "biofield",
                           occurred_at="2026-07-08", origin_ref="2026-07-08") is True
    rows = re.list_events(cx, "a@b.com")
    assert len(rows) == 2
    assert rows[0]["product_key"] == "neuro-magnesium"


def test_record_rejects_empty_email_or_slug():
    cx = _cx()
    assert re.record_event(cx, "", "x", "biofield", occurred_at="d", origin_ref="r") is False
    assert re.record_event(cx, "a@b.com", "", "biofield", occurred_at="d", origin_ref="r") is False
    assert re.list_events(cx, "a@b.com") == []


def test_email_lowercased():
    cx = _cx()
    re.record_event(cx, "MixedCase@X.com", "slug", "purchased", occurred_at="d", origin_ref="7")
    assert len(re.list_events(cx, "mixedcase@x.com")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_events_record.py -q`
Expected: FAIL — `ModuleNotFoundError: dashboard.recommendation_events`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/recommendation_events.py
"""Append-only per-client recommendation-provenance log + aggregates.
One row per counted action: (client_email, product_key=slug, source_key, occurred_at,
origin_ref). Idempotent on (client_email, product_key, source_key, origin_ref).
Pure: functions take an open sqlite connection."""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_recommendation_events(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_email TEXT NOT NULL,
            product_key  TEXT NOT NULL,
            source_key   TEXT NOT NULL,
            occurred_at  TEXT,
            origin_ref   TEXT NOT NULL DEFAULT '',
            created_at   TEXT NOT NULL
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_rec_events "
               "ON recommendation_events(client_email, product_key, source_key, origin_ref)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_rec_events_email "
               "ON recommendation_events(client_email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_hidden (
            client_email TEXT NOT NULL,
            product_key  TEXT NOT NULL,
            hidden_at    TEXT,
            PRIMARY KEY (client_email, product_key)
        )""")
    cx.commit()


def record_event(cx, email, product_key, source_key, *, occurred_at, origin_ref):
    e = (email or "").strip().lower()
    pk = (product_key or "").strip()
    sk = (source_key or "").strip()
    if not e or not pk or not sk:
        return False
    cur = cx.execute(
        "INSERT OR IGNORE INTO recommendation_events "
        "(client_email, product_key, source_key, occurred_at, origin_ref, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (e, pk, sk, occurred_at, str(origin_ref or ""), _now()))
    cx.commit()
    return cur.rowcount == 1


def list_events(cx, email):
    e = (email or "").strip().lower()
    rows = cx.execute(
        "SELECT product_key, source_key, occurred_at, origin_ref FROM recommendation_events "
        "WHERE client_email=? ORDER BY id", (e,)).fetchall()
    return [{"product_key": r[0], "source_key": r[1], "occurred_at": r[2], "origin_ref": r[3]}
            for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_events_record.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/recommendation_events.py tests/test_recommendation_events_record.py
git commit -m "feat(rec): recommendation_events table + idempotent record_event"
```

---

### Task 3: Ingest from biofield reveals + paid orders

**Files:**
- Modify: `dashboard/recommendation_events.py`
- Test: `tests/test_recommendation_ingest.py`

**Interfaces:**
- Consumes: `dashboard.biofield_reveals.list_for_email(cx, email)` (rows with `remedies: [{slug, name, ...}]`, `scan_date`); `dashboard.orders.list_orders_by_email(cx, email)` (rows with `items: [{slug, ...}]`, `pay_status`, `paid_at`, `id`).
- Produces: `ingest_biofield(cx, email) -> int` (new events), `ingest_purchased(cx, email) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recommendation_ingest.py
import json, sqlite3
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    re.init_recommendation_events(cx)
    # minimal biofield_reveals + orders tables the readers use
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, "
               "scan_date TEXT, interpretation_json TEXT DEFAULT '{}', remedies_json TEXT DEFAULT '[]', "
               "first_approved INTEGER DEFAULT 0, token_hash TEXT, approved_at TEXT, approved_by TEXT, "
               "created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '', dropped TEXT DEFAULT '[]', "
               "layers_json TEXT DEFAULT '[]', notified_at TEXT, requested_at TEXT)")
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT, "
               "external_ref TEXT, email TEXT, name TEXT, items_json TEXT, total_cents INTEGER, "
               "status TEXT, pay_status TEXT, paid_at TEXT)")
    return cx


def test_ingest_biofield_one_event_per_remedy_per_reveal(monkeypatch):
    cx = _cx()
    cx.execute("INSERT INTO biofield_reveals (email, scan_date, remedies_json) VALUES (?,?,?)",
               ("a@b.com", "2026-07-01",
                json.dumps([{"name": "Neuro Magnesium", "slug": "neuro-magnesium"},
                            {"name": "No Slug", "slug": ""}])))
    cx.execute("INSERT INTO biofield_reveals (email, scan_date, remedies_json) VALUES (?,?,?)",
               ("a@b.com", "2026-07-08",
                json.dumps([{"name": "Neuro Magnesium", "slug": "neuro-magnesium"}])))
    n = re.ingest_biofield(cx, "a@b.com")
    assert n == 2                       # slug="" skipped; two distinct scan_dates for neuro-magnesium
    ev = [e for e in re.list_events(cx, "a@b.com")]
    assert all(e["source_key"] == "biofield" for e in ev)
    assert {e["origin_ref"] for e in ev} == {"2026-07-01", "2026-07-08"}
    assert re.ingest_biofield(cx, "a@b.com") == 0     # idempotent


def test_ingest_purchased_paid_only_by_line(monkeypatch):
    cx = _cx()
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_at, status) VALUES (?,?,?,?,?)",
               ("a@b.com", json.dumps([{"slug": "neuro-magnesium", "qty": 2},
                                       {"slug": "", "qty": 1}]), "paid", "2026-07-10", "done"))
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_at, status) VALUES (?,?,?,?,?)",
               ("a@b.com", json.dumps([{"slug": "immune-modulation"}]), "unpaid", None, "new"))
    n = re.ingest_purchased(cx, "a@b.com")
    assert n == 1                       # only the paid order's slugged line
    ev = re.list_events(cx, "a@b.com")
    assert ev[0]["source_key"] == "purchased" and ev[0]["product_key"] == "neuro-magnesium"
    assert re.ingest_purchased(cx, "a@b.com") == 0     # idempotent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_ingest.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'ingest_biofield'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/recommendation_events.py`:

```python
def ingest_biofield(cx, email):
    """One biofield event per (remedy slug, reveal). occurred_at/origin_ref = scan_date."""
    from dashboard import biofield_reveals
    try:
        rows = biofield_reveals.list_for_email(cx, email)
    except Exception:
        return 0
    n = 0
    for r in rows:
        sd = (r.get("scan_date") or "")
        for rem in (r.get("remedies") or []):
            slug = (rem.get("slug") or "").strip()
            if not slug:
                continue
            if record_event(cx, email, slug, "biofield", occurred_at=sd, origin_ref=sd):
                n += 1
    return n


def ingest_purchased(cx, email):
    """One purchased event per (line slug, PAID order). occurred_at = paid_at; origin_ref = order id."""
    from dashboard import orders
    try:
        rows = orders.list_orders_by_email(cx, email)
    except Exception:
        return 0
    n = 0
    for o in rows:
        if (o.get("pay_status") or "").strip().lower() != "paid":
            continue
        oid = o.get("id")
        occ = o.get("paid_at") or o.get("created_at") or ""
        for line in (o.get("items") or []):
            slug = (line.get("slug") or "").strip()
            if not slug:
                continue
            if record_event(cx, email, slug, "purchased", occurred_at=occ, origin_ref=str(oid)):
                n += 1
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_ingest.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/recommendation_events.py tests/test_recommendation_ingest.py
git commit -m "feat(rec): ingest biofield reveals + paid orders (idempotent)"
```

---

### Task 4: Aggregates (product_sources) + hide flag

**Files:**
- Modify: `dashboard/recommendation_events.py`
- Test: `tests/test_recommendation_aggregate.py`

**Interfaces:**
- Produces: `product_sources(cx, email) -> list[dict]` where each dict is `{product_key, hidden: bool, sources: [{source, count, first_touch, last_touch}]}` with `sources` ordered by `first_touch` ascending; `set_hidden(cx, email, product_key, hidden=True) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recommendation_aggregate.py
import sqlite3
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx)
    return cx


def test_product_sources_counts_first_last_and_order():
    cx = _cx()
    # neuro-magnesium: self (first, 2026-06), then biofield twice (2026-07-01, 2026-07-08)
    re.record_event(cx, "a@b.com", "neuro-magnesium", "self", occurred_at="2026-06-01", origin_ref="p1")
    re.record_event(cx, "a@b.com", "neuro-magnesium", "biofield", occurred_at="2026-07-01", origin_ref="2026-07-01")
    re.record_event(cx, "a@b.com", "neuro-magnesium", "biofield", occurred_at="2026-07-08", origin_ref="2026-07-08")
    prods = re.product_sources(cx, "a@b.com")
    p = next(x for x in prods if x["product_key"] == "neuro-magnesium")
    assert p["hidden"] is False
    # icon order by first_touch: self (June) before biofield (July)
    assert [s["source"] for s in p["sources"]] == ["self", "biofield"]
    bf = next(s for s in p["sources"] if s["source"] == "biofield")
    assert bf["count"] == 2
    assert bf["first_touch"] == "2026-07-01" and bf["last_touch"] == "2026-07-08"


def test_hidden_flag():
    cx = _cx()
    re.record_event(cx, "a@b.com", "slugx", "purchased", occurred_at="d", origin_ref="1")
    re.set_hidden(cx, "a@b.com", "slugx", True)
    assert re.product_sources(cx, "a@b.com")[0]["hidden"] is True
    re.set_hidden(cx, "a@b.com", "slugx", False)
    assert re.product_sources(cx, "a@b.com")[0]["hidden"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_aggregate.py -q`
Expected: FAIL — `AttributeError: ... 'product_sources'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/recommendation_events.py`:

```python
def set_hidden(cx, email, product_key, hidden=True):
    e = (email or "").strip().lower()
    pk = (product_key or "").strip()
    if not e or not pk:
        return
    if hidden:
        cx.execute("INSERT OR REPLACE INTO recommendation_hidden "
                   "(client_email, product_key, hidden_at) VALUES (?,?,?)", (e, pk, _now()))
    else:
        cx.execute("DELETE FROM recommendation_hidden WHERE client_email=? AND product_key=?", (e, pk))
    cx.commit()


def product_sources(cx, email):
    """Per product: its sources (each with count, first_touch, last_touch), ordered by
    first_touch (icon order), plus a hidden flag. Callers sort/limit products for display."""
    e = (email or "").strip().lower()
    rows = cx.execute(
        "SELECT product_key, source_key, COUNT(*) n, MIN(occurred_at) ft, MAX(occurred_at) lt "
        "FROM recommendation_events WHERE client_email=? GROUP BY product_key, source_key",
        (e,)).fetchall()
    hidden = {r[0] for r in cx.execute(
        "SELECT product_key FROM recommendation_hidden WHERE client_email=?", (e,)).fetchall()}
    prods = {}
    for pk, sk, n, ft, lt in rows:
        p = prods.setdefault(pk, {"product_key": pk, "hidden": pk in hidden, "sources": []})
        p["sources"].append({"source": sk, "count": int(n),
                             "first_touch": ft or "", "last_touch": lt or ""})
    out = []
    for p in prods.values():
        p["sources"].sort(key=lambda s: s["first_touch"])
        out.append(p)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_aggregate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/recommendation_events.py tests/test_recommendation_aggregate.py
git commit -m "feat(rec): product_sources aggregate + hide flag"
```

---

### Task 5: client_360 multi-badge strip + recommendations in bundle

**Files:**
- Modify: `dashboard/client_360.py`
- Test: `tests/test_client_360_process.py` (update existing), `tests/test_client_360_bundle.py` (extend)

**Interfaces:**
- Consumes: `recommendation_events.product_sources`.
- Produces: `process_strip` now returns a `sources` list on the result and on the `recommendation` stage (keeps top-level `source` = first, for back-compat); `bundle` gains a `recommendations` key (= `product_sources(cx, email)`).

- [ ] **Step 1: Write the failing test** (add to `tests/test_client_360_process.py`)

```python
def test_process_strip_returns_multi_sources():
    from dashboard import client_360
    import sqlite3
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, status TEXT, pay_status TEXT, invoice_sent_at TEXT)")
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY, email TEXT, scan_date TEXT)")
    cx.execute("CREATE TABLE ff_match_drafts (email TEXT, scan_date TEXT, status TEXT)")
    cx.execute("CREATE TABLE intake_responses (email TEXT PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE inquiries (id INTEGER PRIMARY KEY, client_email TEXT)")
    cx.execute("INSERT INTO biofield_reveals VALUES (1,'a@b.com','2026-07-01')")
    cx.execute("INSERT INTO ff_match_drafts VALUES ('a@b.com','2026-07-01','draft')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["sources"] == ["biofield", "scan"]     # all present, priority order
    assert res["source"] == "biofield"                # back-compat: first
    rec = next(s for s in res["stages"] if s["key"] == "recommendation")
    assert rec["sources"] == ["biofield", "scan"] and rec["done"] is True
```

Also extend `tests/test_client_360_bundle.py::test_bundle_shape_and_invoice_math` to assert `"recommendations" in b` and that it's a list.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_client_360_process.py::test_process_strip_returns_multi_sources -q`
Expected: FAIL — `KeyError: 'sources'`.

- [ ] **Step 3: Write minimal implementation**

In `dashboard/client_360.py`, add a plural detector, switch `process_strip` to it, and **delete the now-unused `_detect_source`** (its only caller is `process_strip`; leaving it is dead code a reviewer will flag):

```python
def _detect_sources(cx, email):
    """All applicable sources by presence, in priority order (multi-badge)."""
    out = []
    if _exists(cx, "SELECT 1 FROM biofield_reveals WHERE lower(email)=? LIMIT 1", (email,)):
        out.append("biofield")
    if _exists(cx, "SELECT 1 FROM ff_match_drafts WHERE lower(email)=? LIMIT 1", (email,)):
        out.append("scan")
    if _exists(cx, "SELECT 1 FROM intake_responses WHERE lower(email)=? AND status='submitted' LIMIT 1", (email,)):
        out.append("intake")
    if _exists(cx, "SELECT 1 FROM inquiries WHERE lower(client_email)=? LIMIT 1", (email,)):
        out.append("chat")
    return out
```

In `process_strip`, replace the single-source lines:

```python
    sources = _detect_sources(cx, e)
    source = sources[0] if sources else None
```

Change the `recommendation` stage dict and the return to carry `sources`:

```python
        {"key": "recommendation",
         "label": _SOURCE_LABEL.get(source, "Recommendation"),
         "done": bool(sources), "source": source, "sources": sources, "action": rec_action},
```

```python
    return {"source": source, "sources": sources, "order_id": oid, "stages": stages}
```

In `bundle`, add the recommendations block (read-only):

```python
def _recommendations(cx, email):
    from dashboard import recommendation_events
    try:
        return recommendation_events.product_sources(cx, email)
    except Exception:
        return []
```

and add to the returned dict: `"recommendations": _recommendations(cx, e),`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_client_360_process.py tests/test_client_360_bundle.py -q`
Expected: PASS (existing tests still green — top-level `source` preserved — plus the new ones).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/client_360.py tests/test_client_360_process.py tests/test_client_360_bundle.py
git commit -m "feat(rec): multi-badge process strip + recommendations in client-360 bundle"
```

---

### Task 6: Endpoint lazy-ingest + startup init + page Recommendations section

**Files:**
- Modify: `app.py` (startup init near `_init_bos_orders` ~line 39134; lazy ingest in `console_client_360`)
- Modify: `static/console-client.html`
- Test: `tests/test_recommendation_endpoint.py`

**Interfaces:**
- Consumes: `recommendation_events.init_recommendation_events`, `ingest_biofield`, `ingest_purchased`; `client_360.bundle`.
- Produces: `/api/console/client-360?email=` now lazily ingests the client and returns `recommendations` in the payload; the page renders a Recommendations section + multi-badge strip.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recommendation_endpoint.py
import json, sqlite3
import app as app_module
from dashboard import recommendation_events as re


def test_endpoint_lazy_ingests_and_returns_recommendations(monkeypatch, tmp_path):
    # Point LOG_DB at a temp db seeded with a paid order, verify the endpoint ingests it.
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT, "
               "external_ref TEXT, email TEXT, name TEXT, items_json TEXT, total_cents INTEGER, "
               "status TEXT, pay_status TEXT, paid_at TEXT)")
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_at, status) VALUES (?,?,?,?,?)",
               ("a@b.com", json.dumps([{"slug": "neuro-magnesium"}]), "paid", "2026-07-10", "done"))
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_bos_actor", lambda: object())
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.get("/api/console/client-360?email=a@b.com")
    assert r.status_code == 200
    data = r.get_json()
    recs = data["recommendations"]
    assert any(p["product_key"] == "neuro-magnesium"
               and any(s["source"] == "purchased" for s in p["sources"]) for p in recs)
```

Note: this test opens the real app; run with `doppler run -- python3 -m pytest ...`. If `LOG_DB` is consumed as a module constant inside the endpoint, confirm the monkeypatch target matches how the endpoint reads it (it uses the module-global `LOG_DB`); adjust the patch target if the endpoint captured it differently.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_recommendation_endpoint.py -q`
Expected: FAIL — `KeyError: 'recommendations'` (endpoint not yet ingesting / bundle not yet including it in this path) or table-missing error.

- [ ] **Step 3a: Startup init**

In `app.py`, near `_init_bos_orders()` (~line 39134), add:

```python
def _init_recommendation_events():
    from dashboard.recommendation_events import init_recommendation_events
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        init_recommendation_events(cx)
_init_recommendation_events()
```

Add the import with the other `from dashboard import …` lines: `recommendation_events`.

- [ ] **Step 3b: Lazy ingest in the endpoint**

In `console_client_360` (the `/api/console/client-360` handler), between opening the connection and calling `bundle`, ingest then reset row_factory:

```python
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        client_360_recs = None
        if email:
            try:
                recommendation_events.init_recommendation_events(cx)
                recommendation_events.ingest_biofield(cx, email)
                recommendation_events.ingest_purchased(cx, email)
            except Exception:
                pass
        cx.row_factory = sqlite3.Row   # ingest readers may reset it; restore before bundle
        data = client_360.bundle(cx, email)
    return jsonify({"ok": True, **data})
```

(`bundle` already includes `recommendations` from Task 5.)

- [ ] **Step 3c: Page Recommendations section + multi-badge strip**

In `static/console-client.html`:
- In `renderStrip`, render the recommendation stage's `s.sources` as multiple badges when present (fall back to the single `s.source`). Minimal: if `s.key==="recommendation" && s.sources && s.sources.length`, build the badge label from `s.sources` joined.
- Add a **Recommendations** section (a folded `<details>` like the others) that reads `d.recommendations` and lists each product with its source icons + counts, sorted per the display rule (count desc, then last_touch desc), top 5 with a show-more. Use the icon map inline:

```javascript
var SRC_ICON = {biofield:"📡", intake:"📝", scan:"🔬", chat:"💬", self:"🛒",
                email:"✉️", newsletter:"📰", ads:"📣", social:"📱", purchased:"✅"};
function renderRecommendations(recs){
  var host = document.getElementById("b-recs");
  if(!recs || !recs.length){ document.getElementById("h-recs").textContent="none";
    host.innerHTML="<p class=empty>No recommendations yet.</p>"; return; }
  // rank products by top source count, then recency
  recs.forEach(function(p){
    p._max = Math.max.apply(null, p.sources.map(function(s){return s.count;}));
    p._recent = p.sources.map(function(s){return s.last_touch;}).sort().slice(-1)[0]||"";
  });
  recs.sort(function(a,b){ return b._max-a._max || (a._recent<b._recent?1:-1); });
  document.getElementById("h-recs").textContent = recs.length+" products";
  host.innerHTML = "<table>"+recs.slice(0,5).map(function(p){
    var icons = p.sources.map(function(s){
      return "<span title='"+esc(s.source)+" ×"+s.count+" (first "+esc(s.first_touch)+")'>"+
             (SRC_ICON[s.source]||"•")+esc(String(s.count))+"</span>";}).join(" ");
    return "<tr><td>"+esc(p.product_key)+"</td><td>"+icons+"</td></tr>";}).join("")+"</table>"
    + (recs.length>5 ? "<p class=empty>+"+(recs.length-5)+" more</p>" : "");
}
```

Add the section markup near the other `<details>` sections:

```html
<details id="sec-recs"><summary>Recommendations <span class="head-note" id="h-recs"></span></summary>
  <div class="sec-body" id="b-recs"></div></details>
```

And call `renderRecommendations(d.recommendations)` inside the existing `load()` success handler (alongside the other render calls).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_recommendation_endpoint.py tests/test_console_client_page.py -q`
Expected: PASS.

- [ ] **Step 5: Full feature run + commit**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_recommendation_sources.py tests/test_recommendation_events_record.py tests/test_recommendation_ingest.py tests/test_recommendation_aggregate.py tests/test_client_360_process.py tests/test_client_360_bundle.py tests/test_recommendation_endpoint.py tests/test_console_client_page.py -q`
Expected: PASS (all).

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py static/console-client.html tests/test_recommendation_endpoint.py
git commit -m "feat(rec): lazy-ingest endpoint + startup init + client-360 Recommendations section"
```

---

## Self-review checklist (controller, before dispatch)

- Spec coverage: registry (T1), event log (T2), ingest biofield+purchased (T3), aggregates+hide (T4), operator multi-badge + bundle recommendations (T5), lazy-ingest endpoint + operator Recommendations display (T6). Scan/intake/chat/self/marketing correctly deferred per the counting-rule finding.
- Product identity: every ingest keys on `slug`, skips falsy — matches the "canonical key = slug" constraint.
- Idempotency: UNIQUE index + `INSERT OR IGNORE`; re-run tests assert 0 new.
- row_factory: endpoint resets to `Row` after ingest, before `bundle`.
- No money-path/order-creation changes; no client UI; no bare full-suite runs.

## Post-Phase-1 checkpoint

Hit `/api/console/client-360?email=<a client with reveals+paid orders>` and confirm the Recommendations section shows their products with biofield/purchased icons + counts, and the process strip shows multi-badges. Then Phase 2 (client portal + order-line source capture) builds on `recommendation_events` + `product_sources`.
