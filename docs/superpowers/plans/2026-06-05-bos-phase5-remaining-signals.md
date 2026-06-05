# BOS Phase 5: Light up the remaining five Home cells

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the whole Home board live by registering Home signals for the five remaining gray modules (marketing, products, content, comms, b2b), each from real local data (SQLite or a DATA_DIR JSON file), defensive (gray on any error). Full domain actions for these modules remain future phases; this lights the cells.

**Architecture:** One new `dashboard/module_signals.py` registers five `@signal(...)` functions. SQLite-backed signals take the connection; file-backed signals (products, content) read a DATA_DIR JSON via a robust dual-path helper. All wrapped so a missing table/file returns gray. `app.py` imports the module at startup.

**Builds on:** the merged Business OS (spine + Home + Justus + Orders + Money + CRM). New branch `sess/ec0e1f15` off main, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Signal sources (from research):**
- marketing: `inbound_leads` WHERE source='scoreapp' AND status pending AND no outreach -> amber, else green.
- products: count of orderable products in `products.json` -> informational green (no stock data exists).
- content: count of pending concepts in `atlas-pending.json` -> amber, else green (clips are Pinecone-only -> deferred).
- comms: count of visible `calendar_events` in the next 48h -> amber heads-up, else green.
- b2b: count of active (`new`/`packed`) `orders` where source in (wholesale, dispensary) -> informational green (no local application queue; does NOT double-count the orders cell's lifecycle, this is a B2B-scoped read).

---

## File Structure

- `dashboard/module_signals.py` (new): the five `@signal` functions + the `_data_file` helper.
- `tests/test_bos_module_signals.py` (new): unit tests.
- `app.py` (modify): import `dashboard.module_signals` in the BOS startup block.

---

## Task 1: The five signals (`dashboard/module_signals.py`)

**Files:**
- Create: `dashboard/module_signals.py`
- Test: `tests/test_bos_module_signals.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bos_module_signals.py`:

```python
import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_marketing_signal(monkeypatch):
    from dashboard import module_signals as M, signals as S
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE inbound_leads (id INTEGER PRIMARY KEY, source TEXT, status TEXT, "
               "last_outbound_at TEXT, email TEXT)")
    cx.commit()
    assert M.marketing_signal(cx, None)["level"] == S.GREEN
    cx.execute("INSERT INTO inbound_leads (source, status, last_outbound_at, email) "
               "VALUES ('scoreapp', 'pending', '', 'a@b.com')")
    cx.execute("INSERT INTO inbound_leads (source, status, last_outbound_at, email) "
               "VALUES ('groovekart', 'pending', '', 'c@d.com')")  # not scoreapp -> excluded
    cx.commit()
    sig = M.marketing_signal(cx, None)
    assert sig["level"] == S.AMBER and sig["count"] == 1


def test_products_signal(monkeypatch, tmp_path):
    from dashboard import module_signals as M, signals as S
    (tmp_path / "products.json").write_text(json.dumps(
        {"products": {"a": {"name": "A"}, "b": {"name": "B", "info_only": True}}}))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    sig = M.products_signal(None, None)
    assert sig["level"] == S.GREEN and sig["count"] == 1  # info_only excluded


def test_content_signal(monkeypatch, tmp_path):
    from dashboard import module_signals as M, signals as S
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "atlas-pending.json").write_text(json.dumps({"concepts": [{"id": "1"}, {"id": "2"}]}))
    sig = M.content_signal(None, None)
    assert sig["level"] == S.AMBER and sig["count"] == 2
    (tmp_path / "atlas-pending.json").write_text(json.dumps({"concepts": []}))
    assert M.content_signal(None, None)["level"] == S.GREEN


def test_comms_signal():
    from dashboard import module_signals as M, signals as S
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE calendar_events (id INTEGER PRIMARY KEY, status TEXT, start TEXT)")
    cx.commit()
    assert M.comms_signal(cx, None)["level"] == S.GREEN
    soon = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    cx.execute("INSERT INTO calendar_events (status, start) VALUES ('visible', ?)", (soon,))
    cx.execute("INSERT INTO calendar_events (status, start) VALUES ('visible', ?)", (far,))
    cx.commit()
    sig = M.comms_signal(cx, None)
    assert sig["level"] == S.AMBER and sig["count"] == 1  # only the soon one


def test_b2b_signal():
    from dashboard import module_signals as M, signals as S
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, source TEXT, status TEXT)")
    cx.commit()
    assert M.b2b_signal(cx, None)["level"] == S.GREEN and M.b2b_signal(cx, None)["count"] == 0
    cx.execute("INSERT INTO orders (source, status) VALUES ('wholesale', 'new')")
    cx.execute("INSERT INTO orders (source, status) VALUES ('funnel', 'new')")  # not b2b
    cx.commit()
    sig = M.b2b_signal(cx, None)
    assert sig["level"] == S.GREEN and sig["count"] == 1


def test_all_defensive_gray_on_missing():
    from dashboard import module_signals as M, signals as S
    cx = sqlite3.connect(":memory:")  # no tables
    assert M.marketing_signal(cx, None)["level"] == S.GRAY
    assert M.comms_signal(cx, None)["level"] == S.GRAY
    assert M.b2b_signal(cx, None)["level"] == S.GRAY


def test_all_registered():
    from dashboard import module_signals as M  # noqa: F401
    from dashboard import signals as S
    for m in ("marketing", "products", "content", "comms", "b2b"):
        assert S.SIGNAL_REGISTRY.get(m) is not None, m
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_module_signals.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'dashboard.module_signals'`).

- [ ] **Step 3: Write the implementation**

Create `dashboard/module_signals.py`:

```python
"""Business-OS lightweight Home-board signals for the modules whose full domain
logic is a future phase. Each reads real local data (SQLite or a DATA_DIR JSON
file) and is defensive (gray on any error). Registers on import."""
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dashboard.signals import signal, RED, AMBER, GREEN, GRAY  # noqa: F401 (RED reserved)

_REPO = Path(__file__).resolve().parent.parent


def _data_file(name):
    """Find a DATA_DIR JSON file across the env-var path and the repo data dir."""
    for base in (os.environ.get("DATA_DIR"), str(_REPO / "data"), str(_REPO)):
        if not base:
            continue
        p = Path(base) / name
        if p.exists():
            return p
    return None


def _plural(n):
    return "s" if n != 1 else ""


@signal("marketing")
def marketing_signal(cx, actor=None):
    try:
        n = cx.execute(
            "SELECT COUNT(*) FROM inbound_leads "
            "WHERE source='scoreapp' AND (status IS NULL OR status='pending') "
            "  AND (last_outbound_at IS NULL OR last_outbound_at='')").fetchone()[0]
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    if n == 0:
        return {"level": GREEN, "summary": "No new quiz leads", "top_actions": [], "count": 0}
    return {"level": AMBER, "summary": f"{n} new quiz lead{_plural(n)} to reach",
            "top_actions": [{"label": "Open people", "href": "/console"}], "count": n}


@signal("products")
def products_signal(cx, actor=None):
    p = _data_file("products.json")
    if not p:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    try:
        prods = (json.loads(p.read_text()).get("products") or {})
        n = sum(1 for v in prods.values()
                if not (isinstance(v, dict) and v.get("info_only")))
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    return {"level": GREEN, "summary": f"{n} products in catalog", "top_actions": [], "count": n}


@signal("content")
def content_signal(cx, actor=None):
    p = _data_file("atlas-pending.json")
    if not p:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    try:
        n = len(json.loads(p.read_text()).get("concepts") or [])
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    if n == 0:
        return {"level": GREEN, "summary": "No concepts to review", "top_actions": [], "count": 0}
    return {"level": AMBER, "summary": f"{n} atlas concept{_plural(n)} to approve",
            "top_actions": [{"label": "Review atlas", "href": "/admin/atlas"}], "count": n}


@signal("comms")
def comms_signal(cx, actor=None):
    try:
        rows = cx.execute(
            "SELECT start FROM calendar_events WHERE status='visible'").fetchall()
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    now = datetime.now(timezone.utc)
    soon = now + timedelta(hours=48)
    n = 0
    for r in rows:
        s = r[0]
        if not s:
            continue
        try:
            if "T" in s or " " in s:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(s[:10])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if now <= dt <= soon:
                n += 1
        except Exception:
            continue
    if n == 0:
        return {"level": GREEN, "summary": "Nothing in the next 48h", "top_actions": [], "count": 0}
    return {"level": AMBER, "summary": f"{n} event{_plural(n)} in the next 48h",
            "top_actions": [{"label": "Open console", "href": "/console"}], "count": n}


@signal("b2b")
def b2b_signal(cx, actor=None):
    try:
        n = cx.execute(
            "SELECT COUNT(*) FROM orders "
            "WHERE source IN ('wholesale','dispensary') AND status IN ('new','packed')").fetchone()[0]
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    if n == 0:
        return {"level": GREEN, "summary": "No active B2B orders", "top_actions": [], "count": 0}
    return {"level": GREEN, "summary": f"{n} active practitioner order{_plural(n)}",
            "top_actions": [{"label": "Open orders", "href": "/console/orders"}], "count": n}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_module_signals.py -q`
Expected: 7 passed.

Run: `python3 -m pytest tests/test_bos_signals.py -q` (the new signals register, but the 1b test only asserts money==gray + that all nine cells return a valid color; the new cells read against an in-memory DB with no tables -> defensive gray, or products/content read the repo `data/` files -> green, both valid).
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/module_signals.py tests/test_bos_module_signals.py
git commit -m "feat(bos): home signals for marketing/products/content/comms/b2b"
```

---

## Task 2: Register in `app.py` (verified under doppler)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Import the module in the BOS startup block** (near `import dashboard.crm as _bos_crm`):

```python
import dashboard.module_signals as _bos_module_signals  # noqa: F401 (registers 5 cell signals)
```

- [ ] **Step 2: Compile + verify under doppler**

Run: `python3 -m py_compile app.py` (OK).
Run:
```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app, sqlite3
from dashboard import signals as S
cx = sqlite3.connect(app.LOG_DB); cx.row_factory=sqlite3.Row
cells = {c["module"]: c for c in S.aggregate_signals(cx, None)}
for m in ("marketing","products","content","comms","b2b"):
    print(f"{m}: {cells[m][\"level\"]} - {cells[m][\"summary\"]}")
gray = [m for m,c in cells.items() if c["level"]=="gray"]
print("still gray:", gray)
print("ALL_CELLS_OK")
PY'
rm -rf /tmp/bostest
```
Expected: prints each of the five cells' level + summary and `ALL_CELLS_OK`. Note: with a fresh temp DATA_DIR, products/content read the repo `data/` files if present (green/amber) and the SQLite ones read the freshly-created tables (green/empty); the real production cells reflect the live data after deploy.

Run: `python3 -m pytest tests/test_bos_module_signals.py tests/test_bos_signals.py tests/test_bos_spine.py -q` (green).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(bos): wire the five remaining home cell signals"
```

---

## Self-Review

**Spec coverage:** all nine Home cells now have a real `signal()` (orders/money/crm shipped earlier; marketing/products/content/comms/b2b here). Each reads local data and is defensive.

**Honest scoping:** products and b2b are informational green (no local work-queue data exists for stock or practitioner-applications); content covers Atlas only (clips are Pinecone-only -> deferred); the full domain ACTIONS for these five modules are future phases.

**Placeholder scan:** none.

**Type consistency:** every signal returns the `{level, summary, top_actions, count}` cell shape; the module keys (marketing/products/content/comms/b2b) match `signals.MODULES`.
