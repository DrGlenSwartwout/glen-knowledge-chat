# Record-Level Deep-Links — Phase 3 (Orders) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make order mentions in the clients-pipeline briefing link to their card on `/console/orders` (scrolled-to + flash-highlighted).

**Architecture:** Reuses the Phase-1/2 citation-token registry. Add open orders to the snapshot; mint `type:"order"` linkables (`/console/orders?order=<id>`); the existing client resolver + `recNavigate` carry them with NO dashboard change. Only new UI is the orders-board card highlight.

**Tech Stack:** Python 3 (plain-function modules), pytest, vanilla JS in `static/console-orders.html`.

**Spec:** `docs/superpowers/specs/2026-06-26-record-level-deep-links-phase3-orders.md`
**Builds on (merged):** Phase 1 (#349/#350), Phase 2 (#352) — `briefing_links.py` (`person_url`, `invoice_url`, `build_linkables` with people+invoice passes), the links sidecar, `dashboard.html` resolver+`recNavigate`, and the `console-money.html` `data-inv`/`?invoice=`/`inv-flash` highlight pattern.

## Global Constraints

- **Reuse the mechanism — no dashboard/app changes.** Do NOT touch `static/dashboard.html`, `resolveRefLinks`, `recNavigate`, or `app.py`. Order links are registry entries resolving to a URL.
- **No console key in stored files or the resolved href.** `order_url` returns a bare `/console/orders?order=<id>`; the key is appended at click time by `recNavigate`.
- **Graceful degradation.** `snapshot["orders"]` may be a list (success) OR a `{"_error": …}` dict (off-prod / DB failure). No order linkables minted on `_error`/empty; no crash. Unknown refs already unwrap client-side.
- **Needs-attention filter:** open orders only = `status NOT IN {"shipped","delivered","done","cancelled"}` (i.e. `proposed/confirmed/new/packed`), newest first, capped 20. `status` + `pay_status` travel so the LLM separates unpaid carts from unshipped orders.
- **Same-table join:** briefing orders + board both read the local `orders` table by `orders.id` (no Phase-2-style mismatch).
- **Ref numbering:** people → invoices → orders, shared counter, dedup by url.
- **Scope:** orders only (payments cut); clients-pipeline card only.

---

### Task 1: `briefing_links.py` — order linkables

**Files:**
- Modify: `dashboard/briefing_links.py` (add `order_url`, `_iter_order_records`, an order pass in `build_linkables`)
- Test: `tests/test_briefing_links.py` (append)

**Interfaces:**
- Consumes: existing `build_linkables` mint/dedup, `_iter_invoice_records` guard pattern.
- Produces: `order_url(id) -> "/console/orders?order=<urlencoded-id>"`; `build_linkables` also stamps `ref` on `snapshot["orders"]` rows and adds `{type:"order", display, url}` entries (people→invoices→orders order).

- [ ] **Step 1: Write the failing test** (append to `tests/test_briefing_links.py`)

```python
def test_order_url_encodes_id():
    assert bl.order_url("42") == "/console/orders?order=42"


def test_build_linkables_mints_order_links():
    snap = {"orders": [
        {"id": 42, "name": "Jane Doe", "email": "jane@x.com", "status": "new",
         "pay_status": "unpaid", "total_cents": 6997},
        {"id": 43, "name": "", "email": "", "status": "packed", "pay_status": "paid"},
    ]}
    reg = bl.build_linkables(snap)
    rows = snap["orders"]
    assert reg[rows[0]["ref"]] == {"type": "order", "display": "Jane Doe",
                                   "url": "/console/orders?order=42"}
    assert reg[rows[1]["ref"]]["display"] == "Order #43"   # no name/email fallback
    assert reg[rows[1]["ref"]]["url"] == "/console/orders?order=43"
    assert all(v["type"] == "order" for v in reg.values())


def test_build_linkables_orders_error_block_is_safe():
    snap = {"orders": {"_error": "orders: OperationalError"}}
    assert bl.build_linkables(snap) == {}


def test_build_linkables_mixed_person_invoice_order_share_counter():
    snap = {
        "inbox": {"oldest": [{"from": "Real Client <c@example.com>", "age_days": 4}]},
        "money": {"qbo_ar": [{"id": "5", "doc": "9", "customer": "Beta", "balance": 10, "days_overdue": 1}]},
        "orders": [{"id": 7, "name": "Carol", "status": "new", "pay_status": "unpaid"}],
    }
    reg = bl.build_linkables(snap)
    assert snap["inbox"]["oldest"][0]["ref"] == "r1"       # person first
    assert snap["money"]["qbo_ar"][0]["ref"] == "r2"        # invoice second
    assert snap["orders"][0]["ref"] == "r3"                 # order third
    assert reg["r3"] == {"type": "order", "display": "Carol", "url": "/console/orders?order=7"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_briefing_links.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.briefing_links' has no attribute 'order_url'`.

- [ ] **Step 3a: Add `order_url` + `_iter_order_records`**

In `dashboard/briefing_links.py`, add `order_url` after `invoice_url`:

```python
def order_url(order_id):
    """Canonical console destination for an order: its card on the orders board.
    No console key (appended client-side at click time)."""
    return "/console/orders?order=" + quote(str(order_id or ""), safe="")
```

Add an order iterator next to `_iter_invoice_records`:

```python
def _iter_order_records(snapshot):
    """Yield (record_dict, display, order_id) for each order in the snapshot's
    top-level `orders` block (a list on success, {"_error": ...} on failure ->
    skipped)."""
    orders = snapshot.get("orders")
    rows = orders if isinstance(orders, list) else None
    for rec in (rows or []):
        if isinstance(rec, dict) and rec.get("id") is not None:
            display = rec.get("name") or rec.get("email") or ("Order #" + str(rec.get("id")))
            yield rec, display, rec.get("id")
```

- [ ] **Step 3b: Add the order pass to `build_linkables`**

In `build_linkables`, after the existing invoice pass (the `for rec, display, qid in _iter_invoice_records(snapshot):` loop), add:

```python
    for rec, display, oid in _iter_order_records(snapshot):
        oid = str(oid or "").strip()
        if not oid:
            continue
        mint(rec, "order", display or ("Order #" + oid), order_url(oid))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_briefing_links.py -q`
Expected: PASS (all prior + 4 new).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e0f30eb0
git add dashboard/briefing_links.py tests/test_briefing_links.py
git commit -m "feat: order linkables in briefing_links registry"
```

---

### Task 2: `orders.attention_orders()` — snapshot wrapper

**Files:**
- Modify: `dashboard/orders.py` (add `sqlite3` + `Path` imports; add `_TERMINAL_STATUSES` + `attention_orders`)
- Test: `tests/test_attention_orders.py`

**Interfaces:**
- Consumes: existing `list_orders(cx)` (orders.py:183), `order_backorder_units(cx, id)` (orders.py:317), `init_orders_table` / `init_fulfillments_table`.
- Produces: `attention_orders(limit=20) -> list[dict]` of open orders, each `{id, name, email, status, pay_status, total_cents, created_at, backorder_units}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_attention_orders.py
import sqlite3
from dashboard import orders as o


def _seed(db_path):
    cx = sqlite3.connect(str(db_path))
    o.init_orders_table(cx)
    o.init_fulfillments_table(cx)
    rows = [  # (name, email, status, pay_status, total_cents)
        ("Cart Carol",  "carol@x.com", "new",       "unpaid", 6997),
        ("Paid Pat",    "pat@x.com",   "new",       "paid",   5000),
        ("Packed Peg",  "peg@x.com",   "packed",    "paid",   3000),
        ("Shipped Sam", "sam@x.com",   "shipped",   "paid",   2000),
        ("Done Dan",    "dan@x.com",   "done",      "paid",   1000),
        ("Cancel Cal",  "cal@x.com",   "cancelled", "unpaid", 900),
        ("Prop Pria",   "pria@x.com",  "proposed",  "unpaid", 800),
    ]
    for nm, em, st, ps, tc in rows:
        cx.execute("INSERT INTO orders (name,email,status,pay_status,total_cents,"
                   "items_json,address_json,created_at) VALUES (?,?,?,?,?,'[]','{}',?)",
                   (nm, em, st, ps, tc, "2026-06-26T00:00:00+00:00"))
    cx.commit(); cx.close()


def test_attention_orders_returns_open_only(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed(tmp_path / "chat_log.db")
    res = o.attention_orders()
    statuses = sorted(r["status"] for r in res)
    assert statuses == ["new", "new", "packed", "proposed"]   # excludes shipped/delivered/done/cancelled
    r0 = res[0]
    assert set(r0) == {"id", "name", "email", "status", "pay_status",
                       "total_cents", "created_at", "backorder_units"}
    assert r0["backorder_units"] == 0


def test_attention_orders_respects_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = tmp_path / "chat_log.db"
    cx = sqlite3.connect(str(db)); o.init_orders_table(cx); o.init_fulfillments_table(cx)
    for i in range(30):
        cx.execute("INSERT INTO orders (name,status,pay_status,total_cents,"
                   "items_json,address_json,created_at) VALUES (?,?,?,?,'[]','{}',?)",
                   ("c%d" % i, "new", "unpaid", 100, "2026-06-26T00:00:00+00:00"))
    cx.commit(); cx.close()
    assert len(o.attention_orders(limit=20)) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_attention_orders.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.orders' has no attribute 'attention_orders'`.

- [ ] **Step 3: Implement**

In `dashboard/orders.py`, add to the imports (it already imports `json`, `os`):

```python
import sqlite3
from pathlib import Path
```

Then add near the status constants:

```python
_TERMINAL_STATUSES = ("shipped", "delivered", "done", "cancelled")


def attention_orders(limit=20):
    """Open orders needing attention (status not terminal), newest first, as a
    minimal subset for the briefing snapshot. Self-connects to chat_log.db
    (mirrors briefing_actions._DB). Best-effort: callers wrap in _safe."""
    db = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))) / "chat_log.db"
    cx = sqlite3.connect(str(db), timeout=5)
    try:
        cx.row_factory = sqlite3.Row
        out = []
        for o in list_orders(cx, limit=200):
            if o.get("status") in _TERMINAL_STATUSES:
                continue
            out.append({
                "id": o.get("id"),
                "name": o.get("name") or "",
                "email": o.get("email") or "",
                "status": o.get("status"),
                "pay_status": o.get("pay_status") or "",
                "total_cents": o.get("total_cents") or 0,
                "created_at": o.get("created_at"),
                "backorder_units": order_backorder_units(cx, o.get("id")),
            })
            if len(out) >= limit:
                break
        return out
    finally:
        cx.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_attention_orders.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e0f30eb0
git add dashboard/orders.py tests/test_attention_orders.py
git commit -m "feat: orders.attention_orders snapshot wrapper (open orders)"
```

---

### Task 3: snapshot key + clients-pipeline prompt (`briefing_runner.py`)

**Files:**
- Modify: `dashboard/briefing_runner.py` (import `orders`; add top-level `orders` to snapshot; clients-pipeline prompt; RECORD LINKS example)
- Test: `tests/test_briefing_runner_links.py` (append)

**Interfaces:**
- Consumes: `orders.attention_orders` (Task 2), `briefing_links.build_linkables` (Task 1, stamps refs on `snapshot["orders"]`).
- Produces: `gather_snapshot()["orders"]` present; clients-pipeline prompt instructs order callouts; RECORD LINKS example covers orders.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_snapshot_includes_orders():
    src = (_repo() / "dashboard" / "briefing_runner.py").read_text()
    assert "import orders as _orders" in src
    assert '"orders"' in src
    assert "_orders.attention_orders" in src


def test_clients_prompt_calls_out_orders():
    from dashboard import briefing_runner as br
    p = br.SLUG_PROMPTS["clients-pipeline"]
    assert "orders" in p.lower()


def test_record_links_example_covers_orders():
    from dashboard import briefing_runner as br
    snap = {"orders": [{"id": 7, "name": "Carol", "status": "new", "pay_status": "unpaid"}]}
    from dashboard import briefing_links as bl
    bl.build_linkables(snap)
    prompt = br._build_user_prompt(snap, "clients-pipeline")
    assert "order" in prompt.lower()
    assert "(ref:" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && PINECONE_API_KEY=dummy python3 -m pytest tests/test_briefing_runner_links.py -q`
Expected: FAIL — orders not wired / prompt unchanged.

- [ ] **Step 3a: Import orders + add snapshot key**

In `dashboard/briefing_runner.py`, add to the imports block:

```python
from . import orders as _orders
```

In `gather_snapshot()`, after the `"gohighlevel": ...` / `"inbox": ...` lines (top level, NOT inside `money`), add:

```python
        "orders":      _safe(_orders.attention_orders,    label="orders"),
```

- [ ] **Step 3b: Clients-pipeline prompt — call out orders**

In `SLUG_PROMPTS["clients-pipeline"]`, append to the instruction (before the Insight sentence): a directive to surface orders needing action from the top-level `orders` block — e.g.:

```
"Also surface ORDERS NEEDING ACTION from the snapshot's top-level `orders` "
"block: name them by customer and status, separating unpaid carts to recover "
"(status new with pay_status unpaid, or cart/proposed) from paid-but-unshipped "
"or backordered orders to fulfill. These are funnel/fulfillment orders, NOT the "
"QuickBooks accounts receivable the Finance card owns. "
```

(Insert as an additional sentence in the existing parenthesized f-string; keep all current clients-pipeline rules intact.)

- [ ] **Step 3c: RECORD LINKS example covers orders**

In `_build_user_prompt`, the RECORD LINKS parenthetical (from Phase 1/2) lists examples like "(an inbox sender, an invoice client, or an overdue invoice by customer/amount)". Extend it to also mention orders, e.g. append "or an order to act on (by customer/status)" — so Haiku links order rows. Keep the rest of the rule intact.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && PINECONE_API_KEY=dummy python3 -m pytest tests/test_briefing_runner_links.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e0f30eb0
git add dashboard/briefing_runner.py tests/test_briefing_runner_links.py
git commit -m "feat: add open orders to snapshot; clients-pipeline calls out orders to act on"
```

---

### Task 4: `console-orders.html` — card highlight (`?order=`)

**Files:**
- Modify: `static/console-orders.html` (`data-oid` on cards; `?order=` scroll+flash after `load()`; `ord-flash` CSS)
- Test: `tests/test_console_orders_highlight.py` (source-assert; DOM behavior render-verified in Task 5)

**Interfaces:**
- Consumes: arrival URL `/console/orders?order=<id>` (from `order_url` + `recNavigate`).
- Produces: the card with matching `data-oid` scrolls into view + flashes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_orders_highlight.py
from pathlib import Path


def _html():
    return (Path(__file__).resolve().parent.parent / "static" / "console-orders.html").read_text()


def test_order_card_has_data_oid():
    assert 'data-oid="' in _html()


def test_orders_reads_order_param_and_flashes():
    html = _html()
    assert "URLSearchParams(location.search).get('order')" in html
    assert "scrollIntoView" in html
    assert "ord-flash" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_console_orders_highlight.py -q`
Expected: FAIL.

- [ ] **Step 3a: Add `data-oid` to the card**

In `static/console-orders.html`, `cardHtml` opens the card with `'<div class="card">'` (~line 120). Change to include the id:

```javascript
    return '<div class="card" data-oid="'+o.id+'"><div class="who">'+esc(o.name||o.email||'Order #'+o.id)+badge+'</div>'
```

- [ ] **Step 3b: Highlight after render**

In `load()`, after the lanes render loop (`LANES.forEach(...)`, ends ~line 197) and before the function closes, add a call + helper:

```javascript
    maybeHighlight();
  }

  function maybeHighlight(){
    var id = new URLSearchParams(location.search).get('order');
    if (!id) return;
    var card = document.querySelector('.card[data-oid="' + (window.CSS && CSS.escape ? CSS.escape(id) : id) + '"]');
    if (!card) return;
    card.scrollIntoView({behavior:'smooth', block:'center'});
    card.classList.add('ord-flash');
    setTimeout(function(){ card.classList.remove('ord-flash'); }, 2600);
  }
```

(Place `maybeHighlight` at script scope alongside `load`/`cardHtml`. The `}` shown closes `load()`.)

- [ ] **Step 3c: Add the flash CSS**

In the `<style>` block, add:

```css
@keyframes ord-flash { 0% { background: rgba(212,175,55,0.45); } 100% { background: transparent; } }
.card.ord-flash { animation: ord-flash 2.4s ease-out; }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_console_orders_highlight.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e0f30eb0
git add static/console-orders.html tests/test_console_orders_highlight.py
git commit -m "feat: highlight the targeted order card on /console/orders?order="
```

---

### Task 5: Render-verify + prod go-live gate

Verification only (no code).

- [ ] **Step 1: Full suite**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && PINECONE_API_KEY=dummy python3 -m pytest tests/test_briefing_links.py tests/test_attention_orders.py tests/test_briefing_runner_links.py tests/test_console_orders_highlight.py -q`
Expected: all PASS.

- [ ] **Step 2: Boot locally**

```bash
mkdir -p /tmp/dc3-scratch
cat > /tmp/run-dc3.sh <<'SH'
cd /tmp/wt-deploy-chat-e0f30eb0
exec doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc3-scratch PORT=5079 python3 app.py
SH
bash /tmp/run-dc3.sh
```
(background; needs Doppler + writable DATA_DIR.)

- [ ] **Step 3: Render-verify the card highlight (browser)**

Using claude-in-chrome: navigate to `/console/orders?order=<id>&key=<CONSOLE_SECRET>`. Stub `/api/orders` to return a fake order with id `<id>` (and a couple others), re-run `load()`, and assert: the `.card[data-oid="<id>"]` exists, is scrolled into view, and gains `ord-flash`; a non-matching id is a clean no-op; zero console errors. (Local `/api/orders` may be empty on a scratch DB, so stub it like Phase 2 stubbed `/api/finance/ar`.)

- [ ] **Step 4: Stop + clean up**

```bash
pkill -f "DATA_DIR=/tmp/dc3-scratch" 2>/dev/null; lsof -ti tcp:5079 | xargs kill 2>/dev/null
rm -rf /tmp/dc3-scratch /tmp/run-dc3.sh
```

- [ ] **Step 5: Prod go-live (after merge + deploy)**

Trigger `POST /api/regenerate-briefings` (authed) and confirm the clients-pipeline registry contains `type:"order"` entries with `/console/orders?order=…` urls (when open orders exist). Under-linking → prompt tweak, not code.

---

## Notes for the implementer

- **No `dashboard.html`/`app.py` change.** Order links resolve via the existing resolver; `recNavigate` already produces `/console/orders?order=123&key=…`.
- **Run pytest from the worktree.** `briefing_links`/`orders`/console source-asserts are pure; the `briefing_runner` import needs `PINECONE_API_KEY=dummy`.
- **`snapshot["orders"]` shape:** list on success, `{"_error": …}` on failure — `_iter_order_records` guards with `isinstance(..., list)`.
- **Terminal statuses excluded:** `shipped, delivered, done, cancelled` (cancelled included — no action on a cancelled order).
