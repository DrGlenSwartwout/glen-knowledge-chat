# BOS Phase 2a: Orders Model + Ingestion + Lifecycle + Signal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Create the missing order object: one `orders` table that unifies every order flow (funnel/QBO retail+wholesale, GrooveKart, dispensary, manual) into a single audited lifecycle (new -> packed -> shipped -> done / cancelled), with the lifecycle actions on the dispatch spine and a real Orders signal on the Home board.

**Architecture:** A new `dashboard/orders.py` (pure, takes a sqlite connection, like `begin_funnel.py`) holds the model + idempotent upsert + the lifecycle registry actions + the Orders `signal()`. `app.py` ingests orders at each existing order-creation site (best-effort, never breaks checkout) and registers the module at startup. The board UI (2b) and EasyPost label automation (2c) are separate follow-on plans.

**Builds on:** Phase 1a/1b/1c. Same branch `sess/ec0e1f15`, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Decision (confirmed):** the new funnel (QBO checkout) is the main order source for retail + wholesale; the board also integrates GrooveKart, dispensary, and manual orders. Label automation targets EasyPost, flag-gated, in 2c.

---

## File Structure

- `dashboard/orders.py` (new): orders table + `upsert_order`/`get_order`/`list_orders`/`set_order_status`/`set_order_tracking`, the four lifecycle actions, and the Orders `signal()`. One responsibility: the order model + its lifecycle.
- `tests/test_bos_orders.py` (new): unit tests (in-memory cx).
- `app.py` (modify): register the orders module at startup (init table + action/signal registration), ingest at the funnel buy / GrooveKart webhook / wholesale checkout / dispensary sites, and add a `POST /api/orders` manual-save endpoint.

---

## Task 1: Orders model (`dashboard/orders.py` data layer)

**Files:**
- Create: `dashboard/orders.py`
- Test: `tests/test_bos_orders.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bos_orders.py`:

```python
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import orders as O
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    return O, cx


def test_upsert_is_idempotent_and_preserves_status():
    O, cx = _db()
    oid = O.upsert_order(cx, source="funnel", external_ref="INV-1", email="a@b.com",
                         name="Ann", items=[{"name": "Mag", "qty": 2}], total_cents=7000,
                         channel="retail")
    assert oid > 0
    # advance status, then re-ingest the same order: status must NOT regress
    assert O.set_order_status(cx, oid, "packed") is True
    again = O.upsert_order(cx, source="funnel", external_ref="INV-1", email="a@b.com",
                           name="Ann Updated", total_cents=7000)
    assert again == oid
    row = O.get_order(cx, oid)
    assert row["status"] == "packed"
    assert row["name"] == "Ann Updated"
    assert row["items"] == [{"name": "Mag", "qty": 2}]


def test_upsert_requires_external_ref():
    O, cx = _db()
    try:
        O.upsert_order(cx, source="manual", external_ref="")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_list_orders_filter_and_set_tracking():
    O, cx = _db()
    a = O.upsert_order(cx, source="gk", external_ref="1")
    b = O.upsert_order(cx, source="gk", external_ref="2")
    O.set_order_status(cx, b, "shipped")
    assert len(O.list_orders(cx)) == 2
    assert len(O.list_orders(cx, status="new")) == 1
    O.set_order_tracking(cx, a, "9400111899", shipment_id=5)
    assert O.get_order(cx, a)["tracking_number"] == "9400111899"
    assert O.get_order(cx, a)["shipment_id"] == 5
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_orders.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.orders'`.

- [ ] **Step 3: Write the data layer**

Create `dashboard/orders.py`:

```python
"""Business-OS Orders & Fulfillment model. One orders table unifies every order
flow (funnel/QBO retail+wholesale, GrooveKart, dispensary, manual) into a single
lifecycle: new -> packed -> shipped -> done (+ cancelled). Functions take a
sqlite connection for testability. Lifecycle actions + the Home signal register
on import (see Task 2)."""
import json
from datetime import datetime, timezone, timedelta

ORDER_STATUSES = ("new", "packed", "shipped", "done", "cancelled")
_OPEN = ("new", "packed")  # unfulfilled


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_orders_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            external_ref TEXT NOT NULL,
            channel TEXT DEFAULT 'retail',
            email TEXT, name TEXT, phone TEXT,
            items_json TEXT, total_cents INTEGER DEFAULT 0,
            address_json TEXT,
            status TEXT NOT NULL DEFAULT 'new',
            tracking_number TEXT, shipment_id INTEGER,
            notes TEXT, updated_at TEXT,
            UNIQUE(source, external_ref)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
    cx.commit()


def upsert_order(cx, *, source, external_ref, email="", name="", phone="",
                 items=None, total_cents=0, address=None, channel="retail",
                 status="new"):
    """Idempotent on (source, external_ref). Inserts a new order, or updates the
    soft fields of an existing one WITHOUT regressing its lifecycle status."""
    ref = str(external_ref or "").strip()
    if not ref:
        raise ValueError("external_ref required")
    row = cx.execute("SELECT id FROM orders WHERE source=? AND external_ref=?",
                     (source, ref)).fetchone()
    if row:
        cx.execute("UPDATE orders SET email=?, name=?, phone=?, items_json=?, "
                   "total_cents=?, address_json=?, channel=?, updated_at=? WHERE id=?",
                   (email, name, phone, json.dumps(items or []), int(total_cents or 0),
                    json.dumps(address or {}), channel, _now(), row[0]))
        cx.commit()
        return row[0]
    cur = cx.execute(
        "INSERT INTO orders (created_at, source, external_ref, channel, email, name, "
        "phone, items_json, total_cents, address_json, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (_now(), source, ref, channel, email, name, phone,
         json.dumps(items or []), int(total_cents or 0), json.dumps(address or {}), status))
    cx.commit()
    return cur.lastrowid


def _row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    d["items"] = json.loads(d.pop("items_json") or "[]")
    d["address"] = json.loads(d.pop("address_json") or "{}")
    return d


def get_order(cx, order_id):
    cur = cx.execute("SELECT * FROM orders WHERE id=?", (order_id,))
    return _row_to_dict(cur.fetchone())


def list_orders(cx, *, status=None, limit=200):
    if status:
        cur = cx.execute("SELECT * FROM orders WHERE status=? ORDER BY id DESC LIMIT ?",
                         (status, limit))
    else:
        cur = cx.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
    return [_row_to_dict(r) for r in cur.fetchall()]


def set_order_status(cx, order_id, status):
    if status not in ORDER_STATUSES:
        raise ValueError(f"unknown status: {status}")
    cur = cx.execute("UPDATE orders SET status=?, updated_at=? WHERE id=?",
                     (status, _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def set_order_tracking(cx, order_id, tracking_number, shipment_id=None):
    cur = cx.execute("UPDATE orders SET tracking_number=?, shipment_id=?, updated_at=? WHERE id=?",
                     (tracking_number, shipment_id, _now(), order_id))
    cx.commit()
    return cur.rowcount > 0
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_orders.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/orders.py tests/test_bos_orders.py
git commit -m "feat(bos): orders model + idempotent upsert + lifecycle setters"
```

---

## Task 2: Lifecycle actions + Orders signal (`dashboard/orders.py`)

**Files:**
- Modify: `dashboard/orders.py` (append)
- Test: `tests/test_bos_orders.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_orders_signal_levels():
    from dashboard import orders as O, signals as S
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    now = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)
    # no orders -> green
    assert O.orders_signal(cx, None, now=now)["level"] == S.GREEN
    # a fresh open order -> amber
    O.upsert_order(cx, source="funnel", external_ref="A")
    sig = O.orders_signal(cx, None, now=now)
    assert sig["level"] == S.AMBER and sig["count"] == 1
    # an order created >24h ago -> red
    cx.execute("UPDATE orders SET created_at=? WHERE external_ref='A'",
               ((now - timedelta(hours=48)).isoformat(),))
    cx.commit()
    assert O.orders_signal(cx, None, now=now)["level"] == S.RED


def test_orders_signal_gray_when_table_missing():
    from dashboard import orders as O, signals as S
    cx = sqlite3.connect(":memory:")  # no orders table
    assert O.orders_signal(cx, None)["level"] == S.GRAY


def test_lifecycle_action_marks_packed_and_logs_event():
    from dashboard import orders as O, dispatch as D, events as E, rbac as R, actions as A
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="Z")
    assert A.get_action("orders.mark_packed") is not None
    res = D.dispatch_action(cx, "orders.mark_packed", {"order_id": oid}, R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    assert O.get_order(cx, oid)["status"] == "packed"
    ev = E.list_events(cx, module="orders")
    assert ev and ev[0]["action_key"] == "orders.mark_packed"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_orders.py -k "signal or lifecycle" -q`
Expected: FAIL (`AttributeError: module 'dashboard.orders' has no attribute 'orders_signal'`).

- [ ] **Step 3: Append the actions + signal to `dashboard/orders.py`**

```python
# --- Lifecycle actions + Home signal (register on import) ---
from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS, VA
from dashboard.signals import signal as _signal, RED, AMBER, GREEN, GRAY


def _status_action(new_status, verb):
    def _exec(params, ctx):
        cx = (ctx or {}).get("cx") or (params or {}).get("cx")
        if cx is None:
            raise ValueError("no db connection")
        oid = int(params["order_id"])
        if not set_order_status(cx, oid, new_status):
            raise ValueError(f"order #{oid} not found")
        return {"order_id": oid, "status": new_status, "message": f"Order #{oid} {verb}."}
    return _exec


action(key="orders.mark_packed", module="orders", title="Mark packed",
       description="Mark an order packed.", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_status_action("packed", "marked packed"))
action(key="orders.mark_shipped", module="orders", title="Mark shipped",
       description="Mark an order shipped.", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_status_action("shipped", "marked shipped"))
action(key="orders.mark_done", module="orders", title="Mark done",
       description="Mark an order complete/delivered.", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_status_action("done", "marked done"))
action(key="orders.cancel", module="orders", title="Cancel order",
       description="Cancel an order.", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_status_action("cancelled", "cancelled"))


@_signal("orders")
def orders_signal(cx, actor=None, now=None):
    try:
        rows = cx.execute(
            "SELECT created_at FROM orders WHERE status IN ('new','packed')").fetchall()
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    n = len(rows)
    if n == 0:
        return {"level": GREEN, "summary": "No orders to fulfill", "top_actions": [], "count": 0}
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=24)
    aging = 0
    for r in rows:
        try:
            ts = datetime.fromisoformat(r[0])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                aging += 1
        except Exception:
            pass
    level = RED if aging else AMBER
    summary = f"{n} to fulfill" + (f", {aging} aging over 24h" if aging else "")
    return {"level": level, "summary": summary,
            "top_actions": [{"label": "Open orders board", "href": "/console/orders"}],
            "count": n}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_orders.py -q`
Expected: 6 passed.

Then confirm no regression in the 1b signals test (the defensive gray-on-missing-table keeps the orders cell valid there):
Run: `python3 -m pytest tests/test_bos_signals.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/orders.py tests/test_bos_orders.py
git commit -m "feat(bos): order lifecycle actions + Orders home signal"
```

---

## Task 3: Ingestion + startup wiring in `app.py` (verified under doppler)

**Files:**
- Modify: `app.py`

This touches Pinecone-importing code; verify with the doppler commands in Step 5.

- [ ] **Step 1: Register the orders module at startup**

In the BOS startup block (search `import dashboard.signals as _bos_signals`), add beneath it:

```python
import dashboard.orders as _bos_orders  # noqa: F401 (registers order actions + signal)
```

And in `_init_bos_events()` (or right after it), initialize the orders table. Add after the `_init_bos_events()` call:

```python
def _init_bos_orders():
    cx = _sqlite3.connect(LOG_DB)
    try:
        _bos_orders.init_orders_table(cx)
    finally:
        cx.close()


_init_bos_orders()
```

- [ ] **Step 2: Add a best-effort ingestion helper + the funnel hook**

Near the orders init above, add:

```python
def _ingest_order(*, source, external_ref, email="", name="", phone="",
                  items=None, total_cents=0, address=None, channel="retail"):
    """Best-effort: record an order into the BOS orders table. Never raises into
    a checkout path."""
    try:
        cx = _sqlite3.connect(LOG_DB)
        try:
            _bos_orders.upsert_order(
                cx, source=source, external_ref=external_ref, email=email, name=name,
                phone=phone, items=items or [], total_cents=int(total_cents or 0),
                address=address or {}, channel=channel)
        finally:
            cx.close()
    except Exception as e:
        print(f"[orders] ingest {source}/{external_ref}: {e!r}", flush=True)
```

In the funnel buy route, right after the `out = {...}` dict is built (just before the dispensary hook, around app.py:2464), insert:

```python
        _ingest_order(source="funnel", external_ref=inv.get("Id"), email=email, name=name,
                      items=[{"name": p["name"], "qty": qty, "desc": desc}],
                      total_cents=int(round(float(inv.get("TotalAmt") or 0) * 100)),
                      channel="retail")
```

- [ ] **Step 3: Add the other ingestion hooks**

GrooveKart webhook (`def groovekart_webhook`, ~app.py:6128): just before its `return`, add (use the parsed `email`, `first`, `last`, `product`, `order_total` already in scope):

```python
    _ingest_order(source="groovekart", external_ref=str(data.get("id") or data.get("order_id") or email or _bos_orders._now()),
                  email=email, name=(first + " " + last).strip(),
                  items=[{"name": product}] if product else [],
                  total_cents=int(round(float(order_total or 0) * 100)), channel="retail")
```

Wholesale checkout (`def api_practitioner_checkout`, ~app.py:5103): right after `out = _wc.build_order(...)` succeeds and `out` contains the invoice id, add:

```python
        _ingest_order(source="wholesale", external_ref=str(out.get("invoice_id") or out.get("Id") or ""),
                      email=(prac.get("email") if isinstance(prac, dict) else "") or "",
                      name=(prac.get("name") if isinstance(prac, dict) else "") or "",
                      total_cents=int(out.get("total_cents") or 0), channel="wholesale")
```
(If `out` has different key names, adapt to the actual `build_order` return; inspect `dashboard/wholesale_checkout.py` `build_order` return shape. The external_ref MUST be the QBO invoice id; if absent, skip the call rather than insert a blank ref.)

Dispensary (`def _record_dispensary_sale`, ~app.py:5275): after `record_dispensary_order(...)`, add:

```python
    _ingest_order(source="dispensary", external_ref=str(invoice_id or ""),
                  email=customer_email or "", items=[{"name": "Dispensary", "qty": bottles}],
                  channel="retail")
```
(Guard: only call when `invoice_id` is truthy.)

- [ ] **Step 4: Add a manual-save endpoint** (near the other `/api/orders`-style routes / the bos_* routes)

```python
@app.route("/api/orders", methods=["POST"])
def bos_orders_create():
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    b = request.get_json(silent=True) or {}
    ref = str(b.get("external_ref") or f"manual-{_bos_orders._now()}")
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        oid = _bos_orders.upsert_order(
            cx, source="manual", external_ref=ref, email=b.get("email", ""),
            name=b.get("name", ""), phone=b.get("phone", ""), items=b.get("items") or [],
            total_cents=int(b.get("total_cents") or 0), address=b.get("address") or {},
            channel=b.get("channel", "retail"))
    finally:
        cx.close()
    return jsonify({"ok": True, "order_id": oid})
```

- [ ] **Step 5: Compile + verify under doppler**

Run: `python3 -m py_compile app.py` (OK).

Run the verification (app imports Pinecone, so real creds + local DATA_DIR):
```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app, sqlite3
from dashboard import orders as O, actions as A, signals as S
# table exists + actions + signal registered
cx = sqlite3.connect(app.LOG_DB); cx.row_factory = sqlite3.Row
O.upsert_order(cx, source="funnel", external_ref="VERIFY-1", email="x@y.com", total_cents=7000)
assert O.list_orders(cx, status="new"), "order not stored"
for k in ("orders.mark_packed","orders.mark_shipped","orders.mark_done","orders.cancel"):
    assert A.get_action(k) is not None, "missing "+k
cells = {c["module"]: c for c in S.aggregate_signals(cx, None)}
assert cells["orders"]["level"] in (S.AMBER, S.RED), "orders signal not live: "+str(cells["orders"])
print("ORDERS_2A_OK level=", cells["orders"]["level"], "count=", cells["orders"]["count"])
PY'
rm -rf /tmp/bostest
```
Expected: prints `ORDERS_2A_OK level= amber count= 1` (or higher) with no assertion error.

Run: `python3 -m pytest tests/test_bos_orders.py tests/test_bos_signals.py tests/test_bos_spine.py -q` (all green).

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(bos): ingest orders from funnel/groovekart/wholesale/dispensary + manual save"
```

---

## Self-Review

**Spec coverage** (blueprint section 5.4, the buildable-now part):
- One unified `orders` table across all flows -> Task 1 + Task 3 ingestion.
- Lifecycle new/packed/shipped/done/cancelled through the audited dispatch path -> Task 2 actions.
- Orders signal lights up the Home cell -> Task 2 `orders_signal` (defensive gray when the table is absent, so the 1b signals test still passes).
- Funnel/QBO as the main source + GrooveKart + wholesale + dispensary + manual -> Task 3.

**Out of scope (follow-on plans):** the Orders board UI (2b); `orders.send_tracking` + linking the `shipments` table + setting its `sent` status (2b/2c); EasyPost label buying + auto-tracking, flag-gated on `EASYPOST_API_KEY` (2c); pulling per-order line items from QBO/`inbound_leads.raw_json` enrichment.

**Placeholder scan:** none. Task 3 notes where to adapt to the real `build_order` return shape and to guard blank refs.

**Type consistency:** `upsert_order(... external_ref ...)`, the order dict keys (`status`, `items`, `address`, `tracking_number`, `shipment_id`), `ORDER_STATUSES`, the signal cell shape, and the action keys (`orders.mark_packed/mark_shipped/mark_done/cancel`) are consistent across Tasks 1-3 and the Phase 1a dispatch + 1b signal contracts.
