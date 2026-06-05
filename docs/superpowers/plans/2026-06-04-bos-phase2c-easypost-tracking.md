# BOS Phase 2c: EasyPost Labels + Customer Tracking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the two fulfillment actions that close the loop from the Orders board: `orders.create_label` (buy a USPS label via EasyPost when `EASYPOST_API_KEY` is set, otherwise a Click-N-Ship handoff) and `orders.send_tracking` (email the customer their tracking number, reusing the existing tracking-email machinery, and record the shipment).

**Architecture:** A new `dashboard/easypost.py` isolates the carrier integration behind a feature flag: pure helpers (`is_configured`, payload builder, handoff URL) are unit-tested; the live label purchase runs only when the key is present (production-only, since no key exists yet). The actions live in `dashboard/orders.py`. `orders.send_tracking` reuses `dashboard/tracking.build_tracking_email` + `record_shipment` and best-effort sends via `dashboard.inbox.send_email`. Board buttons call these through the audited `/api/action/<key>` path.

**Builds on:** Phase 2a/2b (orders model + board) + 1a spine. Same branch `sess/ec0e1f15`, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Honest verification note:** there is no `EASYPOST_API_KEY` and no Gmail token in the local/doppler-with-temp-DATA_DIR environment, so the live label purchase and the live email send are PRODUCTION-only. This plan fully tests the flag-off (handoff) path, the payload builder, the shipment recording, and the action registration; the live carrier/email calls are best-effort and logged.

---

## File Structure

- `dashboard/easypost.py` (new): `is_configured`, `build_shipment`, `CLICKNSHIP_URL`, `buy_label` (gated). One responsibility: the carrier integration boundary.
- `dashboard/orders.py` (modify): add `label_url` column + `set_order_label`; add `orders.create_label` and `orders.send_tracking` actions.
- `tests/test_bos_easypost.py` (new): easypost helpers + the create_label handoff path + send_tracking recording.
- `app.py` (modify): import `dashboard.easypost`; ensure tracking schema init; verify under doppler.
- `static/console-orders.html` (modify): add Create-label + Send-tracking buttons.

---

## Task 1: EasyPost boundary (`dashboard/easypost.py`)

**Files:**
- Create: `dashboard/easypost.py`
- Test: `tests/test_bos_easypost.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bos_easypost.py`:

```python
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_is_configured_reads_env(monkeypatch):
    from dashboard import easypost as EP
    monkeypatch.delenv("EASYPOST_API_KEY", raising=False)
    assert EP.is_configured() is False
    monkeypatch.setenv("EASYPOST_API_KEY", "ezk_test")
    assert EP.is_configured() is True


def test_build_shipment_shape():
    from dashboard import easypost as EP
    order = {"name": "Ann Buyer", "address": {"street": "1 Main St", "city": "Hilo",
             "state": "HI", "zip": "96720"}, "items": [{"qty": 2}]}
    s = EP.build_shipment(order, from_address={"name": "Remedy Match", "street": "x",
                          "city": "Hilo", "state": "HI", "zip": "96720"})
    assert s["to_address"]["name"] == "Ann Buyer"
    assert s["to_address"]["zip"] == "96720"
    assert s["from_address"]["zip"] == "96720"
    assert s["parcel"]["weight"] > 0  # ounces, derived from item count


def test_clicknship_url_constant():
    from dashboard import easypost as EP
    assert EP.CLICKNSHIP_URL.startswith("https://")
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_easypost.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'dashboard.easypost'`).

- [ ] **Step 3: Write the implementation**

Create `dashboard/easypost.py`:

```python
"""EasyPost carrier boundary, behind the EASYPOST_API_KEY feature flag.

When the key is absent (current state), the Orders module falls back to a manual
USPS Click-N-Ship handoff. When set, buy_label() purchases the lowest USPS rate
and returns the label URL + tracking number. The pure helpers are unit-tested;
the live API call is production-only."""
import json
import os
import urllib.request

CLICKNSHIP_URL = "https://cns.usps.com"
_API = "https://api.easypost.com/v2"
_DEFAULT_OZ = 4  # base weight per parcel
_PER_ITEM_OZ = 4  # rough per-bottle weight


def is_configured():
    return bool(os.environ.get("EASYPOST_API_KEY"))


def build_shipment(order, from_address):
    """Pure: build the EasyPost shipment payload from an order dict + the ship-from
    address. Weight is estimated from item count (ounces)."""
    addr = order.get("address") or {}
    n_items = sum(int(i.get("qty", 1) or 1) for i in (order.get("items") or [])) or 1
    return {
        "to_address": {
            "name": order.get("name") or order.get("email") or "Customer",
            "street1": addr.get("street", ""), "city": addr.get("city", ""),
            "state": addr.get("state", ""), "zip": addr.get("zip", ""),
            "country": addr.get("country", "US"),
        },
        "from_address": dict(from_address or {}, **{"street1": (from_address or {}).get("street", "")}),
        "parcel": {"weight": _DEFAULT_OZ + _PER_ITEM_OZ * n_items},
    }


def buy_label(order, from_address):
    """Live: create a shipment, buy the lowest rate. Production-only (requires
    EASYPOST_API_KEY). Returns {tracking_number, label_url} or raises."""
    key = os.environ.get("EASYPOST_API_KEY")
    if not key:
        raise RuntimeError("EASYPOST_API_KEY not set")
    import base64
    auth = base64.b64encode((key + ":").encode()).decode()
    payload = json.dumps({"shipment": build_shipment(order, from_address)}).encode()
    req = urllib.request.Request(_API + "/shipments", data=payload, method="POST",
                                 headers={"Authorization": "Basic " + auth,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        shp = json.loads(r.read())
    rates = shp.get("rates") or []
    if not rates:
        raise RuntimeError("no rates returned")
    lowest = min(rates, key=lambda x: float(x.get("rate", "9999")))
    buy = json.dumps({"rate": {"id": lowest["id"]}}).encode()
    breq = urllib.request.Request(_API + "/shipments/" + shp["id"] + "/buy", data=buy,
                                  method="POST",
                                  headers={"Authorization": "Basic " + auth,
                                           "Content-Type": "application/json"})
    with urllib.request.urlopen(breq, timeout=30) as r:
        bought = json.loads(r.read())
    return {"tracking_number": bought.get("tracking_code", ""),
            "label_url": (bought.get("postage_label") or {}).get("label_url", "")}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_easypost.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/easypost.py tests/test_bos_easypost.py
git commit -m "feat(bos): easypost carrier boundary (flag-gated label purchase)"
```

---

## Task 2: create_label action + label_url column (`dashboard/orders.py`)

**Files:**
- Modify: `dashboard/orders.py`
- Test: `tests/test_bos_easypost.py` (append)

- [ ] **Step 1: Write the failing test** (append)

```python
def test_create_label_handoff_when_unconfigured(monkeypatch):
    import sqlite3
    monkeypatch.delenv("EASYPOST_API_KEY", raising=False)
    from dashboard import orders as O, dispatch as D, events as E, rbac as R, actions as A
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx); O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="LBL-1", name="Ann")
    assert A.get_action("orders.create_label") is not None
    res = D.dispatch_action(cx, "orders.create_label", {"order_id": oid}, R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    msg = (res["result"] or {}).get("message", "")
    assert "click-n-ship" in msg.lower() or "cns.usps" in (res["result"] or {}).get("handoff", "").lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_easypost.py -k create_label -q`
Expected: FAIL (`orders.create_label` not registered -> None).

- [ ] **Step 3: Append to `dashboard/orders.py`**

First, add `label_url` to the schema. In `init_orders_table`, after the `CREATE TABLE` + index, add a best-effort migration:

```python
    try:
        cx.execute("ALTER TABLE orders ADD COLUMN label_url TEXT")
    except Exception:
        pass  # already present
    cx.commit()
```

Add a setter near `set_order_tracking`:

```python
def set_order_label(cx, order_id, label_url, tracking_number=None):
    if tracking_number:
        cx.execute("UPDATE orders SET label_url=?, tracking_number=?, updated_at=? WHERE id=?",
                   (label_url, tracking_number, _now(), order_id))
    else:
        cx.execute("UPDATE orders SET label_url=?, updated_at=? WHERE id=?",
                   (label_url, _now(), order_id))
    cx.commit()
    return cx.total_changes >= 0
```

Add the action (next to the other lifecycle actions):

```python
def _create_label_exec(params, ctx):
    from dashboard import easypost as EP
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    oid = int(params["order_id"])
    order = get_order(cx, oid)
    if not order:
        raise ValueError(f"order #{oid} not found")
    if not EP.is_configured():
        return {"order_id": oid, "handoff": EP.CLICKNSHIP_URL,
                "message": "No label API configured. Buy the label on USPS Click-N-Ship, "
                           "then use Ship + tracking to record the number."}
    from_addr = (ctx or {}).get("from_address") or {}
    out = EP.buy_label(order, from_addr)
    set_order_label(cx, oid, out.get("label_url", ""), out.get("tracking_number"))
    return {"order_id": oid, "tracking_number": out.get("tracking_number", ""),
            "label_url": out.get("label_url", ""),
            "message": f"Label bought for order #{oid} (tracking {out.get('tracking_number','')})."}


action(key="orders.create_label", module="orders", title="Create shipping label",
       description="Buy a USPS label (EasyPost) or hand off to Click-N-Ship.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_create_label_exec)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_easypost.py -q`
Expected: 4 passed.

Run: `python3 -m pytest tests/test_bos_orders.py -q` (still 7 passed; the new label_url column is additive).

- [ ] **Step 5: Commit**

```bash
git add dashboard/orders.py tests/test_bos_easypost.py
git commit -m "feat(bos): orders.create_label (easypost or click-n-ship handoff) + label_url"
```

---

## Task 3: send_tracking action (`dashboard/orders.py`)

**Files:**
- Modify: `dashboard/orders.py`
- Test: `tests/test_bos_easypost.py` (append)

- [ ] **Step 1: Write the failing test** (append)

```python
def test_send_tracking_records_shipment(monkeypatch):
    import sqlite3
    from dashboard import orders as O, dispatch as D, events as E, rbac as R, actions as A
    from dashboard import tracking as T
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx); O.init_orders_table(cx); T.init_tracking_schema(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="TR-1", name="Ann",
                         email="ann@x.com")
    O.set_order_tracking(cx, oid, "9400111899")
    # stub the gmail send so the test never hits the network
    import dashboard.orders as OM
    monkeypatch.setattr(OM, "_gmail_send_tracking", lambda to, subj, html: True, raising=False)
    res = D.dispatch_action(cx, "orders.send_tracking", {"order_id": oid}, R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    sh = cx.execute("SELECT status, resolved_email FROM shipments WHERE tracking_number='9400111899'").fetchone()
    assert sh is not None and sh["resolved_email"] == "ann@x.com"


def test_send_tracking_requires_tracking_number():
    import sqlite3
    from dashboard import orders as O, dispatch as D, events as E, rbac as R
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx); O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="TR-2", email="b@x.com")
    res = D.dispatch_action(cx, "orders.send_tracking", {"order_id": oid}, R.Actor(role=R.OWNER))
    assert res["status"] == "failed"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_easypost.py -k send_tracking -q`
Expected: FAIL (`orders.send_tracking` not registered).

- [ ] **Step 3: Append to `dashboard/orders.py`**

```python
def _gmail_send_tracking(to, subject, html):
    """Best-effort customer email via the inbox Gmail sender. Production-only;
    returns False (not an error) when Gmail is unavailable."""
    try:
        from dashboard.inbox import send_email
        send_email(to, subject, html, from_name="Dr. Glen Swartwout")
        return True
    except Exception as e:
        print(f"[orders] tracking email skipped: {e!r}", flush=True)
        return False


def _send_tracking_exec(params, ctx):
    from dashboard import tracking as T
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    oid = int(params["order_id"])
    order = get_order(cx, oid)
    if not order:
        raise ValueError(f"order #{oid} not found")
    tn = order.get("tracking_number") or ""
    if not tn:
        raise ValueError("order has no tracking number yet (create a label or set tracking first)")
    email = order.get("email") or ""
    em = T.build_tracking_email(tn, order.get("name"))
    sent = _gmail_send_tracking(email, em.get("subject", "tracking number"), em.get("html", "")) if email else False
    try:
        T.init_tracking_schema(cx)
        T.record_shipment(cx, tracking_number=tn, recipient_name=order.get("name"),
                          resolved_email=email, status=("sent" if sent else "drafted"),
                          order_uuid=order.get("external_ref"))
        sh = cx.execute("SELECT id FROM shipments WHERE tracking_number=?", (tn,)).fetchone()
        if sh:
            set_order_tracking(cx, oid, tn, shipment_id=sh[0])
    except Exception as e:
        print(f"[orders] shipment record: {e!r}", flush=True)
    verb = "sent to" if sent else "drafted for"
    return {"order_id": oid, "tracking_number": tn, "emailed": sent,
            "message": f"Tracking {tn} {verb} {email or 'customer'}."}


action(key="orders.send_tracking", module="orders", title="Send tracking email",
       description="Email the customer their tracking number and record the shipment.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_send_tracking_exec)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_easypost.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/orders.py tests/test_bos_easypost.py
git commit -m "feat(bos): orders.send_tracking (customer email + shipment record)"
```

---

## Task 4: Wire into app.py + board (verified under doppler)

**Files:**
- Modify: `app.py`, `static/console-orders.html`

- [ ] **Step 1: Import the easypost module in the BOS startup block** (so it's loaded; the actions register via `dashboard.orders` which is already imported). Add near `import dashboard.orders as _bos_orders`:

```python
import dashboard.easypost as _bos_easypost  # noqa: F401
```

- [ ] **Step 2: Add board buttons** in `static/console-orders.html` `cardHtml`. For `packed` orders add a "Create label" button; for `shipped` orders add "Send tracking". Update the action block:

```javascript
    if (o.status==='new')     acts = btn(o.id,'orders.mark_packed','Pack');
    else if (o.status==='packed') acts = btn(o.id,'orders.create_label','Create label') + ' ' + shipBtn(o.id);
    else if (o.status==='shipped') acts = btn(o.id,'orders.send_tracking','Send tracking') + ' ' + btn(o.id,'orders.mark_done','Delivered');
```

- [ ] **Step 3: Compile + verify under doppler**

Run: `python3 -m py_compile app.py` (OK).
Run:
```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app, sqlite3
from dashboard import actions as A, orders as O
for k in ("orders.create_label","orders.send_tracking"):
    assert A.get_action(k) is not None, "missing "+k
# label_url column exists
cx = sqlite3.connect(app.LOG_DB); cx.row_factory=sqlite3.Row
cols = [r[1] for r in cx.execute("PRAGMA table_info(orders)")]
assert "label_url" in cols, "label_url column missing"
# create_label handoff path (no EASYPOST_API_KEY in this env)
oid = O.upsert_order(cx, source="manual", external_ref="C-1", name="Z")
from dashboard import dispatch as D, rbac as R
res = D.dispatch_action(cx, "orders.create_label", {"order_id": oid}, R.Actor(role=R.OWNER))
assert res["status"]=="done" and (res["result"].get("handoff") or "").startswith("https://cns")
print("ORDERS_2C_OK", res["result"]["message"][:40])
PY'
rm -rf /tmp/bostest
```
Expected: `ORDERS_2C_OK ...` no assertion error.

Run: `python3 -m pytest tests/test_bos_easypost.py tests/test_bos_orders.py tests/test_bos_spine.py -q` (green).

- [ ] **Step 4: Commit**

```bash
git add app.py static/console-orders.html
git commit -m "feat(bos): wire easypost + add create-label/send-tracking board buttons"
```

---

## Self-Review

**Spec coverage** (the label/tracking half of blueprint 5.4):
- `orders.create_label`: EasyPost when keyed, Click-N-Ship handoff when not -> Task 1 + 2.
- `orders.send_tracking`: customer email + shipment record (sets the `shipments.sent`/`drafted` status that was previously a dead end) -> Task 3.
- Board buttons for both -> Task 4.
- Flag-gated on `EASYPOST_API_KEY` (switches on when Glen adds the key) -> Task 1 `is_configured`.

**Production-only paths (cannot run in local/doppler-temp env):** the live EasyPost purchase (no key) and the live Gmail send (no token on a temp DATA_DIR). Both are best-effort + logged; the flag-off handoff path, the payload builder, and the shipment recording are fully tested.

**Out of scope (later):** rate shopping UI, auto-send toggle, multi-parcel, return labels, the ship-from address config UI (passed via ctx/from_address, defaults empty until configured).

**Placeholder scan:** none.

**Type consistency:** `is_configured`/`build_shipment`/`buy_label`, the order dict keys, `set_order_label`, the action keys (`orders.create_label`, `orders.send_tracking`), and the dispatch result `status` values are consistent across Tasks 1-4 and Phase 2a/2b.
