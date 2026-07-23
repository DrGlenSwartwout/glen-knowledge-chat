# Caregiver Payer Attribution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a caregiver pay another *adult* household member's order and have it book as the caregiver's own payment/expense, while the order, remedies, and clinical record stay the member's.

**Architecture:** Reuse the shipped directional household link (`household_members`) for consent — a member-controlled `pay_consent` flag (default OFF) + `pay_share_scope`. Add a nullable `payer_email` to the `order_payments` ledger; that is the *only* place the money re-homes. Attribution surfaces read `COALESCE(payer_email, orders.email)`. The caregiver sees a dedicated thin "orders I'm paying for" block — never the member's email-keyed portal view — so clinical data can't leak.

**Tech Stack:** Python (Flask), SQLite in dev / Postgres in prod via `dashboard/db.py` + `dashboard/dbwrite.insert_returning_id`, Stripe Checkout (`dashboard/stripe_pay.py`), pytest.

## Global Constraints

- **Feature flag:** everything ships behind `CAREGIVER_PAY_ENABLED` (default OFF). Mirror the existing `_household_sharing_enabled()` helper. Flags are read at startup → a flip needs a Render restart.
- **Postgres-portable writes:** never use `cur.lastrowid`. New-row inserts go through `dashboard.dbwrite.insert_returning_id` (already used by `order_payments._insert`).
- **Backward-compatible by construction:** `payer_email` is nullable; `NULL` means "payer = order owner." No backfill. Every existing row/query behaves identically.
- **Security anchor:** only the **member** sets `pay_consent`, from their own token. A caregiver can never self-authorize. Reject any authorization where `payer == member`.
- **Clinical firewall:** the caregiver's payable-orders surface is a *separate* query returning only `{beneficiary_name, order_id, amount, status, optional line_items}`. It must never call `get_portal_view` for the member.
- **Test hygiene:** tests must not send live email (stub any send). Pin the catalog if a test imports product lookups (`$DATA_DIR` strips `products.json` in the full suite). App-importing tests need dummy `OPENAI_API_KEY`/`PINECONE_API_KEY`.
- **Email normalization:** compare/store emails via `household._norm` / `.strip().lower()` as the surrounding code does.

---

### Task 1: Household pay-consent columns + helpers

The consent foundation. Mirrors the shipped `share_consent`/`set_share_consent`/`can_view`/`viewable_members_for` idiom exactly.

**Files:**
- Modify: `dashboard/household.py` (`init_household_tables`, plus new helpers)
- Test: `tests/test_household_pay_consent.py` (create)

**Interfaces:**
- Produces:
  - `can_pay(cx, payer_email, member_email) -> bool`
  - `payable_members_for(cx, payer_email) -> list[dict]` — each `{"member_email", "label", "pay_share_scope"}`
  - `set_pay_consent(cx, primary_email, member_email, consent, share_scope=None) -> None`
  - columns `household_members.pay_consent INTEGER DEFAULT 0`, `pay_share_scope TEXT DEFAULT 'amount_only'`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_household_pay_consent.py
import sqlite3
from dashboard import household as hh

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    hh.init_household_tables(cx)
    return cx

def test_pay_consent_default_off_and_grant_flow():
    cx = _cx()
    hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
    # default: no pay consent
    assert hh.can_pay(cx, "steve@x.com", "michael@x.com") is False
    assert hh.payable_members_for(cx, "steve@x.com") == []
    # member grants pay consent with line-item visibility
    hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 1, share_scope="line_items")
    assert hh.can_pay(cx, "steve@x.com", "michael@x.com") is True
    pm = hh.payable_members_for(cx, "steve@x.com")
    assert pm == [{"member_email": "michael@x.com", "label": "", "pay_share_scope": "line_items"}]
    # revoke is non-destructive to the link, just flips the flag
    hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 0)
    assert hh.can_pay(cx, "steve@x.com", "michael@x.com") is False

def test_pay_consent_self_pay_guard():
    cx = _cx()
    hh.add_member(cx, "steve@x.com", "steve@x.com", relationship="")
    hh.set_pay_consent(cx, "steve@x.com", "steve@x.com", 1)
    assert hh.can_pay(cx, "steve@x.com", "steve@x.com") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_household_pay_consent.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.household' has no attribute 'can_pay'`

- [ ] **Step 3: Add the columns in `init_household_tables`**

After the existing `cc_enabled` ALTER block (near line 52), add — mirroring that try/except idiom:

```python
    # caregiver-pay columns (additive). pay_consent default 0 (money is more
    # sensitive than view: an adult must opt IN). pay_share_scope gates what the
    # payer sees of a payable order.
    try:
        cx.execute("ALTER TABLE household_members ADD COLUMN pay_consent INTEGER DEFAULT 0")
    except Exception:
        pass
    try:
        cx.execute("ALTER TABLE household_members ADD COLUMN pay_share_scope TEXT DEFAULT 'amount_only'")
    except Exception:
        pass
```

- [ ] **Step 4: Add the helpers** (place beside `can_view`/`viewable_members_for`/`set_share_consent`)

```python
def can_pay(cx, payer_email, member_email):
    """True iff the member granted this payer pay-consent. Self-pay never qualifies."""
    p, m = _norm(payer_email), _norm(member_email)
    if not p or not m or p == m:
        return False
    return cx.execute(
        "SELECT 1 FROM household_members WHERE primary_email=? AND member_email=? "
        "AND pay_consent=1 LIMIT 1", (p, m)).fetchone() is not None


def payable_members_for(cx, payer_email):
    """Members who granted this payer pay-consent, with each one's share scope."""
    rows = cx.execute(
        "SELECT member_email, label, COALESCE(pay_share_scope,'amount_only') "
        "FROM household_members WHERE primary_email=? AND pay_consent=1 "
        "ORDER BY created_at, id", (_norm(payer_email),)).fetchall()
    return [{"member_email": r[0], "label": r[1] or "",
             "pay_share_scope": r[2] or "amount_only"} for r in rows]


def set_pay_consent(cx, primary_email, member_email, consent, share_scope=None):
    """MEMBER-controlled. Optionally set the share scope in the same write.
    Self-pay (payer==member) is rejected — you never authorize paying your own orders."""
    p, m = _norm(primary_email), _norm(member_email)
    if p == m:
        return
    if share_scope in ("amount_only", "line_items"):
        cx.execute("UPDATE household_members SET pay_consent=?, pay_share_scope=? "
                   "WHERE primary_email=? AND member_email=?",
                   (1 if consent else 0, share_scope, p, m))
    else:
        cx.execute("UPDATE household_members SET pay_consent=? "
                   "WHERE primary_email=? AND member_email=?",
                   (1 if consent else 0, p, m))
    cx.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_household_pay_consent.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add dashboard/household.py tests/test_household_pay_consent.py
git commit -m "feat(caregiver-pay): pay_consent + pay_share_scope on household link"
```

---

### Task 2: `payer_email` on the order_payments ledger

The attribution write. `payer_email` on the row; `add_payment` learns to stamp it.

**Files:**
- Modify: `dashboard/order_payments.py` (`ensure_table`, `_insert`, `add_payment`)
- Test: `tests/test_order_payments_payer.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `add_payment(cx, order_id, amount_cents, method, *, ..., payer_email=None)` — stamps `order_payments.payer_email`. Column `order_payments.payer_email TEXT` (nullable).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_order_payments_payer.py
import sqlite3
from dashboard import order_payments as op

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    op.ensure_table(cx)
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, name TEXT, channel TEXT)")
    cx.execute("INSERT INTO orders (id, email, name, channel) VALUES (1,'michael@x.com','Michael','web')")
    return cx

def test_add_payment_without_payer_is_null():
    cx = _cx()
    row = op.add_payment(cx, 1, 5000, "Zelle")
    assert row["payer_email"] is None

def test_add_payment_stamps_payer():
    cx = _cx()
    row = op.add_payment(cx, 1, 5000, "Zelle", payer_email="steve@x.com")
    assert row["payer_email"] == "steve@x.com"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_order_payments_payer.py -v`
Expected: FAIL — `KeyError: 'payer_email'` (column absent)

- [ ] **Step 3: Add the column in `ensure_table`**

After the `CREATE INDEX ... idx_order_payments_order` line, add:

```python
    try:
        cx.execute("ALTER TABLE order_payments ADD COLUMN payer_email TEXT")
    except Exception:
        pass
```

- [ ] **Step 4: Thread `payer_email` through `_insert` and `add_payment`**

In `_insert`, add the param and column:

```python
def _insert(cx, order_id, *, kind, amount_cents, method, source, external_ref,
            refunds_payment_id, paid_at, note, actor, payer_email=None):
    now = _now()
    new_id = dbwrite.insert_returning_id(
        cx,
        "INSERT INTO order_payments (order_id, kind, amount_cents, method, "
        "source, external_ref, refunds_payment_id, paid_at, note, status, "
        "qbo_sync, created_at, updated_at, created_by, payer_email) "
        "VALUES (?,?,?,?,?,?,?,?,?,'active','pending',?,?,?,?)",
        (order_id, kind, int(amount_cents), method, source, external_ref,
         refunds_payment_id, paid_at or now, note, now, now, actor, payer_email))
    cx.commit()
    return _row(cx, new_id)
```

In `add_payment`, add `payer_email=None` to the signature and pass it to `_insert`:

```python
def add_payment(cx, order_id, amount_cents, method, *, source="manual",
                external_ref=None, paid_at=None, note=None, actor=None,
                qbo_txn_id=None, skip_qbo_push=False, payer_email=None):
    ...
    row = _insert(cx, order_id, kind="payment", amount_cents=amount_cents,
                  method=method, source=source, external_ref=external_ref,
                  refunds_payment_id=None, paid_at=paid_at, note=note,
                  actor=actor, payer_email=(payer_email or None))
    ...
```

Note: the existing `external_ref` duplicate short-circuit (returns the existing row before insert) is unchanged — payer attribution rides the first successful insert, keeping idempotency keyed on `order_id + external_ref`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_order_payments_payer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add dashboard/order_payments.py tests/test_order_payments_payer.py
git commit -m "feat(caregiver-pay): nullable payer_email on order_payments ledger"
```

---

### Task 3: Money-view attribution (`COALESCE` seam) + refund-follows-payer

The one read change: the money view attributes a row to its payer when set. Also make refunds inherit the original payment's payer so they net against the payer.

**Files:**
- Modify: `dashboard/order_payments.py` (`ledger_rows_for_payments_view` SELECT; `add_refund`/refund insert path)
- Test: `tests/test_order_payments_payer.py` (extend)

**Interfaces:**
- Consumes: `add_payment(..., payer_email=)` from Task 2.
- Produces: `ledger_rows_for_payments_view` rows whose `email` is `COALESCE(payer_email, o.email)`. A refund row copies its parent payment's `payer_email`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_money_view_attributes_to_payer():
    cx = _cx()
    op.add_payment(cx, 1, 5000, "Zelle", payer_email="steve@x.com")
    op.add_payment(cx, 1, 2000, "Zelle")  # self-paid, payer NULL
    cx.row_factory = sqlite3.Row
    rows = op.ledger_rows_for_payments_view(cx)
    emails = sorted(r["email"] for r in rows)
    assert emails == ["michael@x.com", "steve@x.com"]

def test_refund_inherits_payer():
    cx = _cx()
    pay = op.add_payment(cx, 1, 5000, "Zelle", payer_email="steve@x.com")
    op.add_refund(cx, 1, 5000, "Zelle", refunds_payment_id=pay["id"])
    ref = cx.execute("SELECT payer_email FROM order_payments WHERE kind='refund'").fetchone()
    assert ref[0] == "steve@x.com"
```

> If the refund entry point is named differently (e.g. `record_refund`), use that name and its existing required args; the assertion — refund row carries the parent's `payer_email` — is what matters.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_order_payments_payer.py -v`
Expected: FAIL — money-view test returns `["michael@x.com", "michael@x.com"]`; refund test returns `None`.

- [ ] **Step 3: Change the money-view SELECT**

In `ledger_rows_for_payments_view`, change the selected email from `o.email` to a coalesce over the payer, keeping the alias `email` the loop reads:

```python
        "op.external_ref AS op_ref, op.paid_at, op.created_at, "
        "COALESCE(op.payer_email, o.email) AS email, o.name, o.channel "
```

- [ ] **Step 4: Make refunds inherit the payer**

In the refund insert path (`add_refund` / the `_insert(..., kind="refund", ...)` call), look up the parent payment's `payer_email` when `refunds_payment_id` is given and pass it through:

```python
    parent_payer = None
    if refunds_payment_id is not None:
        _p = _row(cx, refunds_payment_id)
        parent_payer = (_p or {}).get("payer_email")
    row = _insert(cx, order_id, kind="refund", amount_cents=amount_cents,
                  method=method, source=source, external_ref=external_ref,
                  refunds_payment_id=refunds_payment_id, paid_at=paid_at,
                  note=note, actor=actor, payer_email=parent_payer)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_order_payments_payer.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Mutation-check the guard, then commit**

Temporarily revert the SELECT to `o.email`; run `test_money_view_attributes_to_payer` and confirm it goes **red**; restore.

```bash
git add dashboard/order_payments.py tests/test_order_payments_payer.py
git commit -m "feat(caregiver-pay): money view attributes to payer; refunds inherit payer"
```

---

### Task 4: Member pay-consent endpoint + feature flag

Michael grants/revokes from his own portal. Mirrors `/api/portal/<token>/share-consent`.

**Files:**
- Modify: `app.py` (add `_caregiver_pay_enabled()`, add `api_portal_pay_consent`)
- Test: `tests/test_caregiver_pay_endpoint.py` (create)

**Interfaces:**
- Consumes: `household.set_pay_consent` (Task 1).
- Produces: `POST /api/portal/<token>/pay-consent` body `{caregiver_email, consent, share_scope?}`; token owner is treated as the **member**.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_caregiver_pay_endpoint.py
import os
os.environ.setdefault("OPENAI_API_KEY", "test"); os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ["CAREGIVER_PAY_ENABLED"] = "1"
import app as appmod

def test_pay_consent_endpoint_sets_member_consent(monkeypatch):
    client = appmod.app.test_client()
    # portal token resolves to the MEMBER (michael)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "michael@x.com"})
    # seed the link so the UPDATE has a row
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        from dashboard import household as hh
        hh.init_household_tables(cx)
        hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
    r = client.post("/api/portal/tok123/pay-consent",
                    json={"caregiver_email": "steve@x.com", "consent": True, "share_scope": "amount_only"})
    assert r.get_json()["recorded"] is True
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        from dashboard import household as hh
        assert hh.can_pay(cx, "steve@x.com", "michael@x.com") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_caregiver_pay_endpoint.py -v`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Add the flag helper** (beside `_household_sharing_enabled`)

```python
def _caregiver_pay_enabled():
    return (os.environ.get("CAREGIVER_PAY_ENABLED", "") or "").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 4: Add the endpoint** (beside `api_portal_share_consent`)

```python
@app.route("/api/portal/<token>/pay-consent", methods=["POST"])
def api_portal_pay_consent(token):
    """The MEMBER authorizes a caregiver to pay their orders. Token-scoped: only
    affects a link where the token's email is the MEMBER."""
    if not _caregiver_pay_enabled():
        return jsonify({"ok": True, "recorded": False, "reason": "disabled"})
    from dashboard import client_portal as _cp
    from dashboard import household as _hh
    data = request.get_json(silent=True) or {}
    caregiver = (data.get("caregiver_email") or "").strip().lower()
    consent = 1 if data.get("consent") else 0
    scope = data.get("share_scope") if data.get("share_scope") in ("amount_only", "line_items") else None
    with _db_lock, db.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx); _hh.init_household_tables(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"ok": False, "error": "not found"}), 404
        member = (portal.get("email") or "").strip().lower()
        _hh.set_pay_consent(cx, caregiver, member, consent, share_scope=scope)
    return jsonify({"ok": True, "recorded": True, "consent": consent})
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_caregiver_pay_endpoint.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_caregiver_pay_endpoint.py
git commit -m "feat(caregiver-pay): member pay-consent endpoint behind CAREGIVER_PAY_ENABLED"
```

---

### Task 5: Caregiver payable-orders portal block (Steve sees)

The caregiver's thin, firewall-safe surface — a new `_caregiver_pay_block` wired into the portal payload. It must never touch the member's biofield/consult/points.

**Files:**
- Modify: `dashboard/portal_view.py` (add `_caregiver_pay_block`, wire into `get_portal_view`)
- Test: `tests/test_caregiver_pay_block.py` (create)

**Interfaces:**
- Consumes: `household.payable_members_for` (Task 1).
- Produces: `get_portal_view(...)["caregiver_pay"]` → `{"members": [...], "orders": [ {order_id, beneficiary_email, beneficiary_name, amount_dollars, token, items} ]}`. `items` is `None` unless that member's scope is `line_items`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_caregiver_pay_block.py
import sqlite3
from dashboard import portal_view as pv
from dashboard import household as hh

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    hh.init_household_tables(cx)
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, total_cents INTEGER, "
               "invoice_token TEXT, items_json TEXT, pay_status TEXT, status TEXT)")
    cx.execute("INSERT INTO orders VALUES (1,'michael@x.com',5000,'tok1','[{\"slug\":\"a\"}]','','open')")
    return cx

def test_block_hides_items_when_amount_only():
    cx = _cx()
    hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
    hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 1, share_scope="amount_only")
    block = pv._caregiver_pay_block(cx, "steve@x.com", True)
    assert len(block["orders"]) == 1
    o = block["orders"][0]
    assert o["order_id"] == 1 and o["amount_dollars"] == "50.00"
    assert o["items"] is None  # amount_only hides line items

def test_block_empty_without_consent_and_when_disabled():
    cx = _cx()
    hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
    assert pv._caregiver_pay_block(cx, "steve@x.com", True)["orders"] == []   # no consent
    hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 1)
    assert pv._caregiver_pay_block(cx, "steve@x.com", False) == {"members": [], "orders": []}  # flag off
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_caregiver_pay_block.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.portal_view' has no attribute '_caregiver_pay_block'`

- [ ] **Step 3: Add the block builder** (beside the other `_*_block` functions)

```python
def _caregiver_pay_block(cx, email, enabled):
    """Orders this person may pay for household members who granted pay-consent.
    Thin + firewall-safe: amounts/status only, line items only when the member's
    scope allows. NEVER reads the member's clinical data."""
    if not enabled:
        return {"members": [], "orders": []}
    try:
        from dashboard import household as _hh
        members = _hh.payable_members_for(cx, email)
    except Exception:
        return {"members": [], "orders": []}
    orders = []
    for mem in members:
        scope = mem["pay_share_scope"]
        try:
            rows = cx.execute(
                "SELECT id, total_cents, COALESCE(invoice_token,''), COALESCE(items_json,'[]') "
                "FROM orders WHERE lower(coalesce(email,''))=? "
                "AND coalesce(pay_status,'')<>'paid' AND coalesce(invoice_token,'')<>'' "
                "AND coalesce(status,'') NOT IN ('cancelled','delivered','done') "
                "ORDER BY id DESC", (mem["member_email"],)).fetchall()
        except Exception:
            rows = []
        for oid, tc, tok, items in rows:
            orders.append({
                "order_id": oid,
                "beneficiary_email": mem["member_email"],
                "beneficiary_name": mem["label"] or mem["member_email"],
                "amount_dollars": f"{(tc or 0) / 100:.2f}",
                "token": tok,
                "items": (items if scope == "line_items" else None),
            })
    return {"members": members, "orders": orders}
```

- [ ] **Step 4: Wire it into `get_portal_view`**

Add a parameter `caregiver_pay_enabled=False` to `get_portal_view`, and in the returned dict (beside `"orders": _orders_block(...)`):

```python
        "caregiver_pay": _caregiver_pay_block(cx, email, caregiver_pay_enabled),
```

At the `get_portal_view` call site in `app.py`, pass `caregiver_pay_enabled=_caregiver_pay_enabled()`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_caregiver_pay_block.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Firewall assertion, then commit**

Add a test asserting the block dict's keys are exactly `{"members","orders"}` and each order dict's keys ⊆ `{"order_id","beneficiary_email","beneficiary_name","amount_dollars","token","items"}` — proving no clinical field can ride along. Then:

```bash
git add dashboard/portal_view.py app.py tests/test_caregiver_pay_block.py
git commit -m "feat(caregiver-pay): firewall-safe payable-orders portal block"
```

---

### Task 6: Caregiver Stripe checkout + payer-stamped fulfillment (Steve pays)

Steve initiates a card payment for Michael's order; the payment books to Steve. The testable seam is `_fulfill_caregiver_pay`, which maps a Stripe session's metadata → `add_payment(payer_email=…)`. Idempotency rides `external_ref=payment_intent` (existing dedup).

**Files:**
- Modify: `app.py` (add `_fulfill_caregiver_pay`, `POST /api/portal/<token>/caregiver-pay`, `/caregiver-pay/return`, `/webhook/stripe` dispatch branch)
- Test: `tests/test_caregiver_pay_fulfill.py` (create)

**Interfaces:**
- Consumes: `household.can_pay` (Task 1), `add_payment(..., payer_email=)` (Task 2), `stripe_pay.create_checkout_session` / `get_session`.
- Produces: `_fulfill_caregiver_pay(session_id) -> None` (never raises); metadata contract `{"kind":"caregiver-pay","order_id":"<id>","payer_email":"<caregiver>"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_caregiver_pay_fulfill.py
import os
os.environ.setdefault("OPENAI_API_KEY", "test"); os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ["CAREGIVER_PAY_ENABLED"] = "1"
import app as appmod
from dashboard import order_payments as op

def test_fulfill_books_payment_to_payer(monkeypatch):
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        op.ensure_table(cx)
        cx.execute("INSERT INTO orders (email, name) VALUES ('michael@x.com','Michael')")
        oid = cx.execute("SELECT id FROM orders WHERE email='michael@x.com' ORDER BY id DESC LIMIT 1").fetchone()[0]
        cx.commit()
    fake = {"payment_status": "paid", "payment_intent": "pi_test_1", "amount_total": 5000,
            "metadata": {"kind": "caregiver-pay", "order_id": str(oid), "payer_email": "steve@x.com"}}
    monkeypatch.setattr(appmod, "_bos_actor", lambda: "system", raising=False)
    monkeypatch.setattr("dashboard.stripe_pay.get_session", lambda sid: fake)
    appmod._fulfill_caregiver_pay("cs_test_1")
    appmod._fulfill_caregiver_pay("cs_test_1")  # idempotent — same pi, no double row
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        rows = cx.execute("SELECT payer_email FROM order_payments WHERE order_id=? AND kind='payment'", (oid,)).fetchall()
    assert [r[0] for r in rows] == ["steve@x.com"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_caregiver_pay_fulfill.py -v`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_fulfill_caregiver_pay'`

- [ ] **Step 3: Add `_fulfill_caregiver_pay`** (model on the in-house branch at app.py ~10883 and `_fulfill_family_plan`)

```python
def _fulfill_caregiver_pay(session_id):
    """Record a caregiver's card payment against the beneficiary's order, stamped
    to the PAYER. Never raises. Idempotent via external_ref=payment_intent."""
    try:
        from dashboard import stripe_pay as _sp
        sess = _sp.get_session(session_id)
        md = sess.get("metadata") or {}
        if md.get("kind") != "caregiver-pay":
            return
        if sess.get("payment_status") != "paid":
            return
        order_id = int(md.get("order_id") or 0)
        payer = (md.get("payer_email") or "").strip().lower()
        pi_id = sess.get("payment_intent")
        if not order_id or not payer:
            return
        with _db_lock, db.connect(LOG_DB) as cx:
            _op.ensure_table(cx)
            _op.add_payment(cx, order_id, int(sess.get("amount_total") or 0),
                            "Credit card (Stripe)", source="stripe",
                            external_ref=pi_id, payer_email=payer)
            _bos_orders.set_order_payment(cx, order_id, method="card",
                                          amount_cents=int(sess.get("amount_total") or 0))
            if pi_id:
                _bos_orders.set_order_stripe_pi(cx, order_id, pi_id)
    except Exception as _e:
        print(f"[caregiver-pay] fulfill: {_e!r}", flush=True)
```

- [ ] **Step 4: Add the initiate endpoint + return route + webhook branch**

```python
@app.route("/api/portal/<token>/caregiver-pay", methods=["POST"])
def api_portal_caregiver_pay(token):
    """Payer starts a card checkout for a beneficiary's order. Gated on active
    pay-consent AT INITIATION; metadata then carries the payer snapshot."""
    if not _caregiver_pay_enabled():
        return jsonify({"ok": False, "error": "disabled"}), 404
    from dashboard import stripe_pay as _sp, household as _hh
    data = request.get_json(silent=True) or {}
    order_id = int(data.get("order_id") or 0)
    with _db_lock, db.connect(LOG_DB) as cx:
        _hh.init_household_tables(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"ok": False, "error": "not found"}), 404
        payer = (portal.get("email") or "").strip().lower()
        o = cx.execute("SELECT email, total_cents FROM orders WHERE id=?", (order_id,)).fetchone()
        if not o:
            return jsonify({"ok": False, "error": "order not found"}), 404
        beneficiary = (o["email"] or "").strip().lower()
        if not _hh.can_pay(cx, payer, beneficiary):
            return jsonify({"ok": False, "error": "not authorized"}), 403
        amount = int(o["total_cents"] or 0)
    base = PUBLIC_BASE_URL.rstrip("/")
    sess = _sp.create_checkout_session(
        amount, customer_email=payer, description=f"Payment for order #{order_id}",
        metadata={"kind": "caregiver-pay", "order_id": str(order_id), "payer_email": payer},
        success_url=f"{base}/caregiver-pay/return?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/portal/{token}")
    return jsonify({"ok": True, "url": sess.get("url")})


@app.route("/caregiver-pay/return", methods=["GET"])
def caregiver_pay_return():
    sid = request.args.get("session_id", "")
    if sid:
        _fulfill_caregiver_pay(sid)
    return redirect(f"{PUBLIC_BASE_URL.rstrip('/')}/")
```

In the `/webhook/stripe` dispatch, add a branch mirroring the family-plan one:

```python
        if (md.get("kind") == "caregiver-pay"):
            _fulfill_caregiver_pay(session_id)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_caregiver_pay_fulfill.py -v`
Expected: PASS (payer stamped once; second call is a no-op via external_ref dedup)

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_caregiver_pay_fulfill.py
git commit -m "feat(caregiver-pay): Stripe checkout + payer-stamped fulfillment (receipt to payer)"
```

---

### Task 7: Manual / Zelle payer selector (console records Steve's Zelle)

A Zelle from Steve for Michael's order books to Steve. The console `add-payment` route learns an optional `payer_email`, validated by `can_pay`.

**Files:**
- Modify: `app.py` (`api_order_payments_add`, ~line 42455)
- Test: `tests/test_caregiver_pay_manual.py` (create)

**Interfaces:**
- Consumes: `household.can_pay` (Task 1), `add_payment(..., payer_email=)` (Task 2).
- Produces: the console add-payment endpoint accepts `payer_email` and stamps it when `can_pay(payer, order.email)` holds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_caregiver_pay_manual.py
import os
os.environ.setdefault("OPENAI_API_KEY", "test"); os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ["CAREGIVER_PAY_ENABLED"] = "1"
import app as appmod
from dashboard import order_payments as op, household as hh

def test_manual_payment_stamps_authorized_payer(monkeypatch):
    monkeypatch.setattr(appmod, "_bos_actor", lambda: "console", raising=False)
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        op.ensure_table(cx); hh.init_household_tables(cx)
        cx.execute("INSERT INTO orders (email, name, total_cents) VALUES ('michael@x.com','Michael',5000)")
        oid = cx.execute("SELECT id FROM orders WHERE email='michael@x.com' ORDER BY id DESC LIMIT 1").fetchone()[0]
        hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
        hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 1)
        cx.commit()
    client = appmod.app.test_client()
    r = client.post(f"/api/console/order/{oid}/payments/add",
                    json={"amount_cents": 5000, "method": "Zelle", "payer_email": "steve@x.com"},
                    headers={"X-Console-Key": appmod.CONSOLE_SECRET})
    assert r.status_code == 200
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        row = cx.execute("SELECT payer_email FROM order_payments WHERE order_id=?", (oid,)).fetchone()
    assert row[0] == "steve@x.com"
```

> Use the real route path and auth header for `api_order_payments_add` as they exist at app.py ~42455 (adjust the URL/headers in the test to match; the assertion is the invariant).

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_caregiver_pay_manual.py -v`
Expected: FAIL — `payer_email` is `None` (route ignores it).

- [ ] **Step 3: Thread `payer_email` through the route**

In `api_order_payments_add`, after resolving the order and its email, read and validate the optional payer:

```python
    payer_email = (data.get("payer_email") or "").strip().lower() or None
    if payer_email and _caregiver_pay_enabled():
        from dashboard import household as _hh
        owner = (order_row.get("email") or "").strip().lower()
        if not _hh.can_pay(cx, payer_email, owner):
            return jsonify({"ok": False, "error": "payer not authorized"}), 403
    else:
        payer_email = None
```

Pass `payer_email=payer_email` into the existing `add_payment(...)` call in that route.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_caregiver_pay_manual.py -v`
Expected: PASS

- [ ] **Step 5: Mutation-check the auth guard, then commit**

Temporarily drop the `can_pay` check (always stamp); add/keep a test that an *unauthorized* payer is rejected 403 and confirm it goes **red** without the guard; restore.

```bash
git add app.py tests/test_caregiver_pay_manual.py
git commit -m "feat(caregiver-pay): console manual/Zelle payer selector, gated by can_pay"
```

---

### Task 8: Beneficiary "paid by caregiver" badge (Michael's view)

Michael still sees his order; it now reads "paid by caregiver" when someone else's `payer_email` is on a payment. Per-payment, so a split reads correctly.

**Files:**
- Modify: `dashboard/order_payments.py` (add `caregiver_payers_for`), `dashboard/portal_view.py` (`_orders_block`)
- Test: `tests/test_order_payments_payer.py` (extend)

**Interfaces:**
- Consumes: `order_payments.caregiver_payers_for`.
- Produces: `_orders_block` order dicts gain `paid_by_caregiver: bool` (and `caregiver_payers: list[str]`).

- [ ] **Step 1: Write the failing test** (append to `tests/test_order_payments_payer.py`)

```python
def test_caregiver_payers_for_lists_foreign_payers():
    cx = _cx()
    op.add_payment(cx, 1, 3000, "Zelle", payer_email="steve@x.com")
    op.add_payment(cx, 1, 2000, "Zelle")  # self-paid → not a caregiver payer
    assert op.caregiver_payers_for(cx, 1, "michael@x.com") == ["steve@x.com"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_order_payments_payer.py::test_caregiver_payers_for_lists_foreign_payers -v`
Expected: FAIL — `AttributeError: ... has no attribute 'caregiver_payers_for'`

- [ ] **Step 3: Add `caregiver_payers_for`**

```python
def caregiver_payers_for(cx, order_id, owner_email):
    """Distinct non-owner payer_emails on active payments for this order."""
    rows = cx.execute(
        "SELECT DISTINCT payer_email FROM order_payments WHERE order_id=? "
        "AND kind='payment' AND status='active' AND payer_email IS NOT NULL "
        "AND lower(payer_email) <> lower(?)", (order_id, owner_email or "")).fetchall()
    return [r[0] for r in rows]
```

- [ ] **Step 4: Annotate orders in `_orders_block`**

Inside the `for o in _o.list_orders_by_email(cx, email, ...)` loop, after building each order dict, add:

```python
            try:
                payers = _op.caregiver_payers_for(cx, o["id"], email)
            except Exception:
                payers = []
            order_dict["paid_by_caregiver"] = bool(payers)
            order_dict["caregiver_payers"] = payers
```

(Use the loop's existing per-order dict variable name and `_op` = `dashboard.order_payments`, imported at the top of the module as the other `_x` imports are.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_order_payments_payer.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add dashboard/order_payments.py dashboard/portal_view.py tests/test_order_payments_payer.py
git commit -m "feat(caregiver-pay): 'paid by caregiver' badge on beneficiary's order"
```

---

## Rollout (after all tasks pass)

- [ ] Run the full suite via the repo's standard runner; confirm no new failures against the `known_failures` ratchet.
- [ ] Render-verify with `CAREGIVER_PAY_ENABLED=1` against a synthetic payload: Steve's `caregiver_pay` block renders payable orders (amount-only hides items); Michael's order shows the badge; Michael's biofield/consult are absent from Steve's payload.
- [ ] Flip `CAREGIVER_PAY_ENABLED` in Doppler (`remedy-match/prd`) — remember it's two deploys and read at startup, so restart Render; verify live via `/api/portal/<token>` for a real Steve↔Michael pair.

## Follow-up (not in this plan)

- **Identity-merge remap.** When a beneficiary's email is merged/changed, the identity-merge routine must remap `order_payments.payer_email` and `household_members` (both email-keyed) alongside `orders.email`. Handle in the identity-merge path as a separate change; not blocking for the Steve↔Michael case.

## Spec deviations (intentional, logged here)

- **`payer_person_id` dropped for v1** (YAGNI): portal identity is email-keyed end to end, so `payer_email` alone carries attribution. Add `payer_person_id` only if a numeric identity join becomes needed.
- **`_published_invoices_for` / `_past_invoices_for` are NOT changed.** They read the `orders` table (no `payer_email`), and the order legitimately stays the beneficiary's — Michael *should* see his own order as a paid receipt. The real `COALESCE` seam is the `order_payments`-joined money view (Task 3). The spec's claim that the same filter applies to those two invoice reads was an over-reach.
- **No `expense_target` column** (YAGNI): expense always books to the payer — that is the feature.
