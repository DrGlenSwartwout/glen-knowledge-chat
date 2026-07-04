# Dispensary L2 Settlement Coverage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Credit L2 points to a practitioner's upline on EVERY paid dispensary order — first purchase and every reorder, across ALL pay methods (card, Zelle, Wise) — with no L1, and record `pay_method` on dispensary orders so the card/alt-pay split is exact going forward.

**Architecture:** A single pure settlement helper `dashboard/dispensary_rewards.settle_dispensary_l2(cx, order, order_ref)` credits the practitioner's upline L2, resolving the practitioner from the order's OWN `practitioner_id` (stamped at checkout for every pay method), never from the patient's referral row (which could be an unrelated Ambassador owner). It is called from both paid-transition paths: the card path (`app._settle_order_points`) and the alt-pay manual-confirm path (`dashboard.orders._record_payment_exec`). The existing `_settle_referrer_reward` L2 branch is disabled for `kind='dispensary_portal'` rows so the first order is never double-credited.

**Tech Stack:** Python, Flask, SQLite (`chat_log.db`/`LOG_DB`) for orders/referral_redemptions/points_ledger; Supabase for `practitioners` (pid→email). pytest.

**Context — why now:** prod has ZERO dispensary sales as of 2026-07-04 (verified via `/api/console/dispensary-pay-mix` and the backfill dry-run, both zero). No history to migrate and no live sales at risk — the safest moment to change the shared settlement path. Spec: `docs/superpowers/specs/2026-07-04-dispensary-reward-settlement-coverage.md`.

## Global Constraints

- **L2 only, never L1** on dispensary orders (the practitioner's pay is their wholesale markup). No code path in this plan credits the practitioner (L1).
- **Resolution rule (critical):** resolve the owning practitioner from `order["practitioner_id"]` → `practitioner_email_by_id(pid)`. NEVER from `owner_of_referee(patient_email)` — the patient may carry a prior `kind='referral'` Ambassador row (PK on `referee_email`), which would misattribute L2 to an unrelated Ambassador's upline.
- **No double-pay:** the new L2 credit is keyed `reason='referral_reward_l2', order_ref='disp_l2:<invoice>'` — a distinct key space from the Ambassador L2 (`order_ref='referral_l2:<referee_email>'`). Because those keys differ, key-collision idempotency will NOT prevent a first-order double-pay; the guard is instead that `_settle_referrer_reward`'s L2 branch is turned OFF for `kind='dispensary_portal'` (Task 4). The two L2 sources must be mutually exclusive by construction, not by key collision.
- **Idempotent per order** via `points.has_entry(order_ref='disp_l2:<invoice>', reason='referral_reward_l2')` — each reorder has its own invoice → its own key → pays once; replays no-op.
- **Config parsing mirrors app.py exactly:** `REFERRALS` truthy = value in `("1","true","yes")`; `REFERRAL_TIER2_ENABLED` = value in `("1","true","yes","on")`; reward pct = `max(0, int(os.environ.get("REFERRER_REWARD_PCT","0")))`. L2 = `product_cents * pct // 200` (half the L1 rate), matching `_settle_referrer_reward`.
- **Anti-cycle:** L2 owner must exist and differ from BOTH the patient and the practitioner.
- **Never raise into checkout or settlement.** Every new call site is best-effort (try/except, log, continue).
- `pay_method`/`practitioner_id` threaded via `_ingest_order`/`upsert_order` default to `None` and must NOT clobber a later payment-time write when None.

## Files

- Modify: `dashboard/orders.py` — `init_orders_table` (add `practitioner_id` column), `upsert_order` (thread `pay_method`, `practitioner_id`), `_record_payment_exec` (alt-pay L2 call)
- Modify: `app.py` — `_ingest_order` (thread the two fields), `api_client_checkout` (pass them), `_settle_referrer_reward` (L2 double-pay guard), `_settle_order_points` (card L2 call)
- Create: `dashboard/dispensary_rewards.py` — `settle_dispensary_l2`
- Test: `tests/test_dispensary_l2.py`, `tests/test_orders_pay_method_pid.py` (create); extend nothing destructively

---

### Task 1: Thread `pay_method` + `practitioner_id` onto orders

**Files:**
- Modify: `dashboard/orders.py` (`init_orders_table` ~line 64-90; `upsert_order` lines 93-145)
- Test: `tests/test_orders_pay_method_pid.py` (create)

**Interfaces:**
- Produces: `upsert_order(..., pay_method=None, practitioner_id=None)` — writes each only when not None (conditional SET on update, always on insert with NULL default). Orders table gains a `practitioner_id TEXT` column (lazy ALTER). `pay_method` column already exists.

- [ ] **Step 1: Write the failing test**

Create `tests/test_orders_pay_method_pid.py`:

```python
import sqlite3
from dashboard import orders as o


def _cx():
    cx = sqlite3.connect(":memory:")
    o.init_orders_table(cx)
    return cx


def test_practitioner_id_column_exists():
    cx = _cx()
    cols = {r[1] for r in cx.execute("PRAGMA table_info(orders)")}
    assert "practitioner_id" in cols and "pay_method" in cols


def test_upsert_writes_pay_method_and_pid_on_insert():
    cx = _cx()
    o.upsert_order(cx, source="dispensary", external_ref="INV1", email="p@x.com",
                   total_cents=7000, pay_method="zelle", practitioner_id="prac-1")
    row = cx.execute("SELECT pay_method, practitioner_id FROM orders WHERE external_ref='INV1'").fetchone()
    assert row == ("zelle", "prac-1")


def test_upsert_none_does_not_clobber_existing():
    cx = _cx()
    o.upsert_order(cx, source="dispensary", external_ref="INV2", pay_method="card",
                   practitioner_id="prac-2", total_cents=100)
    # a later ingest without the fields (None) must not wipe them
    o.upsert_order(cx, source="dispensary", external_ref="INV2", total_cents=200)
    row = cx.execute("SELECT pay_method, practitioner_id, total_cents FROM orders WHERE external_ref='INV2'").fetchone()
    assert row == ("card", "prac-2", 200)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_orders_pay_method_pid.py -v`
Expected: FAIL — `practitioner_id` column missing / `upsert_order` has no such kwargs.

- [ ] **Step 3: Add the `practitioner_id` column**

In `dashboard/orders.py` `init_orders_table`, inside the `for ddl in (...)` additive-column loop (the block that already adds `pay_method`, `pay_status`, etc. ~line 65-85), add one entry:

```python
        "ALTER TABLE orders ADD COLUMN practitioner_id TEXT",
```

- [ ] **Step 4: Thread the two params through `upsert_order`**

Change the signature (line 93-97) to add the two params:

```python
def upsert_order(cx, *, source, external_ref, email="", name="", phone="",
                 items=None, total_cents=0, address=None, channel="retail",
                 status="new", get_cents=0, person_id=None,
                 discount_cents=0, points_redeemed_cents=0, shipping_cents=0,
                 invoice_note=None, adjustment_cents=0,
                 pay_method=None, practitioner_id=None):
```

In the UPDATE branch, after the `if invoice_note is not None:` block (line 126-128), add:

```python
        if pay_method is not None:
            sets.append("pay_method=?")
            vals.append(str(pay_method))
        if practitioner_id is not None:
            sets.append("practitioner_id=?")
            vals.append(str(practitioner_id))
```

In the INSERT branch, add both columns. Change the INSERT (lines 133-143) to:

```python
    cur = cx.execute(
        "INSERT INTO orders (created_at, source, external_ref, channel, email, name, "
        "phone, items_json, total_cents, address_json, status, get_cents, person_id, "
        "discount_cents, points_redeemed_cents, shipping_cents, invoice_note, adjustment_cents, "
        "pay_method, practitioner_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (_now(), source, ref, channel, email, name, phone,
         json.dumps(items or []), int(total_cents or 0), json.dumps(address or {}),
         status, int(get_cents or 0),
         (int(person_id) if person_id is not None else None),
         int(discount_cents or 0), int(points_redeemed_cents or 0), int(shipping_cents or 0),
         (str(invoice_note) if invoice_note is not None else None), int(adjustment_cents or 0),
         (str(pay_method) if pay_method is not None else None),
         (str(practitioner_id) if practitioner_id is not None else None)))
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_orders_pay_method_pid.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/orders.py tests/test_orders_pay_method_pid.py
git commit -m "feat(orders): thread pay_method + practitioner_id through upsert_order"
```

---

### Task 2: Capture `pay_method` + `practitioner_id` at the dispensary checkout

**Files:**
- Modify: `app.py` — `_ingest_order` (signature ~27540 + the `upsert_order` call ~27553); `api_client_checkout` (the `_ingest_order(source="dispensary", …)` call ~12526)
- Test: `tests/test_dispensary_checkout_capture.py` (create)

**Interfaces:**
- Consumes: `upsert_order(..., pay_method=, practitioner_id=)` (Task 1).
- Produces: `_ingest_order(..., pay_method=None, practitioner_id=None)` passes both to `upsert_order`. The dispensary checkout stamps `pay_method=<method>` and `practitioner_id=<pid>` on the order at ingest, for every pay method.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dispensary_checkout_capture.py`:

```python
import importlib
import sqlite3


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config["TESTING"] = True
    return appmod


def test_dispensary_checkout_stamps_method_and_pid(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: "prac-9")
    monkeypatch.setattr(appmod._pp, "practitioner_email_by_id", lambda pid: "doc@x.com")
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid, **kw: {"modules_completed": 1})
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: True)
    monkeypatch.setattr(appmod._dropship, "build_client_order",
                        lambda *a, **k: {"ok": True, "invoice_id": "INV-Z", "total": 70.0,
                                         "get_cents": 0})
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)
    c = appmod.app.test_client()
    # mirror the valid body shape from tests/test_client_routes.py::_VALID_BODY
    r = c.post("/api/client/DCODE/checkout",
               json={"email": "pat@x.com", "name": "Pat", "method": "zelle",
                     "items": [{"slug": "bone-builder", "qty": 1}],
                     "address": {"street": "1 A St", "city": "Hilo", "state": "HI",
                                 "zip": "96720", "country": "US"}})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = cx.execute("SELECT pay_method, practitioner_id FROM orders "
                         "WHERE external_ref='INV-Z' AND source='dispensary'").fetchone()
    assert row == ("zelle", "prac-9")
```

(If the request body is rejected, copy the exact `_VALID_BODY` from `tests/test_client_routes.py`.)

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc python3 -m pytest tests/test_dispensary_checkout_capture.py -v` (mkdir /tmp/dc first)
Expected: FAIL — pay_method/practitioner_id are NULL.

- [ ] **Step 3: Thread the fields through `_ingest_order`**

Change `_ingest_order`'s signature (app.py ~27540-27543) to add the two params:

```python
def _ingest_order(*, source, external_ref, email="", name="", phone="",
                  items=None, total_cents=0, address=None, channel="retail",
                  get_cents=0, discount_cents=0, points_redeemed_cents=0, shipping_cents=0,
                  status="new", paid_cents=None, pay_method=None, practitioner_id=None):
```

And pass them into the `upsert_order(...)` call (app.py ~27553-27559) by adding to its argument list:

```python
                shipping_cents=int(shipping_cents or 0), status=status,
                pay_method=pay_method, practitioner_id=practitioner_id)
```

- [ ] **Step 4: Pass them at the dispensary checkout**

In `api_client_checkout` (app.py ~12526), the `_ingest_order(source="dispensary", …)` call — add the two kwargs (`method` and `pid` are already in local scope at that point):

```python
    _ingest_order(source="dispensary",
                  external_ref=str(out.get("invoice_id") or ""),
                  email=email,
                  name=name,
                  total_cents=int(round((out.get("total") or 0) * 100)),
                  items=items,
                  address=ship,
                  channel="retail",
                  get_cents=out.get("get_cents", 0),
                  pay_method=method,
                  practitioner_id=pid)
```

- [ ] **Step 5: Run to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc python3 -m pytest tests/test_dispensary_checkout_capture.py tests/test_client_routes.py -v`
Expected: PASS (capture works; existing client-route suite still green).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_dispensary_checkout_capture.py
git commit -m "feat(dispensary): stamp pay_method + practitioner_id on the order at checkout"
```

---

### Task 3: `settle_dispensary_l2` — the shared L2 settlement helper

**Files:**
- Create: `dashboard/dispensary_rewards.py`
- Test: `tests/test_dispensary_l2.py` (create)

**Interfaces:**
- Consumes: `practitioner_email_by_id` (dashboard.practitioner_portal), `owner_of_referee` (dashboard.referrals), `points.has_entry/credit/init_points_table`.
- Produces: `settle_dispensary_l2(cx, order, order_ref) -> int` — L2 cents credited to the practitioner's upline (0 if none). Reads config from env. Idempotent per `disp_l2:<order_ref>`. Never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dispensary_l2.py`:

```python
import sqlite3
import dashboard.practitioner_portal as pp_mod
from dashboard import dispensary_rewards as dr, referrals as rf, points


def _cx(monkeypatch):
    monkeypatch.setenv("REFERRALS", "true")
    monkeypatch.setenv("REFERRAL_TIER2_ENABLED", "true")
    monkeypatch.setenv("REFERRER_REWARD_PCT", "20")
    cx = sqlite3.connect(":memory:")
    rf.init_tables(cx)
    points.init_points_table(cx)
    return cx


def _order(pid="prac-1", patient="pat@x.com", total=7000, shipping=1300, ref="INV1"):
    return {"practitioner_id": pid, "email": patient, "total_cents": total,
            "shipping_cents": shipping, "get_cents": 0, "external_ref": ref}


def test_credits_upline_l2_no_l1(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")  # who referred the doc
    got = dr.settle_dispensary_l2(cx, _order(), "INV1")
    # product 5700; L2 = 5700 * 20 // 200 = 570
    assert got == 570
    assert points.balance(cx, "upline@x.com") == 570
    assert points.balance(cx, "doc@x.com") == 0        # practitioner (L1) never credited
    assert points.balance(cx, "pat@x.com") == 0


def test_idempotent_per_invoice(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    dr.settle_dispensary_l2(cx, _order(), "INV1")
    assert dr.settle_dispensary_l2(cx, _order(), "INV1") == 0   # replay
    assert points.balance(cx, "upline@x.com") == 570


def test_reorder_new_invoice_pays_again(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    dr.settle_dispensary_l2(cx, _order(ref="INV1"), "INV1")
    dr.settle_dispensary_l2(cx, _order(ref="INV2"), "INV2")   # reorder
    assert points.balance(cx, "upline@x.com") == 1140          # 570 * 2


def test_no_l2_when_practitioner_has_no_upline(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    assert dr.settle_dispensary_l2(cx, _order(), "INV1") == 0   # no redemption row for doc


def test_resolves_from_order_pid_not_patient_referral(monkeypatch):
    """The patient carries an Ambassador referral row; L2 must NOT follow that chain."""
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "AMB", "ambassador@x.com", "pat@x.com", "INV-AMB")  # patient's own referrer
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    dr.settle_dispensary_l2(cx, _order(), "INV1")
    assert points.balance(cx, "upline@x.com") == 570              # doc's upline, correct
    assert points.balance(cx, "ambassador@x.com") == 0            # not the patient's ambassador chain


def test_no_l2_when_tier2_off(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setenv("REFERRAL_TIER2_ENABLED", "false")
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    assert dr.settle_dispensary_l2(cx, _order(), "INV1") == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_dispensary_l2.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the module**

Create `dashboard/dispensary_rewards.py`:

```python
"""L2-only settlement for dispensary sales.

A dispensary/drop-ship sale is paid at wholesale by the practitioner (their pay is the
markup), so it pays NO L1 — only the L2 override (points) to whoever referred the
PRACTITIONER into the system. Fires on every paid dispensary order (first + reorders),
across all pay methods, called from both the card path (app._settle_order_points) and
the alt-pay manual-confirm path (dashboard.orders._record_payment_exec).

Config is read from env, mirroring app.py's parsing exactly, so the two callers behave
identically without importing app.
"""
import os


def _referrals_on():
    return os.environ.get("REFERRALS", "").strip().lower() in ("1", "true", "yes")


def _tier2_on():
    return os.environ.get("REFERRAL_TIER2_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _reward_pct():
    try:
        return max(0, int(os.environ.get("REFERRER_REWARD_PCT", "0")))
    except (TypeError, ValueError):
        return 0


def settle_dispensary_l2(cx, order, order_ref):
    """Credit L2 points to the practitioner's upline on a paid dispensary order.
    Resolves the practitioner from order['practitioner_id'] (stamped at checkout), NOT
    from the patient's referral row. Idempotent per order_ref. Returns cents credited.
    Never raises into the caller."""
    try:
        if not _referrals_on() or not _tier2_on():
            return 0
        pct = _reward_pct()
        if pct <= 0:
            return 0
        pid = (str(order.get("practitioner_id") or "")).strip()
        if not pid:
            return 0
        from dashboard.practitioner_portal import practitioner_email_by_id
        from dashboard import referrals as _rf, points as _points
        practitioner = (practitioner_email_by_id(pid) or "").strip().lower()
        if not practitioner:
            return 0
        l2 = (_rf.owner_of_referee(cx, practitioner) or "").strip().lower()
        patient = (str(order.get("email") or "")).strip().lower()
        if not l2 or l2 == practitioner or l2 == patient:
            return 0
        product_cents = max(0, int(order.get("total_cents") or 0)
                            - int(order.get("shipping_cents") or 0)
                            - int(order.get("get_cents") or 0))
        reward_l2 = product_cents * pct // 200
        if reward_l2 <= 0:
            return 0
        _points.init_points_table(cx)
        key = f"disp_l2:{order_ref}"
        if _points.has_entry(cx, order_ref=key, reason="referral_reward_l2"):
            return 0
        _points.credit(cx, l2, value_cents=reward_l2, reason="referral_reward_l2", order_ref=key)
        return reward_l2
    except Exception as _e:
        print(f"[dispensary-l2] settle skipped ref={order_ref!r}: {_e!r}", flush=True)
        return 0
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_dispensary_l2.py -v`
Expected: PASS (all 6).

- [ ] **Step 5: Commit**

```bash
git add dashboard/dispensary_rewards.py tests/test_dispensary_l2.py
git commit -m "feat(dispensary): settle_dispensary_l2 — upline L2 on every paid dispensary order"
```

---

### Task 4: Double-pay guard — retire the old L2 path for `dispensary_portal`

**Files:**
- Modify: `app.py` — `_settle_referrer_reward` (the `if REFERRAL_TIER2_ENABLED:` block, ~line 5195)
- Test: `tests/test_portal_referral_reward.py` (extend — add one assertion)

**Interfaces:** no signature change. After this task, `_settle_referrer_reward` credits NO L2 for `kind='dispensary_portal'` redemptions (delegated entirely to `settle_dispensary_l2`), while `kind='referral'` L2 is unchanged.

- [ ] **Step 1: Add the failing assertion**

In `tests/test_portal_referral_reward.py`, add a test:

```python
def test_dispensary_portal_pays_no_l2_via_old_path(monkeypatch, tmp_path):
    # The old _settle_referrer_reward must NOT credit L2 for a dispensary_portal row;
    # dispensary L2 is owned solely by settle_dispensary_l2 now.
    appmod = _reload(monkeypatch, tmp_path, tier2=True)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
        rf.record_redemption(cx, "DISP", "doc@x.com", "patient@x.com", "INV-1",
                             kind="dispensary_portal")
        appmod._settle_referrer_reward(cx, _order(), "INV-1")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "upline@x.com") == 0   # old path no longer pays dispensary L2
        assert points.balance(cx, "doc@x.com") == 0      # L1 already suppressed
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/t4 python3 -m pytest tests/test_portal_referral_reward.py::test_dispensary_portal_pays_no_l2_via_old_path -v` (mkdir /tmp/t4)
Expected: FAIL — old path credits upline 570.

- [ ] **Step 3: Gate the L2 block on `not l1_suppressed`**

In `app.py` `_settle_referrer_reward`, change the tier-2 condition (currently `if REFERRAL_TIER2_ENABLED:`, ~line 5195):

```python
    if REFERRAL_TIER2_ENABLED and not l1_suppressed:
```

Add a short comment above it:

```python
    # dispensary_portal L2 is settled per-order by dashboard.dispensary_rewards
    # (every reorder, every pay method), so it is excluded here to avoid double-paying
    # the first order. Ambassador (kind='referral') L2 is unchanged.
```

- [ ] **Step 4: Run to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/t4 python3 -m pytest tests/test_portal_referral_reward.py tests/test_two_tier_referral.py tests/test_referrer_reward_spec2b2.py -v`
Expected: PASS (new assertion green; Ambassador L1+L2 and two-tier suites still green).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_portal_referral_reward.py
git commit -m "fix(referrals): retire old L2 path for dispensary_portal (delegated to per-order settle)"
```

---

### Task 5: Wire `settle_dispensary_l2` into both paid transitions

**Files:**
- Modify: `app.py` — `_settle_order_points` (after the `_settle_referrer_reward` call, ~line 5288, card path)
- Modify: `dashboard/orders.py` — `_record_payment_exec` (after `settle_order_points`, ~line 688, alt-pay path)
- Test: `tests/test_dispensary_l2_wiring.py` (create)

**Interfaces:**
- Consumes: `dashboard.dispensary_rewards.settle_dispensary_l2` (Task 3).
- Produces: both paid transitions credit dispensary L2. Card orders settle via `_settle_order_points`; alt-pay via `_record_payment_exec`. Gated on `order source == 'dispensary'`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dispensary_l2_wiring.py`:

```python
import importlib
import sqlite3
import dashboard.practitioner_portal as pp_mod
from dashboard import referrals as rf, points, orders as o


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    monkeypatch.setenv("REFERRAL_TIER2_ENABLED", "true")
    monkeypatch.setenv("REFERRER_REWARD_PCT", "20")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _dispensary_order(email="pat@x.com", ref="INV1", total=7000, shipping=1300):
    return {"source": "dispensary", "practitioner_id": "prac-1", "email": email,
            "total_cents": total, "shipping_cents": shipping, "get_cents": 0,
            "external_ref": ref}


def test_card_path_settle_order_points_credits_dispensary_l2(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.init_tables(cx)
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    appmod._settle_order_points(_dispensary_order(), order_ref="INV1")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "upline@x.com") == 570


def test_altpay_record_payment_credits_dispensary_l2(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        o.init_orders_table(cx)
        rf.init_tables(cx)
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
        oid = o.upsert_order(cx, source="dispensary", external_ref="INV1", email="pat@x.com",
                             total_cents=7000, shipping_cents=1300, practitioner_id="prac-1")
        cx.commit()
        # invoke the alt-pay confirmation exec directly
        from dashboard.orders import _record_payment_exec
        _record_payment_exec({"order_id": oid, "method": "zelle", "amount_cents": 7000},
                             {"cx": cx})
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "upline@x.com") == 570


def test_non_dispensary_order_no_dispensary_l2(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.init_tables(cx)
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    order = _dispensary_order()
    order["source"] = "reorder"   # not dispensary
    appmod._settle_order_points(order, order_ref="INV1")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "upline@x.com") == 0
```

Confirmed shape (`dashboard/orders.py:665-689`): `_record_payment_exec(params, ctx)` reads `cx` from `ctx["cx"]` (or `params["cx"]`), `params["order_id"]`, `params.get("method")`, `params.get("amount_cents")`. The test's dicts above match. A dispensary order ingested at checkout has `status='new'`/`pay_status='unpaid'`, which passes the exec's pre-fulfillment guard.

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/t5 python3 -m pytest tests/test_dispensary_l2_wiring.py -v` (mkdir /tmp/t5)
Expected: FAIL — no dispensary L2 wired yet.

- [ ] **Step 3: Wire the card path**

In `app.py` `_settle_order_points`, after the `_settle_referrer_reward` try/except block (~line 5288-5291), add:

```python
        # Dispensary sales pay L2-only (upline), on every paid order incl. reorders,
        # via the dedicated per-order settler. Gated on source so it never touches
        # non-dispensary orders.
        if (order.get("source") or "") == "dispensary":
            try:
                from dashboard import dispensary_rewards as _dr
                _dr.settle_dispensary_l2(cx, order, order_ref)
            except Exception as _de:
                print(f"[dispensary-l2] card settle skipped: {_de!r}", flush=True)
```

(`order` here is a dict — the existing `_settle_order_points` body already calls `order.get("email")`, `order.get("total_cents")`, etc., and the card path passes `find_order_by_external_ref`'s `SELECT *` dict result, so `order.get("source")` and the new `order.get("practitioner_id")` are both present and safe. No Row-access concern.)

- [ ] **Step 4: Wire the alt-pay path**

In `dashboard/orders.py` `_record_payment_exec` (~line 688), after the `settle_order_points(cx, get_order(cx, oid))` line, add:

```python
    _o = get_order(cx, oid)
    if _o and (_o.get("source") or "") == "dispensary":
        try:
            from dashboard.dispensary_rewards import settle_dispensary_l2
            settle_dispensary_l2(cx, _o, _o.get("external_ref"))
        except Exception as _de:
            print(f"[dispensary-l2] altpay settle skipped: {_de!r}", flush=True)
```

(Verify `get_order` returns a dict-like with `source`/`external_ref`; adjust access if it returns a Row.)

- [ ] **Step 5: Run to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/t5 python3 -m pytest tests/test_dispensary_l2_wiring.py tests/test_referral_settlement.py -v`
Expected: PASS (both paths credit L2; existing settlement suite still green).

- [ ] **Step 6: Commit**

```bash
git add app.py dashboard/orders.py tests/test_dispensary_l2_wiring.py
git commit -m "feat(dispensary): credit upline L2 on card + alt-pay paid transitions"
```

---

## Adjacent gap discovered — NOT in this plan (needs Dr. Glen's decision)

The settlement map surfaced a related gap this plan does NOT close: **alt-pay (Zelle/Wise) dispensary orders currently receive no practitioner Wellness Credit (wallet margin) and write no `dispensary_orders` row** — both happen only on the card settlement path (`begin_checkout_return` client-kind block). So a doctor who takes Zelle gets the upline's L2 (after this plan) but not their own $20/bottle drop-ship credit. Closing that needs `margin_cents` threaded onto the order (like `practitioner_id` here) and the wallet credit + `dispensary_orders` write moved into the shared alt-pay path. Recommend a short follow-up once Dr. Glen confirms the intended alt-pay economics; flagged rather than silently expanded, since it changes the practitioner's own payout, not just L2.

## Post-implementation (controller)

- Full suite once green: `mkdir -p /tmp/final && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/final python3 -m pytest tests/test_orders_pay_method_pid.py tests/test_dispensary_checkout_capture.py tests/test_dispensary_l2.py tests/test_dispensary_l2_wiring.py tests/test_portal_referral_reward.py tests/test_two_tier_referral.py tests/test_referrer_reward_spec2b2.py tests/test_referral_settlement.py tests/test_client_routes.py -q`
- No env flag flip required if `REFERRALS`/`REFERRAL_TIER2_ENABLED`/`REFERRER_REWARD_PCT` are already set on prod — confirm their live values before relying on L2 firing (L2 is gated on `REFERRAL_TIER2_ENABLED`, which the map noted is "flag-dark").

## Self-review notes

- **Spec coverage:** L2 on every dispensary order across pay methods (Tasks 3+5), reorders (per-invoice key, Task 3), no L1 (Task 3 credits only the upline), pay_method capture (Tasks 1+2). All four decisions covered.
- **Double-pay:** Task 4 makes the old and new L2 paths mutually exclusive for dispensary_portal rows — the one real trap the map flagged.
- **Resolution safety:** every L2 resolution goes through `order['practitioner_id']`, never `owner_of_referee(patient)` — Task 3's `test_resolves_from_order_pid_not_patient_referral` locks this in.
- **Type consistency:** `settle_dispensary_l2(cx, order, order_ref)` defined in Task 3, consumed identically in Task 5's two call sites; `upsert_order(..., pay_method=, practitioner_id=)` defined in Task 1, consumed in Task 2.
- **Verified (was open):** `_record_payment_exec` param/ctx shape and that `get_order` / `find_order_by_external_ref` return dicts (both use `.get()` already) — confirmed against source, so `.get("source")`/`.get("practitioner_id")` are safe at both call sites.
