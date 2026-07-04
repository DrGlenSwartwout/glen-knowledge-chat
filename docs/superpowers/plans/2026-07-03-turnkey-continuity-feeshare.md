# Turnkey Continuity Fee-Share (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pay a doctor a cert-scaled recurring share (30%→50%) of every successful Continuous Care ($99/mo) payment made by a patient the doctor enrolled, credited to the doctor's existing wallet.

**Architecture:** A pure `care_share` module holds the rate/amount math and the per-charge credit orchestration. A new nullable `subscriptions.attributed_practitioner_id` (SQLite) records which doctor owns a membership. A new idempotent `wallet.earn_care_share` credit (Supabase, mirroring `earn_dropship_margin`) fires on each successful charge — the enrollment charge (from a new dispensary "Start Continuous Care" flow) and every renewal in `cron_charge_subscriptions()`. Refund reversal ships as a primitive + owner console action (no Stripe refund webhook exists yet).

**Tech Stack:** Python 3, Flask, SQLite (`subscriptions`, `wallet_ledger` is Supabase), Supabase/psycopg2 (`practitioners`, wallet), Stripe, pytest.

## Global Constraints

- Rate: `rate(m) = 0.30 + m × (0.20/12)`, `m = modules_completed` clamped to `[0, 12]` → 0→0.30, 6→0.40, 12→0.50.
- Share base = the **full charged amount** (no product carve-out): $99/mo = 9900 cents; prepay lumps 54600 / 99000.
- `share_cents = round(charge_cents × rate(m))`, integer cents.
- Credit fires **only on a confirmed successful charge**, **per-payment-event** (monthly → each month; prepay → once at purchase-time rate).
- **Idempotent** per `event_ref = "care_share:<sub_id>:<order_count_at_charge>"`.
- Attribution is a **membership-scoped stamp** (`subscriptions.attributed_practitioner_id`), permanent for the membership's life; NULL → no credit. It is **read-only w.r.t. the referral/points ledger** — never write `referral_redemptions` from this feature.
- `wallet.earn_care_share` is **credit-only** (no path to cash) and mirrors `earn_dropship_margin`'s `_apply(..., qbo_invoice_id=<idempotency key>)` seam.
- Money is integer cents everywhere.
- `subscriptions` is SQLite (`cx.execute` with `?` placeholders); `practitioners`/wallet are Supabase (`%s`). An `attributed_practitioner_id` is a Supabase practitioner id string stored in the SQLite row — a cross-store reference (same pattern as `dispensary_orders`).

---

## File Structure

- **Create** `dashboard/care_share.py` — pure `rate()` / `share_cents()`, the `modules_for_practitioner()` resolver, and the `credit_for_charge()` orchestration.
- **Create** `tests/test_care_share.py` — pure math + orchestration (stubbed deps).
- **Modify** `dashboard/subscriptions.py` — add `attributed_practitioner_id` column (migration) + `create_membership` kwarg + expose it on row reads.
- **Modify** `dashboard/wallet.py` — add `earn_care_share` + `reverse_care_share`.
- **Modify** `app.py` — hook `cron_charge_subscriptions()`; add the dispensary "Start Continuous Care" enrollment; add the owner console reversal endpoint.
- **Modify** `static/practitioner-portal.html` or the dispensary template — the "Start Continuous Care" entry point (small).
- **Test files** alongside each (`tests/test_subscriptions_attribution.py`, `tests/test_wallet_care_share.py`, plus route/cron tests).

---

### Task 1: Pure `care_share` rate + amount math

**Files:**
- Create: `dashboard/care_share.py`
- Test: `tests/test_care_share.py`

**Interfaces:**
- Produces: `rate(modules_completed: int) -> float`; `share_cents(charge_cents: int, modules_completed: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_care_share.py
from dashboard import care_share as cs


def test_rate_tiers():
    assert cs.rate(0) == 0.30
    assert abs(cs.rate(6) - 0.40) < 1e-9
    assert cs.rate(12) == 0.50


def test_rate_clamps():
    assert cs.rate(-5) == 0.30
    assert cs.rate(99) == 0.50


def test_share_cents_tiers():
    assert cs.share_cents(9900, 0) == 2970
    assert cs.share_cents(9900, 6) == 3960
    assert cs.share_cents(9900, 12) == 4950


def test_share_cents_prepay_lump():
    # 12-month prepay at full cert: 50% of $990.00
    assert cs.share_cents(99000, 12) == 49500


def test_share_cents_rounds_half():
    # 40% of 9901 = 3960.4 -> 3960
    assert cs.share_cents(9901, 6) == 3960
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_care_share.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.care_share`)

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/care_share.py
"""Cert-scaled recurring fee-share for turnkey continuity (Continuous Care).

Pure math (rate/share) + the per-charge credit orchestration. A doctor who
enrolled a patient earns rate(modules_completed) of every successful membership
charge, credited to their wallet, for as long as the patient stays.
"""

_BASE_RATE = 0.30
_MAX_RATE = 0.50
_MAX_MODULES = 12


def rate(modules_completed):
    """Fee-share fraction, linear in completed cert modules: 0.30 at 0 -> 0.50 at 12."""
    m = max(0, min(_MAX_MODULES, int(modules_completed or 0)))
    return _BASE_RATE + m * ((_MAX_RATE - _BASE_RATE) / _MAX_MODULES)


def share_cents(charge_cents, modules_completed):
    """Doctor's share of one charge, in integer cents (banker-free round-half-up-ish)."""
    return int(round(max(0, int(charge_cents or 0)) * rate(modules_completed)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_care_share.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/care_share.py tests/test_care_share.py
git commit -m "feat(care-share): pure rate + share_cents math"
```

---

### Task 2: `subscriptions.attributed_practitioner_id` column + `create_membership` kwarg

**Files:**
- Modify: `dashboard/subscriptions.py` (migration list near lines 276–300; `create_membership` at ~304)
- Test: `tests/test_subscriptions_attribution.py`

**Interfaces:**
- Consumes: existing `create_membership(cx, *, email, stripe_customer_id, stripe_payment_method_id, amount_cents, next_charge_date, cadence_months=1, term_charges_total=None, initial_order_count=0)`.
- Produces: `create_membership(..., attributed_practitioner_id=None)` persists the id; subscription row dicts include `attributed_practitioner_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subscriptions_attribution.py
import sqlite3
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_tables(cx)   # use the module's real table-init entrypoint
    return cx


def test_create_membership_stamps_practitioner():
    cx = _cx()
    sid = subs.create_membership(
        cx, email="p@x.com", stripe_customer_id="cus_1",
        stripe_payment_method_id="pm_1", amount_cents=9900,
        next_charge_date="2026-08-01", attributed_practitioner_id="prac-42")
    row = cx.execute("SELECT attributed_practitioner_id FROM subscriptions WHERE id=?",
                     (sid,)).fetchone()
    assert row["attributed_practitioner_id"] == "prac-42"


def test_create_membership_defaults_null():
    cx = _cx()
    sid = subs.create_membership(
        cx, email="p@x.com", stripe_customer_id="cus_1",
        stripe_payment_method_id="pm_1", amount_cents=9900,
        next_charge_date="2026-08-01")
    row = cx.execute("SELECT attributed_practitioner_id FROM subscriptions WHERE id=?",
                     (sid,)).fetchone()
    assert row["attributed_practitioner_id"] is None
```

Note: confirm the real table-init function name in `subscriptions.py` (it may be `init_tables`/`_ensure_tables`/`init_table`); use whichever the module exposes and other subscription tests use.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_subscriptions_attribution.py -q`
Expected: FAIL (`TypeError: create_membership() got an unexpected keyword argument 'attributed_practitioner_id'`)

- [ ] **Step 3: Add the migration**

In the subscriptions migration list (the block running `ALTER TABLE subscriptions ADD COLUMN ...` around lines 276–300, each guarded so re-runs are safe), add:

```python
        "ALTER TABLE subscriptions ADD COLUMN attributed_practitioner_id TEXT",
```

(Match the surrounding guard style — the existing entries wrap each ALTER in a try/except or an "already exists" swallow. Follow the exact pattern used for `term_charges_total`/`kind`.)

- [ ] **Step 4: Add the kwarg to `create_membership`**

Change the signature and INSERT so the new column is written:

```python
def create_membership(cx, *, email, stripe_customer_id, stripe_payment_method_id,
                      amount_cents, next_charge_date, cadence_months=1,
                      term_charges_total=None, initial_order_count=0,
                      attributed_practitioner_id=None) -> int:
    now = _now_iso()
    cur = cx.execute(
        """INSERT INTO subscriptions
               (email, stripe_customer_id, stripe_payment_method_id, items_json,
                cadence_months, status, order_count, next_charge_date, ship_address_json,
                skip_next, created_at, updated_at, kind, amount_cents, term_charges_total,
                attributed_practitioner_id)
           VALUES (?,?,?,?,?,'active',?,?,?,0,?,?, 'membership', ?, ?, ?)""",
        (email, stripe_customer_id, stripe_payment_method_id, "[]",
         int(cadence_months), int(initial_order_count), next_charge_date, "{}", now, now,
         int(amount_cents), (int(term_charges_total) if term_charges_total is not None else None),
         (str(attributed_practitioner_id) if attributed_practitioner_id else None)),
    )
    cx.commit()
    try:
        _customers.find_or_create_by_email(cx, email=email)
    except Exception:
        pass
    return cur.lastrowid
```

(Leave the docstring; append one line noting the new kwarg.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_subscriptions_attribution.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add dashboard/subscriptions.py tests/test_subscriptions_attribution.py
git commit -m "feat(care-share): subscriptions.attributed_practitioner_id + create_membership stamp"
```

---

### Task 3: `wallet.earn_care_share` + `reverse_care_share`

**Files:**
- Modify: `dashboard/wallet.py` (add functions near `earn_dropship_margin`, ~line 189)
- Test: `tests/test_wallet_care_share.py`

**Interfaces:**
- Consumes: the existing `_apply(practitioner_id, kind, fn, *, qbo_invoice_id=None, note=None)` seam (idempotent per `qbo_invoice_id`; the ledger dedupes on it).
- Produces: `earn_care_share(practitioner_id, share_cents, *, event_ref) -> int` (credit, idempotent per `event_ref`); `reverse_care_share(practitioner_id, share_cents, *, event_ref) -> int` (debit; no-op if the credit for `event_ref` is absent).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wallet_care_share.py
from dashboard import wallet


def test_earn_care_share_credits_once(monkeypatch):
    calls = []
    def fake_apply(pid, kind, fn, *, qbo_invoice_id=None, note=None):
        calls.append((pid, kind, qbo_invoice_id, fn(0)))
        return fn(0)
    monkeypatch.setattr(wallet, "_apply", fake_apply)
    amt = wallet.earn_care_share("prac-42", 4950, event_ref="care_share:7:3")
    assert amt == 4950
    assert calls == [("prac-42", "earn_care_share", "care_share:7:3", 4950)]


def test_earn_care_share_clamps_negative(monkeypatch):
    monkeypatch.setattr(wallet, "_apply",
                        lambda pid, kind, fn, *, qbo_invoice_id=None, note=None: fn(0))
    assert wallet.earn_care_share("prac-42", -10, event_ref="care_share:7:3") == 0


def test_reverse_care_share_debits(monkeypatch):
    seen = {}
    def fake_apply(pid, kind, fn, *, qbo_invoice_id=None, note=None):
        seen["kind"] = kind; seen["ref"] = qbo_invoice_id; return fn(1000)
    monkeypatch.setattr(wallet, "_apply", fake_apply)
    wallet.reverse_care_share("prac-42", 4950, event_ref="care_share:7:3")
    assert seen["kind"] == "reverse_care_share"
    assert seen["ref"] == "reverse:care_share:7:3"
```

Note: match how the real `_apply` dedupes and how sibling wallet tests (`tests/test_wallet*.py`) patch the `_cursor()` seam; the monkeypatched `_apply` above keeps this unit-level. Verify `_apply`'s real signature before finalizing the fakes.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wallet_care_share.py -q`
Expected: FAIL (`AttributeError: module 'dashboard.wallet' has no attribute 'earn_care_share'`)

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/wallet.py — near earn_dropship_margin

def earn_care_share(practitioner_id, share_cents, *, event_ref) -> int:
    """Credit a doctor's cert-scaled share of a Continuous Care charge.

    Credit-only (no path to cash). Idempotent per ``event_ref`` — a second call
    with the same event_ref is a silent no-op, so cron retries are safe.
    """
    amt = max(0, int(share_cents))
    return _apply(practitioner_id, "earn_care_share", lambda _bal: amt,
                  qbo_invoice_id=event_ref, note="care_share")


def reverse_care_share(practitioner_id, share_cents, *, event_ref) -> int:
    """Debit a previously-credited care-share (e.g. on a manual refund). Keyed to
    a distinct ``reverse:`` idempotency ref so it applies at most once."""
    amt = max(0, int(share_cents))
    return _apply(practitioner_id, "reverse_care_share", lambda _bal: -amt,
                  qbo_invoice_id=f"reverse:{event_ref}", note="care_share_reversal")
```

Note: confirm `_apply` accepts a negative delta from `fn` for the debit; if the real `_apply` clamps to non-negative, use the module's actual debit path (grep for how `spend`/redeem debits the ledger) and mirror it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wallet_care_share.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/wallet.py tests/test_wallet_care_share.py
git commit -m "feat(care-share): wallet.earn_care_share + reverse_care_share (idempotent)"
```

---

### Task 4: `care_share` orchestration — resolver + `credit_for_charge`

**Files:**
- Modify: `dashboard/care_share.py`
- Test: `tests/test_care_share.py`

**Interfaces:**
- Consumes: `rate`/`share_cents` (Task 1); `wallet.earn_care_share` (Task 3).
- Produces:
  - `modules_for_practitioner(pid) -> int | None` (reads `SELECT modules_completed FROM practitioners WHERE id=%s`; None if not a practitioner).
  - `credit_for_charge(sub, *, charge_cents, earn=None, resolve_modules=None) -> int` — given a subscription row dict, if `sub["attributed_practitioner_id"]` is set and resolves to a practitioner, credit `share_cents(charge_cents, m)` via `earn` (defaults to `wallet.earn_care_share`) with `event_ref="care_share:<sub_id>:<order_count>"`. Returns cents credited (0 if no attribution / not a practitioner). `resolve_modules`/`earn` are injectable for tests.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_care_share.py
def test_credit_for_charge_credits_attributed():
    seen = {}
    def earn(pid, cents, *, event_ref):
        seen.update(pid=pid, cents=cents, event_ref=event_ref); return cents
    sub = {"id": 7, "order_count": 3, "attributed_practitioner_id": "prac-42"}
    out = cs.credit_for_charge(sub, charge_cents=9900,
                               earn=earn, resolve_modules=lambda pid: 12)
    assert out == 4950
    assert seen == {"pid": "prac-42", "cents": 4950, "event_ref": "care_share:7:3"}


def test_credit_for_charge_no_attribution():
    called = []
    sub = {"id": 7, "order_count": 3, "attributed_practitioner_id": None}
    out = cs.credit_for_charge(sub, charge_cents=9900,
                               earn=lambda *a, **k: called.append(1),
                               resolve_modules=lambda pid: 12)
    assert out == 0 and called == []


def test_credit_for_charge_owner_not_a_practitioner():
    sub = {"id": 7, "order_count": 3, "attributed_practitioner_id": "someone"}
    out = cs.credit_for_charge(sub, charge_cents=9900,
                               earn=lambda *a, **k: 1/0,   # must not be called
                               resolve_modules=lambda pid: None)
    assert out == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_care_share.py -q`
Expected: FAIL (`AttributeError: ... 'credit_for_charge'`)

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/care_share.py

def modules_for_practitioner(pid):
    """Live modules_completed for a practitioner id, or None if not a practitioner."""
    if not pid:
        return None
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute("SELECT modules_completed FROM practitioners WHERE id=%s", (str(pid),))
        row = cur.fetchone()
    if not row:
        return None
    return int(row["modules_completed"] or 0)


def credit_for_charge(sub, *, charge_cents, earn=None, resolve_modules=None):
    """Credit the attributed doctor's cert-scaled share of one successful charge.

    Idempotent per event_ref = care_share:<sub_id>:<order_count>. Returns cents
    credited (0 when unattributed or the owner is not a practitioner).
    """
    pid = (sub or {}).get("attributed_practitioner_id")
    if not pid:
        return 0
    resolve_modules = resolve_modules or modules_for_practitioner
    m = resolve_modules(pid)
    if m is None:
        return 0
    cents = share_cents(charge_cents, m)
    if cents <= 0:
        return 0
    if earn is None:
        from dashboard import wallet as _wallet
        earn = _wallet.earn_care_share
    event_ref = f"care_share:{sub['id']}:{sub['order_count']}"
    earn(str(pid), cents, event_ref=event_ref)
    return cents
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_care_share.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/care_share.py tests/test_care_share.py
git commit -m "feat(care-share): modules resolver + credit_for_charge orchestration"
```

---

### Task 5: Hook the fee-share into the renewal cron

**Files:**
- Modify: `app.py` (`cron_charge_subscriptions()`, ~23509; the successful-charge points at ~23639 and ~23741 that call `_subs.advance_after_charge(cx, sid)`)
- Test: `tests/test_care_share_cron.py`

**Interfaces:**
- Consumes: `care_share.credit_for_charge` (Task 4); the subscription row already in scope in the charge loop.

**Important:** the credit must fire **after** the charge is confirmed successful and after `advance_after_charge` has incremented `order_count` (so `event_ref` uses the just-completed charge's sequence). Read the sub's `attributed_practitioner_id` and `order_count` from the row available in the loop (re-`SELECT` if the loop's `sub` dict predates the advance, so `order_count` reflects the completed charge).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_care_share_cron.py
# Follow the existing cron test pattern (tests/test_subscriptions*.py / any cron_charge test)
# for seeding a due membership + stubbing Stripe. Assert that on a successful charge of an
# attributed membership, care_share.credit_for_charge is invoked with the sub and charge_cents,
# and is NOT invoked for an unattributed membership. Patch care_share.credit_for_charge to record calls.
```

Because the cron is large and Stripe-bound, write this as a focused test that patches the Stripe charge call to "succeed" and patches `app.<care_share ref>.credit_for_charge` (or `dashboard.care_share.credit_for_charge`) to a recorder, seeds one attributed + one unattributed due membership, runs the cron, and asserts the recorder saw exactly the attributed one with `charge_cents == amount_cents`. Mirror the seeding/stubs from the nearest existing `cron_charge_subscriptions` test.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_care_share_cron.py -q`
Expected: FAIL (credit_for_charge never called — hook absent)

- [ ] **Step 3: Add the hook**

In `cron_charge_subscriptions()`, at each place a membership charge is confirmed successful and `_subs.advance_after_charge(cx, sid)` runs, add (immediately after the advance):

```python
                        # Turnkey continuity fee-share: credit the enrolling doctor.
                        try:
                            fresh = _subs.get_by_id(cx, sid)   # row AFTER advance (order_count updated)
                            if fresh and (fresh.get("kind") == "membership"):
                                from dashboard import care_share as _cshare
                                _cshare.credit_for_charge(
                                    fresh, charge_cents=int(fresh.get("amount_cents") or 0))
                        except Exception as e:
                            print(f"[care-share] credit failed sid={sid}: {e!r}", flush=True)
```

Notes: use the module's real single-row read (`get_by_id`/`get`/`by_id` — confirm the name). Guard with try/except so a fee-share failure never breaks billing. Apply at BOTH advance sites (~23639, ~23741) if they are distinct successful-charge paths; if they share a helper, hook the helper once.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_care_share_cron.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_care_share_cron.py
git commit -m "feat(care-share): credit the enrolling doctor on each successful renewal charge"
```

---

### Task 6: Dispensary "Start Continuous Care" enrollment (attributed) + enrollment-charge credit

**Files:**
- Modify: `app.py` (near the dispensary routes; reuse the existing membership-enrollment/Stripe path)
- Modify: the dispensary landing template (the page served by `/dispensary/<code>`) — add the "Start Continuous Care" entry point
- Test: `tests/test_care_share_enroll.py`

**Interfaces:**
- Consumes: `_pp.practitioner_id_by_dispensary_code(code)`; `subscriptions.create_membership(..., attributed_practitioner_id=pid)`; `care_share.credit_for_charge`.

**Approach:** find the existing Continuous Care membership enrollment flow (grep `create_membership(` callers + `MONTHLY_ANCHOR_CENTS`/`portal_offers`) and reuse it. Add a dispensary-scoped enrollment endpoint (e.g. `POST /dispensary/<code>/continuous-care`) that: resolves `pid` from the code, runs the same consent gate as the dispensary checkout, sets up the Stripe customer/payment method + first charge exactly as the existing membership enrollment does, calls `create_membership(..., attributed_practitioner_id=pid, initial_order_count=1)` (month 1 charged at checkout), and then fires the enrollment-charge credit:

```python
        sub_row = _subs.get_by_id(cx, sid)
        from dashboard import care_share as _cshare
        _cshare.credit_for_charge(sub_row, charge_cents=int(sub_row.get("amount_cents") or 0))
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_care_share_enroll.py
# Follow tests/test_practitioner_pricing_routes.py / dispensary route tests for app import + stubs.
# Stub Stripe + practitioner_id_by_dispensary_code; POST the enrollment endpoint for a code that
# resolves to pid "prac-42"; assert (a) a membership row exists with attributed_practitioner_id="prac-42"
# and kind="membership", and (b) care_share.credit_for_charge was invoked once with that sub + charge_cents=9900.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_care_share_enroll.py -q`
Expected: FAIL (endpoint 404 / attribution absent)

- [ ] **Step 3: Implement the endpoint + template entry point**

Add the endpoint per the Approach above (reusing the existing membership-enrollment helper; do NOT hand-roll Stripe — call the same function the current $99/mo enrollment uses). Add a "Start Continuous Care — $99/mo" button/section to the dispensary landing page that posts to it, gated behind the same consent/identify step as the dispensary product checkout.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_care_share_enroll.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_care_share_enroll.py <dispensary template>
git commit -m "feat(care-share): dispensary Start Continuous Care enrollment (attributed) + enrollment credit"
```

---

### Task 7: Owner console reversal action

**Files:**
- Modify: `app.py` (add an owner-guarded endpoint near the other `/api/console/*` routes)
- Test: `tests/test_care_share_reversal.py`

**Interfaces:**
- Consumes: `wallet.reverse_care_share`; the subscription row (for `attributed_practitioner_id`, `id`, and the charge `order_count` to rebuild `event_ref`).
- Produces: `POST /api/console/care-share/reverse` body `{sub_id, order_count}` → reverses the care-share credit for that charge event.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_care_share_reversal.py
# Follow an existing owner-console route test (X-Console-Key guard). Seed an attributed membership,
# POST {sub_id, order_count}; assert wallet.reverse_care_share is called with the attributed pid,
# share_cents(amount_cents, modules) [patch resolve], and event_ref="care_share:<sub_id>:<order_count>".
# Assert a missing/invalid console key -> 401/403, and an unattributed sub -> no reversal + a clear message.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_care_share_reversal.py -q`
Expected: FAIL (route 404)

- [ ] **Step 3: Implement the endpoint**

```python
@app.route("/api/console/care-share/reverse", methods=["POST"])
def api_console_care_share_reverse():
    # owner guard: match the existing X-Console-Key pattern used by other /api/console/* routes
    if not _console_authorized(request):     # use the real guard helper name
        return jsonify({"ok": False, "error": "unauthorized"}), 403
    body = request.get_json(silent=True) or {}
    sid = body.get("sub_id"); oc = body.get("order_count")
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        sub = _subs.get_by_id(cx, sid)
    if not sub or not sub.get("attributed_practitioner_id"):
        return jsonify({"ok": False, "error": "no attributed care-share for that subscription"}), 404
    from dashboard import care_share as _cshare
    pid = sub["attributed_practitioner_id"]
    m = _cshare.modules_for_practitioner(pid)
    cents = _cshare.share_cents(int(sub.get("amount_cents") or 0), m or 0)
    from dashboard import wallet as _wallet
    _wallet.reverse_care_share(str(pid), cents,
                               event_ref=f"care_share:{sub['id']}:{int(oc)}")
    return jsonify({"ok": True, "reversed_cents": cents})
```

Use the real owner-guard helper (grep the other `/api/console/` routes for the exact `X-Console-Key` check). Match the module's real `get_by_id` name.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_care_share_reversal.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_care_share_reversal.py
git commit -m "feat(care-share): owner console reversal action for a care-share credit"
```

---

## Self-Review

**Spec coverage:**
- Economic model (rate/share, full base, per-payment-event) → Task 1 (+ used in 4/5/6). ✓
- Attributed enrollment (stamp, permanent, NULL default) → Task 2 (column) + Task 6 (dispensary enrollment sets it). ✓
- Recurring fee-share on each charge, idempotent per event_ref → Task 3 (wallet) + Task 4 (orchestration) + Task 5 (renewal) + Task 6 (enrollment charge). ✓
- Rate read live at charge → Task 4 `modules_for_practitioner` called per charge. ✓
- Owner-not-a-practitioner / no-attribution / failed-charge edges → Task 4 tests + Task 5 (inside success branch). ✓
- Refund reversal primitive + console action (no auto webhook) → Task 3 (`reverse_care_share`) + Task 7 (console). ✓
- Doctor visibility via existing wallet surface → no new code (credits land in `wallet_ledger`/Ambassador tab); called out, no task needed. ✓
- Alignment with #560 (read referral owner as fallback, never write points ledger) → the stamp is set from the dispensary pid in Task 6; the referral-owner *fallback* seeding is noted in the spec but is not required for the dispensary path (the primary source). If Glen wants the referral-owner fallback wired for non-dispensary arrivals, that is an added task — flagged below.

**Placeholder scan:** Tasks 5–7 intentionally reference sibling test patterns / real helper names to confirm (`get_by_id`, `_console_authorized`, the membership-enrollment helper, table-init name) rather than inventing them — the implementer must grep the real name in the named file. All pure/data/wallet task code (1–4) is complete and self-contained. No "TODO/handle edge cases" placeholders.

**Type consistency:** `rate`/`share_cents`/`modules_for_practitioner`/`credit_for_charge` signatures are consistent across Tasks 1/4/5/6/7. `event_ref` format `care_share:<sub_id>:<order_count>` is identical in Tasks 3/4/7. `earn_care_share(pid, cents, *, event_ref)` / `reverse_care_share(pid, cents, *, event_ref)` consistent across Tasks 3/4/7.

## Notes / open confirmations for review

- **Referral-owner fallback (A):** the dispensary enrollment (Task 6) sets attribution explicitly from the code's pid — the common path. The spec's "fallback = `owner_of_referee`" for patients who arrive via a referral link but *not* the dispensary path is **not** a task here (v1 = dispensary enrollment). If enrollment should also attribute via the referral graph for non-dispensary arrivals, add a task; flag before executing.
- **Refund reversal is manual** (Task 7 console action) — no Stripe `charge.refunded` webhook exists. Automatic reversal is a fast-follow.
- **`_apply` debit path (Task 3):** confirm the real `_apply` accepts a negative delta for `reverse_care_share`; if it clamps non-negative, mirror the module's actual spend/redeem debit path instead.
