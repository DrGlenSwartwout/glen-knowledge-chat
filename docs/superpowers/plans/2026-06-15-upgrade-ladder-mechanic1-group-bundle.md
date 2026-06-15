# Upgrade Ladder — Mechanic 1: Program-Bundled Live Group (auto-continue) — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** When a customer buys a remedy program and opts in, they get **N months of live-group coaching free (1 per program month, capped at 3)**, after which a **$99/mo membership auto-continues on the Stripe vault unless cancelled.** Ships dark behind `GROUP_BUNDLE_ENABLED`.

**Architecture:** Reuse the Subscribe-and-Grow rail. The `subscriptions` table gains a `kind` ('product' default | 'membership') + `amount_cents`; a `create_membership()` helper makes a flat-amount sub with a **delayed first charge** (= end of the free window). The existing charge cron gets a `kind=='membership'` branch that charges the flat amount off-session and writes a one-line invoice. At the program-order Stripe checkout, an explicit **opt-in** sets `save_card=True` + grant metadata; the checkout-return creates/extends the membership trial. Pure logic (included-months, amount) lives in a new `dashboard/group_bundle.py`.

**Tech Stack:** Python 3.11, Flask, sqlite (`subscriptions`), Stripe vault (`stripe_pay.charge_off_session`), QBO (`qb.create_invoice`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-15-upgrade-incentive-ladder-design.md` (Mechanic 1). Rail = Stripe vault (decided; QBO Payments not yet approved). Compliance: explicit trial opt-in + reminder + one-click cancel (negative-option rules).

**Reuse:** `dashboard/subscriptions.py` (`create`, `list_due`, `advance_after_charge`, `get_manageable_by_email`, `list_heads_up_due`, `set_status`, `migrate_add_failed_count` pattern), the charge cron `cron_charge_subscriptions` (app.py ~13133), the subscribe checkout + `save_card=True` + checkout-return `kind=='subscribe'` vault pattern (app.py ~6803/~3031), `stripe_pay.charge_off_session`, `qb.create_invoice`, `_ingest_order`, `_send_subscription_email`, the engine's `volume_months` (from `_price_cart`/preview) for program months.

**Test invocation:** pure modules → `~/.venvs/deploy-chat311/bin/python -m pytest <path> -q`. App/cron tests → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -q` (from inside the worktree; ignore the 2 known pre-existing failures).

---

### Task 1: `group_bundle.py` — included-months + constants (pure)

**Files:** Create `dashboard/group_bundle.py`; Test `tests/test_group_bundle.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_group_bundle.py
from dashboard import group_bundle as gb


def test_included_group_months_one_per_program_month_capped_at_3():
    assert gb.included_group_months(0) == 0      # Biofield alone / no program
    assert gb.included_group_months(1) == 1
    assert gb.included_group_months(2) == 2
    assert gb.included_group_months(3) == 3
    assert gb.included_group_months(6) == 3       # capped
    assert gb.included_group_months(12) == 3


def test_included_group_months_handles_junk():
    assert gb.included_group_months(None) == 0
    assert gb.included_group_months(-4) == 0
    assert gb.included_group_months("2") == 2


def test_membership_amount_default():
    assert gb.MEMBERSHIP_AMOUNT_CENTS == 9900       # $99 founders
    assert gb.MEMBERSHIP_CADENCE_MONTHS == 1
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement**

```python
# dashboard/group_bundle.py
"""Pure logic for the program-bundled live-group offer (Mechanic 1).
1 free live-group month per program month purchased, capped at 3 (the max
recommended program length); 0 for a Biofield-only / no-program purchase."""

MEMBERSHIP_AMOUNT_CENTS = 9900     # $99/mo founders rate (live group)
MEMBERSHIP_CADENCE_MONTHS = 1
MAX_INCLUDED_MONTHS = 3


def included_group_months(program_months) -> int:
    try:
        m = int(program_months or 0)
    except (TypeError, ValueError):
        return 0
    if m <= 0:
        return 0
    return min(m, MAX_INCLUDED_MONTHS)
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(group-bundle): included_group_months pure helper`

---

### Task 2: `subscriptions` — membership kind + `create_membership`

**Files:** Modify `dashboard/subscriptions.py`; Test `tests/test_subscriptions_membership.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_subscriptions_membership.py
import sqlite3
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    return cx


def test_create_membership_sets_kind_and_amount():
    cx = _cx()
    sid = subs.create_membership(cx, email="p@x.com", stripe_customer_id="cus_1",
                                 stripe_payment_method_id="pm_1", amount_cents=9900,
                                 next_charge_date="2026-09-15")
    row = subs.get(cx, sid)
    assert row["kind"] == "membership"
    assert row["amount_cents"] == 9900
    assert row["cadence_months"] == 1
    assert row["status"] == "active"
    assert row["next_charge_date"] == "2026-09-15"


def test_product_subs_default_kind_product():
    cx = _cx()
    sid = subs.create(cx, email="p@x.com", stripe_customer_id="c", stripe_payment_method_id="pm",
                      items=[{"slug": "a", "qty": 1}], cadence_months=1, ship_address={},
                      next_charge_date="2026-08-01")
    assert subs.get(cx, sid)["kind"] == "product"


def test_list_due_returns_membership(monkeypatch):
    cx = _cx()
    subs.create_membership(cx, email="p@x.com", stripe_customer_id="c",
                           stripe_payment_method_id="pm", amount_cents=9900,
                           next_charge_date="2026-01-01")
    due = subs.list_due(cx, as_of="2026-02-01")
    assert len(due) == 1 and due[0]["kind"] == "membership" and due[0]["amount_cents"] == 9900


def test_active_membership_for_email():
    cx = _cx()
    subs.create_membership(cx, email="p@x.com", stripe_customer_id="c",
                           stripe_payment_method_id="pm", amount_cents=9900,
                           next_charge_date="2026-09-15")
    ms = subs.active_memberships_by_email(cx, "p@x.com")
    assert len(ms) == 1 and ms[0]["kind"] == "membership"
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** in `dashboard/subscriptions.py`:

Add an idempotent migration (mirror `migrate_add_failed_count`):
```python
def migrate_add_membership_columns(cx) -> None:
    """Add kind + amount_cents columns if missing. Safe on every startup."""
    for ddl in (
        "ALTER TABLE subscriptions ADD COLUMN kind TEXT NOT NULL DEFAULT 'product'",
        "ALTER TABLE subscriptions ADD COLUMN amount_cents INTEGER NOT NULL DEFAULT 0",
    ):
        try:
            cx.execute(ddl); cx.commit()
        except Exception:
            pass
```

`_row_to_dict` already returns all columns, so `kind`/`amount_cents` surface automatically once the columns exist.

Add the creator + lookup:
```python
def create_membership(cx, *, email, stripe_customer_id, stripe_payment_method_id,
                      amount_cents, next_charge_date, cadence_months=1) -> int:
    """Insert an active flat-amount membership subscription (no product items).
    The first charge lands on next_charge_date (= end of any free window)."""
    now = _now_iso()
    cur = cx.execute(
        """INSERT INTO subscriptions
               (email, stripe_customer_id, stripe_payment_method_id, items_json,
                cadence_months, status, order_count, next_charge_date, ship_address_json,
                skip_next, created_at, updated_at, kind, amount_cents)
           VALUES (?,?,?,?,?,'active',0,?,?,0,?,?, 'membership', ?)""",
        (email, stripe_customer_id, stripe_payment_method_id, "[]",
         int(cadence_months), next_charge_date, "{}", now, now, int(amount_cents)),
    )
    cx.commit()
    return cur.lastrowid


def active_memberships_by_email(cx, email) -> list:
    rows = cx.execute(
        "SELECT * FROM subscriptions WHERE email=? AND status='active' AND kind='membership'"
        " ORDER BY id", (email,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]
```

> NOTE: callers must run BOTH `init_subscriptions_table` and `migrate_add_membership_columns` (the cron + app startup already call init + `migrate_add_failed_count`; add the membership migration alongside).

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(group-bundle): membership kind + create_membership in subscriptions`

---

### Task 3: charge cron — membership branch

**Files:** Modify `app.py` (`cron_charge_subscriptions`, the `for sub in due:` loop ~13217); ensure the membership migration runs where `migrate_add_failed_count` does. Test `tests/test_membership_charge_cron.py`

- [ ] **Step 1: Failing test** — seed an in-memory-style sqlite at `appmod.LOG_DB` (tmp) with one due `kind='membership'` sub (amount 9900), monkeypatch `appmod.stripe_pay.charge_off_session` → `{"status":"succeeded","id":"pi_x"}`, `appmod.qb.find_or_create_customer`/`create_invoice`, and `_ingest_order`. Set `SUBSCRIPTIONS_ENABLED`. POST `/api/cron/charge-subscriptions` (X-Cron-Secret). Assert: `charge_off_session` called with amount 9900; an order recorded with `source="membership"`; `advance_after_charge` moved `next_charge_date` forward one month; response `charged>=1`. Plus a `dry_run=1` test that charges nothing. Mirror the harness in the existing subscription-cron test (search `tests/` for `charge-subscriptions`).

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — at the top of the `for sub in due:` body, branch before the product pricing:

```python
                if sub.get("kind") == "membership":
                    amount_cents = int(sub.get("amount_cents") or 0)
                    if amount_cents <= 0:
                        continue
                    if dry_run:
                        print(f"[sub-cron] DRY membership charge sub={sid} "
                              f"email={sub['email']} amount={amount_cents}", flush=True)
                        charged += 1
                        continue
                    res = stripe_pay.charge_off_session(
                        sub["stripe_customer_id"], sub["stripe_payment_method_id"],
                        amount_cents, description="Remedy Match live group coaching",
                        metadata={"sub": str(sid), "kind": "membership"})
                    if res.get("status") == "succeeded":
                        try:
                            cust = qb.find_or_create_customer(sub["email"], "")
                            inv = qb.create_invoice(
                                cust,
                                [{"name": "Live Group Coaching", "amount": amount_cents / 100.0,
                                  "qty": 1, "description": "Live Group Coaching (monthly)"}],
                                allow_online_pay=False, email_to=sub["email"])
                            inv_id = inv.get("Id", "")
                        except Exception as qe:
                            print(f"[sub-cron] membership QBO sub={sid}: {qe!r}", flush=True)
                            inv_id = ""
                        _ingest_order(source="membership", external_ref=res.get("id") or inv_id,
                                      email=sub["email"], items=[], total_cents=amount_cents,
                                      address={}, channel="retail")
                        _subs.advance_after_charge(cx, sid)
                        _subs.reset_failed_count(cx, sid)
                        updated = _subs.get(cx, sid)
                        _send_subscription_email(sub["email"], "receipt", {
                            "total_cents": amount_cents, "invoice_id": inv_id,
                            "next_charge_date": updated["next_charge_date"] if updated else ""})
                        charged += 1
                    else:
                        _subs.bump_failed_count(cx, sid)
                        failed += 1
                    continue
```

(Place it as the first statement inside the `try:` of the `for sub in due:` loop so product subs fall through to the existing path. Confirm `_ingest_order` accepts `source="membership"`; it already takes arbitrary `source` strings.)

- [ ] **Step 4: Run → pass** (+ re-run the existing subscription-cron test to confirm product charges still work).
- [ ] **Step 5: Commit** — `feat(group-bundle): charge cron membership branch`

---

### Task 4: program-checkout opt-in → vault + grant the trial

**Files:** Modify `app.py` (the program/funnel + reorder Stripe checkout that builds product orders; the `checkout-return` handler). Test `tests/test_group_bundle_grant.py`

- [ ] **Step 1: Failing test** — two parts:
  - **Grant on return:** simulate a paid `checkout-return` whose metadata carries `grant_group_months="3"`, `email`, and a vaulted PaymentIntent (reuse the `kind=='subscribe'` vault test harness — it already stubs the session/PI → customer + payment_method). With `GROUP_BUNDLE_ENABLED=1`, assert a `kind='membership'` subscription is created for the email with `amount_cents=9900`, `cadence_months=1`, and `next_charge_date == today + 3 months` (use `subscriptions.add_months`). Idempotent: a second return with the same invoice/PI does not create a duplicate. With the flag unset → no membership created.
  - **Window stacking:** if an active membership already exists, a new grant **extends** its `next_charge_date` by the granted months instead of creating a second membership.

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.**
  - **Checkout (opt-in):** in the program/funnel + reorder card-checkout builders, when `GROUP_BUNDLE_ENABLED` and the request opted in (`body.get("group_bundle")` truthy — wired to a checkout checkbox in the page) and the order's program months ≥ 1: set `save_card=True` on the Stripe session and add metadata `grant_group_months = str(group_bundle.included_group_months(volume_months))`, plus `email`. (Program months = the engine's `volume_months` for the cart; reuse what `_price_cart`/preview computes.)
  - **Return:** add a branch in `checkout-return` (best-effort, flag-gated) after the existing handlers:
```python
                        if os.environ.get("GROUP_BUNDLE_ENABLED") and int(md.get("grant_group_months") or 0) > 0:
                            try:
                                from dashboard import subscriptions as _subs, group_bundle as _gb
                                n = int(md.get("grant_group_months"))
                                g_email = (md.get("email") or "").strip().lower()
                                start = _subs.add_months(_date.today().isoformat(), n)  # first charge after the free window
                                with sqlite3.connect(LOG_DB) as _gcx:
                                    _gcx.row_factory = sqlite3.Row
                                    _subs.init_subscriptions_table(_gcx)
                                    _subs.migrate_add_membership_columns(_gcx)
                                    existing = _subs.active_memberships_by_email(_gcx, g_email)
                                    if existing:
                                        # window-stack: push the first charge out by n months
                                        cur = existing[0]
                                        _subs.set_next_charge_date(_gcx, cur["id"],
                                            _subs.add_months(cur["next_charge_date"], n))
                                    elif pi_id and g_email and not _subs.active_memberships_by_email(_gcx, g_email):
                                        _subs.create_membership(_gcx, email=g_email,
                                            stripe_customer_id=stripe_cus, stripe_payment_method_id=stripe_pm,
                                            amount_cents=_gb.MEMBERSHIP_AMOUNT_CENTS, next_charge_date=start)
                            except Exception as _ge:
                                app.logger.exception("group bundle grant failed: %s", _ge)
```
    Reuse the same `pi_id → stripe_cus / stripe_pm` extraction the `kind=='subscribe'` branch uses (vaulted via `save_card=True`). Idempotency: keying on "an active membership already exists for this email" both dedups and implements window-stacking; for finer per-invoice idempotency, also guard on the invoice id if the existing subscribe branch does.
  - **Page:** add the opt-in checkbox to the program checkout page(s) with the disclosure copy: "Add live group coaching — {N} months free with your program, then $99/mo, cancel anytime." Only shown when `GROUP_BUNDLE_ENABLED`.

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(group-bundle): checkout opt-in + grant membership trial on return`

---

### Task 5: reminder + cancel (reuse the portal)

**Files:** Modify `app.py` (heads-up email copy branch for membership), confirm cancel path. Test `tests/test_group_bundle_grant.py` (append)

- [ ] **Step 1:** The 3-day heads-up cron (`list_heads_up_due`) is kind-agnostic, so membership trials already get a pre-charge reminder. Branch the email copy: for a `kind=='membership'` sub, the heads-up + receipt copy say "your live group coaching" / "first charge" rather than product-order language. Member cancel reuses the existing subscription-cancel endpoint (`set_status(...'cancelled')`) — confirm a cancelled membership is excluded from `list_due` (it is: `status='active'` filter). Add a test: a cancelled membership is not charged; a heads-up for a membership uses the membership copy.
- [ ] **Step 2:** Confirm the member portal (`get_manageable_by_email`) lists the membership so they can cancel; if it filters by kind, include membership. (It returns all non-cancelled — fine.)
- [ ] **Step 3:** Commit — `feat(group-bundle): membership reminder copy + cancel`

---

### Task 6: doc + suite

**Files:** Create `docs/group-bundle.md`

- [ ] **Step 1:** Document: the offer (1 free live-group month per program month, cap 3, Biofield-alone=0); explicit opt-in + disclosure + reminder + one-click cancel (negative-option compliance); Stripe-vault rail (`kind='membership'`, flat $99, delayed first charge); window-stacking on re-purchase; `GROUP_BUNDLE_ENABLED` flag (default off); the deferred QBO-Payments alternative.
- [ ] **Step 2:** Combined suite — green:
`doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_group_bundle.py tests/test_subscriptions_membership.py tests/test_membership_charge_cron.py tests/test_group_bundle_grant.py -q` plus the existing subscription-cron + subscription tests to confirm no regression.
- [ ] **Step 3:** Commit — `docs(group-bundle): program-bundled live group`

---

## Self-review

- **Spec coverage:** 1-per-program-month / cap 3 / Biofield-alone 0 (Task 1); Stripe-vault flat-$99 `kind='membership'` rail (Task 2-3); delayed first charge = end of free window + window-stacking (Task 4); explicit opt-in + disclosure + reminder + cancel for negative-option compliance (Task 4-5); `GROUP_BUNDLE_ENABLED` dark flag (all).
- **Type consistency:** `included_group_months(program_months)->int`; `create_membership(...,amount_cents,next_charge_date,cadence_months=1)->id`; `active_memberships_by_email`; `migrate_add_membership_columns`; metadata `grant_group_months`/`email`; `kind='membership'`, `source='membership'`.
- **Deferred:** QBO-Payments recurring rail (pending Rae); "what qualifies as a program order" gate (all program orders vs Biofield-designed only vs min-size) — **confirm with Glen before flipping the flag**; multi-membership-per-email beyond the first; proration.
- **Risk:** money path + auto-renewal compliance. Mitigations — explicit opt-in + clear disclosure + pre-charge reminder + one-click cancel; flag default off; charge branch isolated from the product path; idempotent grant; the flat charge can't exceed `amount_cents`. Verify against the existing subscription-cron test that product charges are unaffected.

## Done
A program purchase optionally bundles live-group months free, then auto-continues at $99/mo on the Stripe vault, with compliant opt-in/disclosure/cancel — shipped dark behind `GROUP_BUNDLE_ENABLED`.
