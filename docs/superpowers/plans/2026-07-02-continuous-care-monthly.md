# Continuous Care Monthly Auto-Charge + Term Cap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Add a monthly auto-charged ($99/mo) Continuous Care option for 6/12-month commitments
that STOPS after the term (no auto-renew), reusing the existing subscriptions/charge-cron infra;
retire the 3-month prepay tier.

**Architecture:** New nullable `term_charges_total` cap on `subscriptions`, enforced in the
existing membership charge-cron branch; a new flag-gated `/continuous-care/checkout` + fulfiller
that vaults a card and creates a capped membership sub (mirroring the group-bundle pattern);
prepay picker gains a monthly-vs-up-front choice for 6/12.

**Tech stack:** Flask monolith `app.py`, `dashboard/subscriptions.py`, `dashboard/prepay.py`,
`static/prepay.html`, SQLite, Stripe. Tests via `doppler run -p remedy-match -c dev -- python3 -m pytest`.

## Global Constraints
- **Money-critical.** No path may double-charge, charge past the term, or charge without consent.
- `CONTINUOUS_CARE_MONTHLY_ENABLED` off (default) ⇒ no monthly checkout; picker = up-front only;
  cron behavior for existing subs byte-identical (cap NULL). The $1 deposit (Model #2) stays
  no-auto-charge — untouched.
- `term_charges_total` NULL = uncapped (every existing group-bundle sub is unaffected).
- Rate = `prepay.MONTHLY_ANCHOR_CENTS` (9900) — single source, never hardcode 9900.
- Copy: em-dash-free (Glen's rule).
- Idempotency + closed-tab safety: fulfiller callable from both `/…/return` and `/webhook/stripe`,
  claim-then-create on a `session_id PRIMARY KEY` table, PaymentIntent re-fetched + `succeeded`.

---

### Task 1: `term_charges_total` column + `create_membership` kwargs

**Files:** Modify `dashboard/subscriptions.py`. Test: `tests/test_subscriptions_term_cap.py`.

**Interfaces — Produces:** `migrate_add_term_cap_column(cx)`; `create_membership(..., term_charges_total=None, initial_order_count=0)`.

- [ ] **Step 1: failing test**
```python
# tests/test_subscriptions_term_cap.py
import sqlite3
import dashboard.subscriptions as subs

def _cx():
    cx = sqlite3.connect(":memory:")
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    return cx

def test_migration_idempotent():
    cx = _cx()
    subs.migrate_add_term_cap_column(cx)  # second call must not raise
    cols = {r[1] for r in cx.execute("PRAGMA table_info(subscriptions)")}
    assert "term_charges_total" in cols

def test_create_membership_writes_cap_and_initial_count():
    cx = _cx()
    sid = subs.create_membership(cx, email="a@x.com", stripe_customer_id="cus",
                                 stripe_payment_method_id="pm", amount_cents=9900,
                                 next_charge_date="2026-08-02", cadence_months=1,
                                 term_charges_total=6, initial_order_count=1)
    row = subs.get(cx, sid)
    assert row["term_charges_total"] == 6
    assert row["order_count"] == 1

def test_create_membership_defaults_uncapped():
    cx = _cx()
    sid = subs.create_membership(cx, email="b@x.com", stripe_customer_id="cus",
                                 stripe_payment_method_id="pm", amount_cents=9900,
                                 next_charge_date="2026-08-02")
    row = subs.get(cx, sid)
    assert row["term_charges_total"] is None
    assert row["order_count"] == 0
```
- [ ] **Step 2: run → fail** (`migrate_add_term_cap_column` undefined).
- [ ] **Step 3: implement** in `dashboard/subscriptions.py`, mirroring the other `migrate_add_*`:
```python
def migrate_add_term_cap_column(cx):
    """Idempotent: add term_charges_total (NULL = uncapped) for fixed-term memberships."""
    cols = {r[1] for r in cx.execute("PRAGMA table_info(subscriptions)")}
    if "term_charges_total" not in cols:
        cx.execute("ALTER TABLE subscriptions ADD COLUMN term_charges_total INTEGER")
        cx.commit()
```
Update `create_membership` signature + INSERT:
```python
def create_membership(cx, *, email, stripe_customer_id, stripe_payment_method_id,
                      amount_cents, next_charge_date, cadence_months=1,
                      term_charges_total=None, initial_order_count=0) -> int:
    """Insert an active flat-amount membership subscription (no product items).
    The first charge lands on next_charge_date. term_charges_total caps total charges
    (NULL = uncapped, legacy behavior); initial_order_count records charges already taken
    at checkout (e.g. 1 when month 1 was charged in the checkout session)."""
    now = _now_iso()
    cur = cx.execute(
        """INSERT INTO subscriptions
               (email, stripe_customer_id, stripe_payment_method_id, items_json,
                cadence_months, status, order_count, next_charge_date, ship_address_json,
                skip_next, created_at, updated_at, kind, amount_cents, term_charges_total)
           VALUES (?,?,?,?,?,'active',?,?,?,0,?,?, 'membership', ?, ?)""",
        (email, stripe_customer_id, stripe_payment_method_id, "[]",
         int(cadence_months), int(initial_order_count), next_charge_date, "{}", now, now,
         int(amount_cents), (int(term_charges_total) if term_charges_total is not None else None)),
    )
    cx.commit()
    try:
        _customers.find_or_create_by_email(cx, email=email)
    except Exception:
        pass
    return cur.lastrowid
```
> NOTE: `migrate_add_term_cap_column` must be called wherever the other membership migrations run at startup (find the call site of `migrate_add_membership_columns` and add this beside it) AND in the fulfiller before insert (Task 4). Implementer: grep for `migrate_add_membership_columns(` and add `migrate_add_term_cap_column(` at each call site.
- [ ] **Step 4: run → pass.** `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_subscriptions_term_cap.py -q`
- [ ] **Step 5: commit.**

---

### Task 2: Term-cap enforcement in the charge cron

**Files:** Modify `app.py` `cron_charge_subscriptions` (membership branch, after the receipt-email block ~22887). Test: `tests/test_sub_cron_term_cap.py`.

**Interfaces — Consumes:** Task 1's `term_charges_total` column.

The cap check goes immediately after the receipt-email `try/except` (line ~22887), still inside
the `if sub.get("kind") == "membership":` success path, using the already-fetched `updated` row:
```python
                        # Term cap: a fixed-term Continuous Care sub stops after its committed
                        # number of charges (no auto-renew). NULL cap = legacy uncapped membership.
                        try:
                            _cap = updated.get("term_charges_total") if updated else None
                            if _cap and updated.get("order_count", 0) >= int(_cap):
                                _subs.set_status(cx, sid, "cancelled")
                        except Exception as _ce:
                            print(f"[sub-cron] term-cap sub={sid}: {_ce!r}", flush=True)
```
> The receipt email already fired above; a "term complete" variant is a follow-on. Cancelling
> here means `list_due` won't select it next cycle, so no N+1 charge. `set_status('cancelled')`
> zeroes order_count (fine, the term is over).

- [ ] **Step 1: failing test** — drive the cron with a monkeypatched Stripe so a capped sub cancels exactly at the cap. Use the app's test client + a real in-file LOG_DB seed (pattern: existing `tests/test_*sub*`/`test_membership.py`). Minimum viable:
```python
# tests/test_sub_cron_term_cap.py
import sqlite3, app as appmod
import dashboard.subscriptions as subs

def _seed_capped(cx, cap, order_count, next_date):
    subs.init_subscriptions_table(cx); subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    return subs.create_membership(cx, email="c@x.com", stripe_customer_id="cus",
        stripe_payment_method_id="pm", amount_cents=9900, next_charge_date=next_date,
        cadence_months=1, term_charges_total=cap, initial_order_count=order_count)

def test_capped_sub_cancels_at_cap(monkeypatch):
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "ch_1"})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **k: None)
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: None)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sid = _seed_capped(cx, cap=6, order_count=5, next_date="2000-01-01")  # due, 1 charge from cap
    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions",
               headers={"X-Cron-Secret": appmod.CRON_SECRET or appmod.CONSOLE_SECRET or ""})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = subs.get(cx, sid)
    assert row["status"] == "cancelled"  # order_count hit 6 -> term over

def test_uncapped_sub_stays_active(monkeypatch):
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "ch_2"})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id": "INV2"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **k: None)
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: None)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sid = _seed_capped(cx, cap=None, order_count=99, next_date="2000-01-01")
    c = appmod.app.test_client()
    c.post("/api/cron/charge-subscriptions",
           headers={"X-Cron-Secret": appmod.CRON_SECRET or appmod.CONSOLE_SECRET or ""})
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = subs.get(cx, sid)
    assert row["status"] == "active"
```
> Implementer: adjust auth-header/secret + any required table init to match how existing cron tests authenticate (read `tests/test_*` that POST to `/api/cron/charge-subscriptions`; if none, read `cron_charge_subscriptions` auth block at app.py:22762). Clean the seeded rows if the suite shares LOG_DB (delete by email in a teardown) to avoid pollution.
- [ ] **Step 2: run → fail** (sub stays active at cap).
- [ ] **Step 3: implement** the cap block above.
- [ ] **Step 4: run → pass.**
- [ ] **Step 5: commit.**

---

### Task 3: prepay.py — retire 3mo, expose monthly option data

**Files:** Modify `dashboard/prepay.py`. Test: `tests/test_prepay_monthly.py`.

**Interfaces — Produces:** `monthly_total_cents(key)`, `upfront_savings_cents(key)`; `tiers_public()` items gain `monthly_cents`, `monthly_total_cents`, `upfront_savings_cents`, `commitment`; 3mo excluded from `tiers_public()`.

- [ ] **Step 1: failing test**
```python
# tests/test_prepay_monthly.py
import dashboard.prepay as prepay

def test_public_tiers_exclude_3mo():
    keys = [t["key"] for t in prepay.tiers_public()]
    assert keys == ["1mo", "6mo", "12mo"]

def test_monthly_total_and_savings():
    assert prepay.monthly_total_cents("6mo") == 9900 * 6          # 59400
    assert prepay.upfront_savings_cents("6mo") == 59400 - 54600   # 4800
    assert prepay.monthly_total_cents("12mo") == 9900 * 12        # 118800
    assert prepay.upfront_savings_cents("12mo") == 118800 - 99000 # 19800

def test_commitment_flag_and_fields():
    by = {t["key"]: t for t in prepay.tiers_public()}
    assert by["6mo"]["commitment"] is True and by["12mo"]["commitment"] is True
    assert by["1mo"]["commitment"] is False
    assert by["6mo"]["monthly_cents"] == 9900
    assert by["6mo"]["monthly_total_cents"] == 59400
    assert by["6mo"]["upfront_savings_cents"] == 4800
```
- [ ] **Step 2: run → fail.**
- [ ] **Step 3: implement** in `dashboard/prepay.py`:
  - Add `PUBLIC_TIER_KEYS = ["1mo", "6mo", "12mo"]` and `COMMITMENT_TIER_KEYS = ["6mo", "12mo"]` near TIERS (leave the 3mo entry in TIERS for easy restore).
  - Add:
```python
def monthly_total_cents(key):
    t = get_tier(key)
    return MONTHLY_ANCHOR_CENTS * t["months"] if t else 0

def upfront_savings_cents(key):
    t = get_tier(key)
    return max(0, monthly_total_cents(key) - t["price_cents"]) if t else 0
```
  - In `tiers_public()`: iterate only `PUBLIC_TIER_KEYS`, and for each add
    `"commitment": key in COMMITMENT_TIER_KEYS`, `"monthly_cents": MONTHLY_ANCHOR_CENTS`,
    `"monthly_total_cents": monthly_total_cents(key)`, `"upfront_savings_cents": upfront_savings_cents(key)`
    alongside the existing fields.
- [ ] **Step 4: run → pass.**
- [ ] **Step 5: commit.**

---

### Task 4: Monthly checkout route + fulfiller + flag + webhook wiring

**Files:** Modify `app.py`. Test: `tests/test_continuous_care_monthly.py`.

**Interfaces — Consumes:** Task 1 (`create_membership` cap kwargs, `migrate_add_term_cap_column`). **Produces:** `POST /continuous-care/checkout`, `GET /continuous-care/return`, `_fulfill_continuous_care_monthly(session_id)`, flag `CONTINUOUS_CARE_MONTHLY_ENABLED`.

Follow the EXISTING patterns exactly:
- Flag: near `PREPAY_LADDER_ENABLED` (app.py ~4507):
  `CONTINUOUS_CARE_MONTHLY_ENABLED = os.environ.get("CONTINUOUS_CARE_MONTHLY_ENABLED","").strip().lower() in ("1","true","yes","on")`
- Route `POST /continuous-care/checkout` — mirror `prepay_checkout` (app.py:2943) but:
  `if not CONTINUOUS_CARE_MONTHLY_ENABLED or not _STRIPE_ACTIVE: return jsonify({"ok": False}), 200`;
  read `term_months = int(body.get("term_months") or 0)`; validate `term_months in (6, 12)` else `{ok:False}`;
  `create_checkout_session(prepay.MONTHLY_ANCHOR_CENTS, customer_email=email,
  description=f"Remedy Match Continuous Care - {term_months} month (monthly)",
  metadata={"email": email, "kind": "continuous_care_monthly", "term_months": str(term_months)},
  success_url=f"{base}/continuous-care/return?session_id={{CHECKOUT_SESSION_ID}}",
  cancel_url=f"{base}/", save_card=True)`. Return `{ok:True, url: sess["url"]}`.
- Route `GET /continuous-care/return` — mirror `prepay_return`: call `_fulfill_continuous_care_monthly(session_id)`, redirect `/?care=ok|err`.
- `_fulfill_continuous_care_monthly(session_id)` — mirror `_fulfill_prepay_term` (app.py:7069) for
  security + idempotency, and mirror the REMOVED biofield-trial block for the membership creation:
  - Re-fetch session; `email = session.metadata["email"]`, `term_months = int(session.metadata["term_months"])`, guard `term_months in (6,12)`.
  - Re-fetch PaymentIntent; require `pi["status"] == "succeeded"`. Pull `customer = pi["customer"]`, `pm = pi["payment_method"]`; if either missing → log + return (no membership without a vaulted card).
  - `with _db_lock, sqlite3.connect(LOG_DB) as cx:` — init tables (`_subs.init_subscriptions_table`, `_subs.migrate_add_membership_columns`, `_subs.migrate_add_term_cap_column`, `init_membership_tables`), create claim table `continuous_care_grants(session_id TEXT PRIMARY KEY, email TEXT, created_at TEXT)`, `INSERT OR IGNORE` the session_id; if `cur.rowcount == 0` → already fulfilled, return (idempotent).
  - `next_charge = _subs.add_months(date.today().isoformat(), 1)`.
  - `_subs.create_membership(cx, email=email, stripe_customer_id=customer, stripe_payment_method_id=pm, amount_cents=prepay.MONTHLY_ANCHOR_CENTS, next_charge_date=next_charge, cadence_months=1, term_charges_total=term_months, initial_order_count=1)` — order_count=1 = the month-1 charge just taken; cap = total months.
  - `_grant_membership(cx, email, prepay.term_days(date.today().isoformat(), 1) + 4, "continuous_care")` for immediate access until the first cron charge extends the grant (≈35 days).
  - Mint the FTC cancel token exactly as the removed biofield-trial block (`auth_tokens`, purpose `membership_cancel`, TTL `MEMBERSHIP_CANCEL_TTL_DAYS`).
  - Best-effort `_ingest_order(source="continuous_care_monthly", external_ref=pi["id"], email=email, items=[], total_cents=prepay.MONTHLY_ANCHOR_CENTS, address={}, channel="retail")` + confirmation email. Never raise.
- Wire `_fulfill_continuous_care_monthly(session_id)` into `webhook_stripe` (app.py:17203 area) beside the other fulfillers (it no-ops on non-matching kind because the claim/kind guards return early).

- [ ] **Step 1: failing tests**
```python
# tests/test_continuous_care_monthly.py
import sqlite3, app as appmod
import dashboard.subscriptions as subs

def test_checkout_flag_off(monkeypatch):
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", False)
    c = appmod.app.test_client()
    r = c.post("/continuous-care/checkout", json={"email": "a@x.com", "term_months": 6})
    assert r.status_code == 200 and r.get_json()["ok"] is False

def test_checkout_builds_vaulted_session(monkeypatch):
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    cap = {}
    def fake_sess(amount, **kw):
        cap["amount"] = amount; cap["kw"] = kw
        return {"url": "https://stripe/x"}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session", fake_sess)
    c = appmod.app.test_client()
    r = c.post("/continuous-care/checkout", json={"email": "a@x.com", "term_months": 6})
    assert r.get_json()["ok"] is True
    assert cap["amount"] == appmod.prepay.MONTHLY_ANCHOR_CENTS
    assert cap["kw"]["save_card"] is True
    assert cap["kw"]["metadata"]["kind"] == "continuous_care_monthly"
    assert cap["kw"]["metadata"]["term_months"] == "6"

def test_checkout_rejects_bad_term(monkeypatch):
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    c = appmod.app.test_client()
    r = c.post("/continuous-care/checkout", json={"email": "a@x.com", "term_months": 3})
    assert r.get_json()["ok"] is False

def _mock_stripe_success(monkeypatch, email="m@x.com", term="6"):
    monkeypatch.setattr(appmod.stripe_pay, "get_checkout_session",
        lambda sid: {"metadata": {"email": email, "kind": "continuous_care_monthly", "term_months": term}})
    monkeypatch.setattr(appmod.stripe_pay, "get_payment_intent",
        lambda pi=None, **k: {"status": "succeeded", "id": "pi_1", "customer": "cus_1", "payment_method": "pm_1"})

def test_fulfill_creates_capped_membership_and_grants_access(monkeypatch):
    _mock_stripe_success(monkeypatch)
    monkeypatch.setattr(appmod, "_ingest_order", lambda **k: None)
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: None)
    appmod._fulfill_continuous_care_monthly("sess_ccm_1")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rows = subs.active_memberships_by_email(cx, "m@x.com")
    assert len(rows) == 1
    assert rows[0]["term_charges_total"] == 6
    assert rows[0]["order_count"] == 1
    assert rows[0]["stripe_payment_method_id"] == "pm_1"
    assert appmod._is_paid_member("m@x.com") is True
    assert appmod.membership_category("m@x.com") == "full"

def test_fulfill_idempotent(monkeypatch):
    _mock_stripe_success(monkeypatch, email="idem@x.com")
    monkeypatch.setattr(appmod, "_ingest_order", lambda **k: None)
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: None)
    appmod._fulfill_continuous_care_monthly("sess_ccm_2")
    appmod._fulfill_continuous_care_monthly("sess_ccm_2")  # replay
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rows = subs.active_memberships_by_email(cx, "idem@x.com")
    assert len(rows) == 1  # no double sub

def test_fulfill_no_membership_when_pi_not_succeeded(monkeypatch):
    monkeypatch.setattr(appmod.stripe_pay, "get_checkout_session",
        lambda sid: {"metadata": {"email": "fail@x.com", "kind": "continuous_care_monthly", "term_months": "6"}})
    monkeypatch.setattr(appmod.stripe_pay, "get_payment_intent",
        lambda pi=None, **k: {"status": "requires_payment_method", "id": "pi_x"})
    appmod._fulfill_continuous_care_monthly("sess_ccm_3")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rows = subs.active_memberships_by_email(cx, "fail@x.com")
    assert rows == []
```
> Implementer: match the ACTUAL Stripe-helper accessor names/signatures used by `_fulfill_prepay_term` (e.g. how it re-fetches the session + PI, and how it reads `metadata`) — the mock names above (`get_checkout_session`/`get_payment_intent`) are indicative; read app.py:7069 and align both the code and the test monkeypatches to the real helpers. Guard the tests against LOG_DB pollution (unique emails + best-effort row cleanup).
- [ ] **Step 2: run → fail.**
- [ ] **Step 3: implement** the flag, both routes, the fulfiller, and the webhook wiring.
- [ ] **Step 4: run → pass** the new file, and run `tests/test_prepay_checkout.py tests/test_stripe_webhook.py` to confirm no regression in the sibling flows.
- [ ] **Step 5: commit.**

---

### Task 5: Picker UI — monthly vs up-front for 6/12

**Files:** Modify `static/prepay.html`; add a small flag surface (inject `window.__CARE__ = {monthly_enabled: CONTINUOUS_CARE_MONTHLY_ENABLED}` into the `/prepay` response, OR add `monthly_enabled` to `/api/prepay/tiers`). Test: light — assert the served page + tiers payload carry the hooks.

**Interfaces — Consumes:** Task 3 tier fields; Task 4 flag + `/continuous-care/checkout`.

- [ ] **Step 1:** In `prepay_page` (app.py:2922), inject `window.__CARE__ = {"monthly_enabled": CONTINUOUS_CARE_MONTHLY_ENABLED}` before `</head>` (escape `<>&`, same helper as other injectors). (Alternatively add `monthly_enabled` to the `/api/prepay/tiers` JSON — pick one, keep it consistent.)
- [ ] **Step 2:** In `static/prepay.html`, for tiers where `t.commitment` and `__CARE__.monthly_enabled`, render two actions: **"Pay monthly - $99/mo"** → `POST /continuous-care/checkout {email, term_months: t.months}` (then `location = url`); **"Pay up front - $<price>, save $<upfront_savings>"** → existing `POST /prepay/checkout {email, tier_key: t.key}`. For `1mo` and when monthly disabled, keep today's single up-front "Continue". Reuse existing `.tier` styling; em-dash-free copy.
- [ ] **Step 3:** Test (`tests/test_prepay_monthly.py` add-on or the served-route test): `GET /prepay` with `CONTINUOUS_CARE_MONTHLY_ENABLED` patched true contains `window.__CARE__` with `"monthly_enabled": true`; patched false → `false`. If using the tiers-payload approach, assert `/api/prepay/tiers` carries `monthly_enabled` + the per-tier fields from Task 3.
- [ ] **Step 4:** run → pass.
- [ ] **Step 5:** commit.

---

## Verification (end-to-end)
- All five test files green in isolation.
- Regression: `tests/test_prepay_checkout.py tests/test_prepay_model.py tests/test_membership.py tests/test_stripe_webhook.py` — no NEW failures vs merge-base (suite is pollution-noisy; judge by isolation / before-after diff, not raw counts).
- Flag-off: `/continuous-care/checkout` → `{ok:False}`; `/prepay` shows up-front only; a NULL-cap group-bundle sub still charges (Task 2 uncapped test).
- Ship flag-dark; render-verify on prod before flipping `CONTINUOUS_CARE_MONTHLY_ENABLED`.
