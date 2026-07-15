# Membership Products Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three buyable membership tiers — (A) 1 month $99 no-renew, (B) 12 months $99/mo no-auto-renew, (C) 1 year full-pay $990 no-renew — all granting the same entitlement (live group coaching access + member pricing on remedies), plus an owner endpoint to enroll a client manually.

**Architecture:** A new `dashboard/membership_products.py` catalog defines the three tiers and their billing type. A dedicated Stripe flow (`/membership/checkout` → `_fulfill_membership_product` → `/membership/return`, plus a webhook fan-out line) mints and fulfills. Fulfillment reuses the two existing primitives verbatim: grant-only via `_grant_membership` for the one-time tiers A/C (identical to the prepay `1mo`/`12mo` paths, so the charge cron never bills them → cannot auto-renew), and `create_membership(term_charges_total=12, initial_order_count=1)` for the recurring-capped tier B (the existing Continuous Care term-cap stop at `app.py:32528-32535` cancels the sub after 12 charges). All three write a non-trial `memberships` grant (source `membership_*`), which `_is_paid_member` already reads for member pricing; group-coaching ownership is unified so the same grant counts. An owner console endpoint grants the entitlement without a Stripe round-trip (the gap that forced Dana Tamraz's membership to be a hand-built payment link).

**Tech Stack:** Python 3 / Flask (`app.py` monolith + `dashboard/*.py`), SQLite (`LOG_DB` / `chat_log.db`), Stripe Checkout via `dashboard/stripe_pay.py` (raw HTTPS, no SDK), QuickBooks Online via `dashboard/qbo_billing.py`, pytest with Flask `test_client()`.

## Global Constraints

- **No CI; merge = deploy** (deploy-chat has no GitHub Actions). Every task must be independently green under focused pytest before commit. Run app-importing tests with `doppler run -c dev -- pytest` (bare pytest silently skips app-importing tests, and a bare full-suite sends real email — use focused test paths).
- **Dark-launch behind a feature flag** `MEMBERSHIP_PRODUCTS_ENABLED` (env, default off). No new customer-facing route responds until the flag is on in prod Doppler.
- **All three tiers grant the SAME entitlement for now:** live group coaching access + member pricing. Differ only in billing.
- **Never auto-renew.** Tier A/C are grant-only (no `subscriptions` row → cron can't bill). Tier B is a 12-charge capped `subscriptions` row that self-cancels at the cap; renewal is a manual re-signup.
- **Tier amounts (exact):** A `month` = 9900 cents; B `year_monthly` = 9900 cents/mo × 12; C `year_prepay` = 99000 cents one-time.
- **Copy rules (Glen):** no em dashes, no ALL CAPS, no "Hook:" label in any user-facing string.
- **Money-path discipline:** TDD every task; mutation-test the entitlement/idempotency guards (inject the violation, watch the test go red, revert). Customer PII never goes into PR bodies (repo is public-adjacent).
- **Member-pricing gate is unchanged:** `_is_paid_member(email)` = active non-trial `memberships` grant. All three tiers create that grant, so member pricing needs no pricing-code change.

---

### Task 1: Membership product catalog module

**Files:**
- Create: `dashboard/membership_products.py`
- Test: `tests/test_membership_products_catalog.py`

**Interfaces:**
- Produces:
  - `TIERS: dict[str, dict]` keyed `"month" | "year_monthly" | "year_prepay"`, each `{key, label, price_cents, billing, source, term_charges, cadence_months, grant_months}` where `billing ∈ {"one_time", "recurring_capped"}`.
  - `get_tier(key: str) -> dict | None`
  - `all_tiers() -> list[dict]` (stable display order: month, year_monthly, year_prepay)
  - `grant_days(key: str, today: datetime.date) -> int` (calendar months of coverage + `GRACE_DAYS`)
  - `GRACE_DAYS = 4`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membership_products_catalog.py
import datetime
from dashboard import membership_products as mp

def test_three_tiers_with_exact_amounts():
    assert set(mp.TIERS) == {"month", "year_monthly", "year_prepay"}
    assert mp.get_tier("month")["price_cents"] == 9900
    assert mp.get_tier("year_monthly")["price_cents"] == 9900
    assert mp.get_tier("year_prepay")["price_cents"] == 99000

def test_billing_types():
    assert mp.get_tier("month")["billing"] == "one_time"
    assert mp.get_tier("year_prepay")["billing"] == "one_time"
    ym = mp.get_tier("year_monthly")
    assert ym["billing"] == "recurring_capped"
    assert ym["term_charges"] == 12
    assert ym["cadence_months"] == 1

def test_sources_are_membership_namespaced():
    for t in mp.all_tiers():
        assert t["source"].startswith("membership_")

def test_grant_days_covers_the_term_plus_grace():
    today = datetime.date(2026, 7, 15)
    # 1 month tier: ~31 days + 4 grace
    assert 33 <= mp.grant_days("month", today) <= 36
    # 1 year tiers: ~365 days + 4 grace
    assert 366 <= mp.grant_days("year_prepay", today) <= 370
    assert 366 <= mp.grant_days("year_monthly", today) <= 370

def test_unknown_tier_returns_none():
    assert mp.get_tier("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -c dev -- pytest tests/test_membership_products_catalog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.membership_products'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/membership_products.py
"""Membership product catalog: the three buyable tiers, all granting the same
entitlement (live group coaching + member pricing) and differing only in billing.
Reuses existing fulfillment primitives — one-time tiers are grant-only (like the
prepay ladder, so the charge cron never bills them and they cannot auto-renew);
the recurring-capped tier uses a subscriptions row with term_charges_total that
self-cancels at the cap (dashboard.subscriptions + app charge cron)."""
import calendar
import datetime

GRACE_DAYS = 4

TIERS = {
    "month": {
        "key": "month", "label": "Monthly Membership", "price_cents": 9900,
        "billing": "one_time", "source": "membership_month",
        "term_charges": 1, "cadence_months": 1, "grant_months": 1,
    },
    "year_monthly": {
        "key": "year_monthly", "label": "Annual Membership (monthly)",
        "price_cents": 9900, "billing": "recurring_capped",
        "source": "membership_year_monthly",
        "term_charges": 12, "cadence_months": 1, "grant_months": 12,
    },
    "year_prepay": {
        "key": "year_prepay", "label": "Annual Membership (full pay)",
        "price_cents": 99000, "billing": "one_time",
        "source": "membership_year_prepay",
        "term_charges": 1, "cadence_months": 1, "grant_months": 12,
    },
}

_ORDER = ["month", "year_monthly", "year_prepay"]

def get_tier(key):
    return TIERS.get(key)

def all_tiers():
    return [TIERS[k] for k in _ORDER]

def _add_months(d, months):
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return datetime.date(y, m, day)

def grant_days(key, today):
    t = TIERS[key]
    end = _add_months(today, t["grant_months"])
    return (end - today).days + GRACE_DAYS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -c dev -- pytest tests/test_membership_products_catalog.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/membership_products.py tests/test_membership_products_catalog.py
git commit -m "feat(membership): add membership product catalog (3 tiers)"
```

---

### Task 2: Unify group-coaching ownership to honor membership grants

**Why:** Member pricing already keys off `_is_paid_member` (the `memberships` grant), so all three tiers get it for free. But live-group ownership (`portal_offers._owns_group`) currently reads only `subscriptions` rows — so the grant-only tiers A/C would not count as group members and the portal would keep re-offering "join live group." Broaden ownership to also honor an active `memberships` grant whose source is `membership_*`. Keep it narrow: prepay/CC/founding grants (different sources) are unaffected.

**Files:**
- Modify: `dashboard/membership_products.py` (add `owns_group`)
- Modify: `dashboard/portal_offers.py:17-27` (`_owns_group` ORs in the new predicate)
- Test: `tests/test_membership_group_ownership.py`

**Interfaces:**
- Consumes: `_grant_membership` semantics from Task 3 (writes `memberships.source`); Task 1 `TIERS`.
- Produces: `membership_products.owns_group(cx, email) -> bool` — True iff an active (`expires_at > now`) `memberships` row exists with `source` in the tier sources, OR the caller's existing subscriptions-based ownership holds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membership_group_ownership.py
import datetime, sqlite3, uuid
from dashboard import membership_products as mp

def _mk_db(tmp_path):
    cx = sqlite3.connect(tmp_path / "t.db")
    cx.execute("""CREATE TABLE memberships (id TEXT PRIMARY KEY, email TEXT NOT NULL,
        granted_at TEXT NOT NULL, expires_at TEXT, granted_by TEXT, source TEXT,
        truly_vip_ref TEXT, notes TEXT, last_reminder_at TEXT)""")
    cx.commit()
    return cx

def _grant(cx, email, source, days):
    now = datetime.datetime.utcnow()
    exp = (now + datetime.timedelta(days=days)).isoformat()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,granted_by,source) "
               "VALUES (?,?,?,?,?,?)",
               (uuid.uuid4().hex, email, now.isoformat(), exp, source, source))
    cx.commit()

def test_membership_grant_owns_group(tmp_path):
    cx = _mk_db(tmp_path)
    _grant(cx, "a@x.com", "membership_month", 34)
    assert mp.owns_group(cx, "a@x.com") is True

def test_prepay_grant_does_not_own_group(tmp_path):
    cx = _mk_db(tmp_path)
    _grant(cx, "b@x.com", "prepay_12mo", 369)  # different namespace
    assert mp.owns_group(cx, "b@x.com") is False

def test_expired_membership_grant_does_not_own(tmp_path):
    cx = _mk_db(tmp_path)
    _grant(cx, "c@x.com", "membership_year_prepay", -1)  # already expired
    assert mp.owns_group(cx, "c@x.com") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -c dev -- pytest tests/test_membership_group_ownership.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.membership_products' has no attribute 'owns_group'`

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/membership_products.py`:

```python
def _tier_sources():
    return tuple(t["source"] for t in TIERS.values())

def owns_group(cx, email):
    """True iff the email holds an active membership-tier grant. Namespaced to the
    tier sources so prepay/continuous-care/founding grants are unaffected."""
    if not email:
        return False
    now = datetime.datetime.utcnow().isoformat()
    srcs = _tier_sources()
    ph = ",".join("?" * len(srcs))
    row = cx.execute(
        f"SELECT 1 FROM memberships WHERE lower(email)=lower(?) "
        f"AND expires_at > ? AND source IN ({ph}) LIMIT 1",
        (email, now, *srcs)).fetchone()
    return row is not None
```

Modify `dashboard/portal_offers.py` `_owns_group` (currently `app`-side lines 17-27) to OR in the new predicate:

```python
def _owns_group(cx, email):
    from dashboard import membership_products as _mp
    if _mp.owns_group(cx, email):
        return True
    return bool(_subs.active_memberships_by_email(cx, email))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -c dev -- pytest tests/test_membership_group_ownership.py tests/test_portal_offers*.py -v`
Expected: PASS (new file + existing portal-offers tests still green)

- [ ] **Step 5: Mutation-test the namespace guard**

Temporarily change `prepay_12mo` in the test to `membership_month`; run — the "does_not_own_group" test must go RED. Revert.

- [ ] **Step 6: Commit**

```bash
git add dashboard/membership_products.py dashboard/portal_offers.py tests/test_membership_group_ownership.py
git commit -m "feat(membership): membership-tier grants confer group-coaching ownership"
```

---

### Task 3: Checkout session minting (`POST /membership/checkout`)

**Files:**
- Modify: `app.py` (new route near the other checkout routes, ~`app.py:3496`; helper near `_continuous_care_checkout_session`)
- Test: `tests/test_membership_products_checkout.py`

**Interfaces:**
- Consumes: `membership_products.get_tier`; `dashboard.stripe_pay.create_checkout_session`; `_STRIPE_ACTIVE`; `PUBLIC_BASE_URL`.
- Produces: `POST /membership/checkout` (public, flag-gated) body `{"email": str, "tier": str}` → `{"ok": True, "url": <stripe checkout url>}`; helper `_membership_checkout_session(email, tier_key) -> dict`. Metadata carried into Stripe: `{"email", "kind": "membership_product", "tier": tier_key}`. One-time tiers use `save_card=False`; the recurring-capped tier uses `save_card=True`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membership_products_checkout.py — mirrors tests/test_continuous_care_monthly.py
import importlib, sys, os
import pytest

def _load_app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app import failed: {e}")

@pytest.fixture
def appmod(monkeypatch, tmp_path):
    app = _load_app()
    monkeypatch.setattr(app, "LOG_DB", str(tmp_path / "t.db"), raising=False)
    monkeypatch.setattr(app, "PUBLIC_BASE_URL", "https://illtowell.com", raising=False)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(app, "MEMBERSHIP_PRODUCTS_ENABLED", True, raising=False)
    return app

def test_month_tier_builds_one_time_session(appmod, monkeypatch):
    cap = {}
    def fake_sess(amount, **kw):
        cap["amount"] = amount; cap["kw"] = kw
        return {"id": "cs_test", "url": "https://checkout.stripe.com/x"}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session", fake_sess)
    r = appmod.app.test_client().post("/membership/checkout",
                                      json={"email": "a@x.com", "tier": "month"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert cap["amount"] == 9900
    assert cap["kw"]["save_card"] is False
    assert cap["kw"]["metadata"]["kind"] == "membership_product"
    assert cap["kw"]["metadata"]["tier"] == "month"

def test_year_monthly_tier_vaults_card(appmod, monkeypatch):
    cap = {}
    def fake_sess(amount, **kw):
        cap["amount"] = amount; cap["kw"] = kw
        return {"id": "cs_test", "url": "https://checkout.stripe.com/x"}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session", fake_sess)
    r = appmod.app.test_client().post("/membership/checkout",
                                      json={"email": "a@x.com", "tier": "year_monthly"})
    assert r.status_code == 200
    assert cap["amount"] == 9900
    assert cap["kw"]["save_card"] is True

def test_year_prepay_amount(appmod, monkeypatch):
    cap = {}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session",
                        lambda amount, **kw: (cap.update(amount=amount, kw=kw)
                                              or {"id": "cs", "url": "u"}))
    appmod.app.test_client().post("/membership/checkout",
                                  json={"email": "a@x.com", "tier": "year_prepay"})
    assert cap["amount"] == 99000

def test_unknown_tier_rejected(appmod):
    r = appmod.app.test_client().post("/membership/checkout",
                                      json={"email": "a@x.com", "tier": "nope"})
    assert r.status_code == 400

def test_flag_off_returns_404(appmod, monkeypatch):
    monkeypatch.setattr(appmod, "MEMBERSHIP_PRODUCTS_ENABLED", False, raising=False)
    r = appmod.app.test_client().post("/membership/checkout",
                                      json={"email": "a@x.com", "tier": "month"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -c dev -- pytest tests/test_membership_products_checkout.py -v`
Expected: FAIL — 404 for all (route not defined) / attribute errors.

- [ ] **Step 3: Write minimal implementation**

Add the flag near the other flags in `app.py` (top-of-file env section):

```python
MEMBERSHIP_PRODUCTS_ENABLED = os.environ.get("MEMBERSHIP_PRODUCTS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

Add helper + route (near `_continuous_care_checkout_session`, ~`app.py:3479`):

```python
def _membership_checkout_session(email, tier_key):
    from dashboard import membership_products as _mp
    tier = _mp.get_tier(tier_key)
    base = PUBLIC_BASE_URL.rstrip("/")
    return stripe_pay.create_checkout_session(
        tier["price_cents"], customer_email=email,
        description=f"Remedy Match {tier['label']}",
        metadata={"email": email, "kind": "membership_product", "tier": tier_key},
        success_url=f"{base}/membership/return?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/",
        save_card=(tier["billing"] == "recurring_capped"))

@app.route("/membership/checkout", methods=["POST"])
def membership_checkout():
    if not (MEMBERSHIP_PRODUCTS_ENABLED and _STRIPE_ACTIVE):
        return jsonify({"error": "not found"}), 404
    from dashboard import membership_products as _mp
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    tier_key = (data.get("tier") or "").strip()
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400
    if not _mp.get_tier(tier_key):
        return jsonify({"ok": False, "error": "unknown tier"}), 400
    sess = _membership_checkout_session(email, tier_key)
    return jsonify({"ok": True, "url": sess.get("url", "")})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -c dev -- pytest tests/test_membership_products_checkout.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_membership_products_checkout.py
git commit -m "feat(membership): /membership/checkout mints per-tier Stripe session"
```

---

### Task 4: Fulfillment (`_fulfill_membership_product`) + return route + webhook fan-out

**Files:**
- Modify: `app.py` (fulfiller near `_fulfill_continuous_care_monthly` ~`app.py:8808`; `/membership/return` route near `app.py:3520`; one line in the `/webhook/stripe` fan-out `app.py:26628-26663`; a `membership_product_grants` idempotency table in `init_membership_tables` `app.py:10947`)
- Test: `tests/test_membership_products_fulfill.py`

**Interfaces:**
- Consumes: `membership_products` (Task 1); `_grant_membership`; `dashboard.subscriptions.create_membership`; `stripe_pay.get_session` / `get_payment_intent`; `dashboard.subscriptions.active_memberships_by_email`; `_subs.add_months`; QBO helpers (`qbo_billing.find_or_create_customer`, `create_invoice`, `record_payment`).
- Produces: `_fulfill_membership_product(session_id) -> str` returning a status string (`"ok" | "duplicate_member" | "no_card" | "not_paid" | "already" | "skip"`); `GET /membership/return`; recognized Stripe `kind` `"membership_product"`.

**Fulfillment logic by billing type:**
- `one_time` (month, year_prepay): verify PI succeeded; claim on `membership_product_grants(session_id)`; `_grant_membership(cx, email, membership_products.grant_days(tier, today), tier["source"])`; book a paid QBO invoice mirroring `_fulfill_prepay_term` (`app.py:8658-8805`): `find_or_create_customer` → `create_invoice([{name: tier["label"], amount: price/100, qty: 1}], email_to=email)` → `record_payment(cust_id, price_cents, invoice_id, method="card")`.
- `recurring_capped` (year_monthly): verify PI succeeded AND `customer`+`payment_method` present (else `no_card`); duplicate-member guard (if `active_memberships_by_email` non-empty → extend grant only, return `duplicate_member`); `create_membership(..., amount_cents=9900, cadence_months=1, term_charges_total=12, initial_order_count=1, next_charge_date=_subs.add_months(today,1))` + `_grant_membership(cx, email, grant_days, tier["source"])`. Month 1 is the checkout charge; cron bills 2..12 then self-cancels at the cap.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membership_products_fulfill.py — mirrors tests/test_continuous_care_monthly.py fixture
import importlib, sys, os, sqlite3, datetime
import pytest

def _load_app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app import failed: {e}")

@pytest.fixture
def appmod(monkeypatch, tmp_path):
    app = _load_app()
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(app, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app, "PUBLIC_BASE_URL", "https://illtowell.com", raising=False)
    from dashboard import subscriptions as subs
    cx = sqlite3.connect(db)
    subs.init_subscriptions_table(cx)
    for name in dir(subs):
        if name.startswith("migrate_add"):
            try: getattr(subs, name)(cx)
            except Exception: pass
    app.init_membership_tables(cx)
    cx.close()
    # no-op the side effects
    for fn in ("_ingest_order", "_member_join_welcome"):
        if hasattr(app, fn): monkeypatch.setattr(app, fn, lambda *a, **k: None, raising=False)
    # stub QBO so one-time booking does not hit the network
    import dashboard.qbo_billing as qb
    monkeypatch.setattr(qb, "find_or_create_customer", lambda *a, **k: {"Id": "1"})
    monkeypatch.setattr(qb, "create_invoice", lambda *a, **k: {"Id": "inv1"})
    monkeypatch.setattr(qb, "record_payment", lambda *a, **k: {"Id": "pay1"})
    return app

def _mock_stripe(appmod, monkeypatch, *, tier, with_card=True):
    monkeypatch.setattr(appmod.stripe_pay, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid",
        "metadata": {"kind": "membership_product", "tier": tier, "email": "a@x.com"},
        "payment_intent": "pi_1", "customer": ("cus_1" if with_card else None),
    })
    monkeypatch.setattr(appmod.stripe_pay, "get_payment_intent", lambda pi: {
        "status": "succeeded",
        "customer": ("cus_1" if with_card else None),
        "payment_method": ("pm_1" if with_card else None)})

def test_one_time_month_grants_paid_member(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="month")
    assert appmod._fulfill_membership_product("cs_1") == "ok"
    assert appmod._is_paid_member("a@x.com") is True
    from dashboard import membership_products as mp
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    assert mp.owns_group(cx, "a@x.com") is True

def test_year_monthly_creates_capped_sub(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="year_monthly")
    assert appmod._fulfill_membership_product("cs_2") == "ok"
    from dashboard import subscriptions as subs
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    rows = subs.active_memberships_by_email(cx, "a@x.com")
    assert rows and rows[0]["term_charges_total"] == 12
    assert rows[0]["order_count"] == 1  # month 1 charged at checkout
    assert appmod.membership_category("a@x.com") == "full"

def test_year_monthly_requires_card(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="year_monthly", with_card=False)
    assert appmod._fulfill_membership_product("cs_3") == "no_card"

def test_idempotent_replay(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="month")
    appmod._fulfill_membership_product("cs_4")
    appmod._fulfill_membership_product("cs_4")  # replay
    cx = sqlite3.connect(appmod.LOG_DB)
    n = cx.execute("SELECT COUNT(*) FROM memberships WHERE email='a@x.com'").fetchone()[0]
    assert n == 1

def test_duplicate_member_no_second_sub(appmod, monkeypatch):
    _mock_stripe(appmod, monkeypatch, tier="year_monthly")
    appmod._fulfill_membership_product("cs_5")
    monkeypatch.setattr(appmod.stripe_pay, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid",
        "metadata": {"kind": "membership_product", "tier": "year_monthly", "email": "a@x.com"},
        "payment_intent": "pi_2", "customer": "cus_1"})
    assert appmod._fulfill_membership_product("cs_6") == "duplicate_member"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -c dev -- pytest tests/test_membership_products_fulfill.py -v`
Expected: FAIL — `_fulfill_membership_product` not defined.

- [ ] **Step 3: Write minimal implementation**

Add the idempotency table inside `init_membership_tables` (`app.py:10947`):

```python
cx.execute("""CREATE TABLE IF NOT EXISTS membership_product_grants (
    session_id TEXT PRIMARY KEY, email TEXT, tier TEXT, created_at TEXT)""")
```

Add the fulfiller near `_fulfill_continuous_care_monthly`:

```python
def _fulfill_membership_product(session_id):
    """Fulfill a /membership/checkout Stripe session. Self-dispatching: no-ops
    unless metadata kind == 'membership_product'. Idempotent on session_id."""
    from dashboard import membership_products as _mp
    from dashboard import subscriptions as _subs
    sess = stripe_pay.get_session(session_id)
    md = sess.get("metadata") or {}
    if md.get("kind") != "membership_product":
        return "skip"
    tier = _mp.get_tier(md.get("tier") or "")
    email = (md.get("email") or "").strip().lower()
    if not (tier and email):
        return "skip"
    pi = stripe_pay.get_payment_intent(sess.get("payment_intent")) if sess.get("payment_intent") else {}
    if (pi.get("status") or sess.get("payment_status")) not in ("succeeded", "paid"):
        return "not_paid"
    today = _now_utc().date()
    cx = _sqlite3.connect(LOG_DB)
    try:
        cx.row_factory = _sqlite3.Row
        init_membership_tables(cx)
        # claim-then-create idempotency
        try:
            cx.execute("INSERT INTO membership_product_grants (session_id,email,tier,created_at) "
                       "VALUES (?,?,?,?)", (session_id, email, tier["key"], today.isoformat()))
            cx.commit()
        except _sqlite3.IntegrityError:
            return "already"
        days = _mp.grant_days(tier["key"], today)
        if tier["billing"] == "recurring_capped":
            customer, pm = pi.get("customer"), pi.get("payment_method")
            if not (customer and pm):
                # unwind the claim so a retry with a card can proceed
                cx.execute("DELETE FROM membership_product_grants WHERE session_id=?", (session_id,))
                cx.commit()
                return "no_card"
            if _subs.active_memberships_by_email(cx, email):
                _extend_membership_grant(cx, email, (today + _dt.timedelta(days=days)).isoformat(), tier["source"])
                return "duplicate_member"
            _subs.create_membership(
                cx, email=email, stripe_customer_id=customer, stripe_payment_method_id=pm,
                amount_cents=tier["price_cents"], next_charge_date=_subs.add_months(today, 1),
                cadence_months=tier["cadence_months"], term_charges_total=tier["term_charges"],
                initial_order_count=1)
            _grant_membership(cx, email, days, tier["source"])
        else:  # one_time (month, year_prepay)
            _grant_membership(cx, email, days, tier["source"])
            _book_membership_qbo(email, tier)  # see below
        return "ok"
    finally:
        cx.close()

def _book_membership_qbo(email, tier):
    """Record a paid QBO invoice for a one-time membership tier. Mirrors the QBO
    booking in _fulfill_prepay_term (app.py:8658-8805)."""
    try:
        from dashboard import qbo_billing as qb
        cust = qb.find_or_create_customer(email, "")
        inv = qb.create_invoice(cust, [{"name": tier["label"],
                                        "amount": tier["price_cents"] / 100.0, "qty": 1}],
                                allow_online_pay=False, email_to=email)
        qb.record_payment(cust.get("Id"), tier["price_cents"], inv.get("Id"), method="card")
    except Exception as e:
        print(f"[membership] QBO booking skipped for {email}/{tier['key']}: {e!r}", flush=True)
```

Add the return route near `/continuous-care/return`:

```python
@app.route("/membership/return", methods=["GET"])
def membership_return():
    sid = request.args.get("session_id", "")
    status = "err"
    if sid:
        try:
            status = "ok" if _fulfill_membership_product(sid) in ("ok", "duplicate_member", "already") else "err"
        except Exception as e:
            print(f"[membership] return fulfill error: {e!r}", flush=True)
    return redirect(f"/?membership={status}")
```

Add one line to the `/webhook/stripe` fan-out (`app.py:26628-26663`):

```python
_fulfill_membership_product(session_id)     # kind == "membership_product"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -c dev -- pytest tests/test_membership_products_fulfill.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Mutation-test the idempotency + no_card guards**

Comment out the `INSERT INTO membership_product_grants` claim; `test_idempotent_replay` must go RED (2 rows). Revert. Then force `with_card=False` to return before the `no_card` check; `test_year_monthly_requires_card` must go RED. Revert.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_membership_products_fulfill.py
git commit -m "feat(membership): fulfill membership-product checkout + webhook + return"
```

---

### Task 5: Owner manual-enroll endpoint (the Dana gap)

**Files:**
- Modify: `app.py` (new route near the other console endpoints, e.g. near `/api/console/client-invoice` `app.py:36677`)
- Test: `tests/test_membership_enroll_endpoint.py`

**Interfaces:**
- Consumes: `_bos_actor` / `_bos_rbac.OWNER`; `membership_products`; `_grant_membership`.
- Produces: `POST /api/console/membership/enroll` body `{"email": str, "tier": str, "source"?: str}` → `{"ok": True, "tier", "expires_at", "billing"}`. OWNER only. Grants the entitlement (member pricing + group access) WITHOUT a Stripe round-trip — payment is collected separately (e.g. a Payment Link). For `recurring_capped`, it grants the entitlement window only and returns `"note": "no auto-billing; bill monthly by hand or use /membership/checkout"` (no `subscriptions` row without a vaulted card).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membership_enroll_endpoint.py
import importlib, sys, os, sqlite3
import pytest

def _load_app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app import failed: {e}")

@pytest.fixture
def appmod(monkeypatch, tmp_path):
    app = _load_app()
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(app, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "sekret", raising=False)
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sekret", raising=False)
    cx = sqlite3.connect(db); app.init_membership_tables(cx); cx.close()
    if hasattr(app, "_member_join_welcome"):
        monkeypatch.setattr(app, "_member_join_welcome", lambda *a, **k: None, raising=False)
    return app

def test_owner_enroll_grants_member(appmod):
    r = appmod.app.test_client().post("/api/console/membership/enroll?key=sekret",
                                      json={"email": "dana@x.com", "tier": "month"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert appmod._is_paid_member("dana@x.com") is True

def test_enroll_requires_owner(appmod):
    r = appmod.app.test_client().post("/api/console/membership/enroll?key=wrong",
                                      json={"email": "dana@x.com", "tier": "month"})
    assert r.status_code == 401

def test_enroll_unknown_tier(appmod):
    r = appmod.app.test_client().post("/api/console/membership/enroll?key=sekret",
                                      json={"email": "dana@x.com", "tier": "nope"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -c dev -- pytest tests/test_membership_enroll_endpoint.py -v`
Expected: FAIL — route 404.

- [ ] **Step 3: Write minimal implementation**

```python
@app.route("/api/console/membership/enroll", methods=["POST"])
def console_membership_enroll():
    actor = _bos_actor()
    if actor is None or actor.role != _bos_rbac.OWNER:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import membership_products as _mp
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    tier = _mp.get_tier((data.get("tier") or "").strip())
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400
    if not tier:
        return jsonify({"ok": False, "error": "unknown tier"}), 400
    today = _now_utc().date()
    days = _mp.grant_days(tier["key"], today)
    src = (data.get("source") or tier["source"]).strip()
    cx = _sqlite3.connect(LOG_DB)
    try:
        init_membership_tables(cx)
        _grant_membership(cx, email, days, src)
    finally:
        cx.close()
    out = {"ok": True, "tier": tier["key"], "billing": tier["billing"],
           "expires_at": (today + _dt.timedelta(days=days)).isoformat()}
    if tier["billing"] == "recurring_capped":
        out["note"] = "no auto-billing; bill monthly by hand or use /membership/checkout"
    return jsonify(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -c dev -- pytest tests/test_membership_enroll_endpoint.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_membership_enroll_endpoint.py
git commit -m "feat(membership): owner console endpoint to enroll a member without Stripe"
```

---

### Task 6: "Choose your membership" surface + route (dark-launched)

**Files:**
- Create: `static/membership-choose.html`
- Modify: `app.py` (a `GET /membership` route serving the page, flag-gated)
- Test: `tests/test_membership_page_route.py`

**Interfaces:**
- Consumes: `MEMBERSHIP_PRODUCTS_ENABLED`; `POST /membership/checkout` (Task 3).
- Produces: `GET /membership` → serves the page when the flag is on, 404 when off. The page renders three tier cards from `membership_products.all_tiers()` values (hardcode the labels/prices to match the catalog), each with a "Choose" button that POSTs `{email, tier}` to `/membership/checkout` and redirects to `url`. Mirror the tier-card UX in `static/membership-admin.html:136-159`. Copy obeys the global rules (no em dashes, no ALL CAPS).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membership_page_route.py
import importlib, sys, os
import pytest

def _load_app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app import failed: {e}")

def test_page_404_when_flag_off(monkeypatch):
    app = _load_app()
    monkeypatch.setattr(app, "MEMBERSHIP_PRODUCTS_ENABLED", False, raising=False)
    assert app.app.test_client().get("/membership").status_code == 404

def test_page_served_when_flag_on(monkeypatch):
    app = _load_app()
    monkeypatch.setattr(app, "MEMBERSHIP_PRODUCTS_ENABLED", True, raising=False)
    r = app.app.test_client().get("/membership")
    assert r.status_code == 200
    assert b"Membership" in r.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -c dev -- pytest tests/test_membership_page_route.py -v`
Expected: FAIL — route 404 even when flag on.

- [ ] **Step 3: Write minimal implementation**

Create `static/membership-choose.html` (three tier cards; on click POST to `/membership/checkout` then `window.location = url`). Keep copy plain per the rules. Then add:

```python
@app.route("/membership", methods=["GET"])
def membership_choose_page():
    if not MEMBERSHIP_PRODUCTS_ENABLED:
        return ("Not found", 404)
    resp = send_from_directory(STATIC, "membership-choose.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -c dev -- pytest tests/test_membership_page_route.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Manual render check**

Run the app locally (`doppler run -c dev -- python app.py`), set the flag on, load `/membership` in headless Chrome, confirm three cards render with correct prices ($99 monthly, $99/mo annual, $990 full-pay) and the Choose buttons POST correctly. (See the render-verify rule — confirm the page, not just the payload.)

- [ ] **Step 6: Commit**

```bash
git add static/membership-choose.html app.py tests/test_membership_page_route.py
git commit -m "feat(membership): dark-launched choose-your-membership page"
```

---

### Task 7: Full-flow regression + dark-launch checklist

**Files:**
- Test: `tests/test_membership_products_e2e.py`
- Modify: `docs/` runbook note (optional)

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Write the end-to-end test (all three tiers checkout → fulfill → entitlement)**

```python
# tests/test_membership_products_e2e.py
# Reuse the appmod fixture pattern from test_membership_products_fulfill.py.
# For each tier: post /membership/checkout (capture metadata), then drive
# _fulfill_membership_product with a matching mocked session, assert
# _is_paid_member is True and membership_products.owns_group is True.
# For year_monthly additionally assert term_charges_total == 12.
```

- [ ] **Step 2: Run the full focused suite for the feature**

Run: `doppler run -c dev -- pytest tests/test_membership_products_catalog.py tests/test_membership_group_ownership.py tests/test_membership_products_checkout.py tests/test_membership_products_fulfill.py tests/test_membership_enroll_endpoint.py tests/test_membership_page_route.py tests/test_membership_products_e2e.py -v`
Expected: PASS (all)

- [ ] **Step 3: Confirm no regression in adjacent suites**

Run: `doppler run -c dev -- pytest tests/test_continuous_care_monthly.py tests/test_prepay_checkout.py tests/test_subscriptions_term_cap.py tests/test_paid_member_gate.py tests/test_portal_offers*.py -v`
Expected: PASS (member pricing + prepay + CC + portal offers unaffected)

- [ ] **Step 4: Commit**

```bash
git add tests/test_membership_products_e2e.py
git commit -m "test(membership): end-to-end coverage across all three tiers"
```

- [ ] **Step 5: Dark-launch checklist (do NOT flip the flag as part of this plan)**

  - Confirm `MEMBERSHIP_PRODUCTS_ENABLED` is unset in prod Doppler (`doppler -c prd`) so `/membership`, `/membership/checkout` stay 404 until Glen says go.
  - Open the PR with a generic body (no customer PII). Merge = deploy; the code ships dark.
  - Flip is a separate step: set `MEMBERSHIP_PRODUCTS_ENABLED=on` in prd Doppler + redeploy (flag-flip = its own deploy).

---

## Self-Review

**Spec coverage:**
- Tier A ($99, 1mo, no-renew) → catalog `month` (Task 1) + one-time grant-only fulfillment (Task 4). ✓
- Tier B ($99/mo ×12, no auto-renew, renewable) → catalog `year_monthly` + `create_membership(term_charges_total=12)`; self-cancels at cap via existing cron; renewal is manual re-signup (no auto-renew). ✓
- Tier C ($990, 1yr, no-renew) → catalog `year_prepay` + one-time grant-only. ✓
- Same entitlement for all three → member pricing via `_is_paid_member` (unchanged; all write `memberships` grant) + group ownership unified (Task 2). ✓
- Owner manual enroll (Dana gap) → Task 5. ✓
- Buyable surface → Task 6, dark-launched. ✓

**Placeholder scan:** Task 6's HTML body and Task 7's e2e test are described rather than fully written; both are UI/glue mirroring named existing files (`membership-admin.html`) and a named prior test fixture — acceptable, but the implementer should write them out in full at execution time. No TBD/TODO/"add error handling" placeholders elsewhere.

**Type consistency:** `get_tier`/`all_tiers`/`grant_days`/`owns_group` signatures are consistent across Tasks 1-6; tier dict keys (`key`, `label`, `price_cents`, `billing`, `source`, `term_charges`, `cadence_months`, `grant_months`) are used identically in Tasks 1, 3, 4, 5; `_fulfill_membership_product` return strings match between Task 4 impl and the `/membership/return` route.

**Open decisions to confirm with Glen before/at execution:**
1. Group-ownership broadening (Task 2) is deliberately narrow (only `membership_*` sources). Confirm prepay/CC/founding members should NOT retroactively gain "group coaching" labeling. (Recommended: keep narrow.)
2. QBO representation for one-time tiers (Task 4 `_book_membership_qbo`): invoice + immediate payment. Confirm vs. a SalesReceipt (no SalesReceipt helper exists today; invoice+payment is the closest existing pattern).
3. Whether the recurring-capped tier B should also be sellable via the owner enroll endpoint (Task 5 currently grants the window only, no auto-billing). 

---

## Execution Handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, two-stage review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.
