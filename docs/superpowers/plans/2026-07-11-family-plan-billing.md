# Family Plan Billing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add self-serve recurring billing to the existing Family Plan so a caregiver can buy the $147/mo plan from their portal (card on file, month-1 charged now), a monthly cron recharges it, and the caregiver can cancel — without changing the `covers()` entitlement.

**Architecture:** Mirror the shipped coach subscription 1:1 (`dashboard/coach_subscriptions.py` + `/api/community/coach-subscribe` + `_fulfill_coach_sub` + `/api/cron/coach-subscriptions/charge`). The `family_subscriptions` table + `activate()`/`covers()` already exist; this adds the missing billing helpers, a portal-initiated Stripe checkout with idempotent fulfillment (return route + webhook), a `due()`-driven monthly charge cron with bounded dunning, and a self-cancel route. No per-cycle service grant — a successful charge just keeps `status='active'`, which `covers()` reads.

**Tech Stack:** Python/Flask, sqlite (`LOG_DB`), `dashboard/stripe_pay.py` (hosted Checkout + off-session charges), pytest.

## Global Constraints

- **Prices come from `family_plan.PLAN`** — `amount_cents=14700` ($147), `value_cents=19700` ($197). Never hardcode elsewhere.
- **Comp plans are never charged** — a `source='comp'` / `next_charge_at IS NULL` row must never appear in `due()` or be charged.
- **First charge once** — fulfillment is idempotent via a `family_sub_grants(session_id)` PRIMARY KEY claim (webhook + return double-delivery safe).
- **No mass-charge on first cron** — signup seeds `next_charge_at = add_months(today, 1)`.
- **No cron double-charge** — `mark_charged` advances `next_charge_at` only after a successful charge, in the same locked write; the cron only charges `next_charge_at <= today`.
- **Cron auth** — `X-Console-Key` header must equal `CONSOLE_SECRET` (mirrors the coach cron).
- **Route gating** — the subscribe route 404s when `_family_plan_enabled()` is off and 503s when `_STRIPE_ACTIVE` is false.
- **Statuses** — `active` / `past_due` / `cancelled` (British spelling, matching the existing console cancel + `ACTIVE_STATUSES`). `past_due` still entitles the household (grace) AND is retried by the cron every run; grace is bounded — after 3 total failures the plan is `cancelled` and cover stops. (Because the cron is monthly, worst-case grace on a dead card is ~3 cron cycles.)
- **Copy rules** (client-portal.html): no em dashes, no ALL CAPS.
- **Card capture is Stripe-hosted** — store only `stripe_customer_id` + `payment_method_id`.

---

### Task 1: Store billing helpers (`dashboard/family_plan.py`)

**Files:**
- Modify: `dashboard/family_plan.py` (add a charges table to the DDL + four functions)
- Test: `tests/test_family_plan_billing_store.py` (create)

**Interfaces:**
- Consumes: existing `family_plan.PLAN`, `activate(cx, caregiver_email, *, next_charge_at, customer_id=None, payment_method_id=None, source="stripe")`, `get(cx, caregiver_email)`, `set_status(cx, caregiver_email, status)`, `init_family_plan_table(cx)`.
- Produces:
  - `due(cx, today) -> list[dict]` — active, non-comp subs with `next_charge_at <= today`.
  - `mark_charged(cx, caregiver_email, next_charge_at)` — advance date, reset fail_count, set active, stamp last_charged_at.
  - `mark_failed(cx, caregiver_email)` — increment fail_count, set past_due.
  - `record_charge(cx, *, caregiver_email, amount_cents, pi_id, status)` — insert a `family_sub_charges` ledger row.

- [ ] **Step 1: Write the failing test**

Create `tests/test_family_plan_billing_store.py`:

```python
import sqlite3
from dashboard import family_plan as fp


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    fp.init_family_plan_table(cx)
    return cx


def test_due_excludes_future_cancelled_and_comp():
    cx = _cx()
    fp.activate(cx, "due@x.com", next_charge_at="2026-07-01",
                customer_id="c", payment_method_id="p")            # past -> due
    fp.activate(cx, "future@x.com", next_charge_at="2026-12-01",
                customer_id="c", payment_method_id="p")            # future -> not due
    fp.activate(cx, "cxl@x.com", next_charge_at="2026-07-01",
                customer_id="c", payment_method_id="p")
    fp.set_status(cx, "cxl@x.com", "cancelled")                    # cancelled -> not due
    fp.activate(cx, "pastdue@x.com", next_charge_at="2026-07-02",
                customer_id="c", payment_method_id="p")
    fp.set_status(cx, "pastdue@x.com", "past_due")                 # past_due -> RETRIED (still due)
    fp.activate(cx, "comp@x.com", next_charge_at=None, source="comp")  # comp -> never due
    emails = [d["caregiver_email"] for d in fp.due(cx, "2026-07-15")]
    assert emails == ["due@x.com", "pastdue@x.com"]                # active + past_due, ordered by date


def test_mark_charged_advances_and_resets():
    cx = _cx()
    fp.activate(cx, "m@x.com", next_charge_at="2026-07-01",
                customer_id="c", payment_method_id="p")
    fp.mark_failed(cx, "m@x.com")
    assert fp.get(cx, "m@x.com")["fail_count"] == 1
    assert fp.get(cx, "m@x.com")["status"] == "past_due"
    fp.mark_charged(cx, "m@x.com", "2026-08-01")
    s = fp.get(cx, "m@x.com")
    assert s["next_charge_at"] == "2026-08-01" and s["fail_count"] == 0
    assert s["status"] == "active" and s["last_charged_at"]


def test_record_charge_ledger():
    cx = _cx()
    fp.record_charge(cx, caregiver_email="m@x.com", amount_cents=14700,
                     pi_id="pi_1", status="succeeded")
    row = cx.execute("SELECT * FROM family_sub_charges").fetchone()
    assert row["pi_id"] == "pi_1" and row["amount_cents"] == 14700
    assert row["caregiver_email"] == "m@x.com" and row["status"] == "succeeded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan_billing_store.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.family_plan' has no attribute 'due'` (and no `family_sub_charges` table).

- [ ] **Step 3: Add the charges table to the DDL**

In `dashboard/family_plan.py`, extend `_DDL` (append inside the same string, after the `family_subscriptions` block + its index):

```python
CREATE TABLE IF NOT EXISTS family_sub_charges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caregiver_email TEXT,
    amount_cents INTEGER,
    pi_id TEXT,
    status TEXT,
    charged_at TEXT
);
```

- [ ] **Step 4: Add the four functions**

Append to `dashboard/family_plan.py` (after `covers`):

```python
def due(cx, today):
    """Billable subs whose next_charge_at has arrived. Includes both 'active' and
    'past_due': a past_due sub (a prior failed charge) still entitles the household
    (grace) and MUST be retried on each cron run so its fail_count can climb to the
    cancel threshold — otherwise a single failed payment would cover forever.
    Comped plans (source='comp', next_charge_at NULL) are never billable, excluded."""
    rows = cx.execute(
        "SELECT * FROM family_subscriptions WHERE status IN ('active','past_due') "
        "AND next_charge_at IS NOT NULL AND next_charge_at <= ? "
        "AND (source IS NULL OR source != 'comp') ORDER BY next_charge_at",
        (today,)).fetchall()
    return [dict(r) for r in rows]


def mark_charged(cx, caregiver_email, next_charge_at):
    cx.execute("UPDATE family_subscriptions SET next_charge_at=?, last_charged_at=?, "
               "fail_count=0, status='active' WHERE caregiver_email=?",
               (next_charge_at, _now(), _lc(caregiver_email)))
    cx.commit()


def mark_failed(cx, caregiver_email):
    cx.execute("UPDATE family_subscriptions SET fail_count=fail_count+1, status='past_due' "
               "WHERE caregiver_email=?", (_lc(caregiver_email),))
    cx.commit()


def record_charge(cx, *, caregiver_email, amount_cents, pi_id, status):
    cx.execute("INSERT INTO family_sub_charges (caregiver_email,amount_cents,pi_id,status,charged_at) "
               "VALUES (?,?,?,?,?)", (_lc(caregiver_email), amount_cents, pi_id, status, _now()))
    cx.commit()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan_billing_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Regression — existing family plan store tests still pass**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan.py -v`
Expected: PASS (the entitlement tests are untouched; the new table is additive).

- [ ] **Step 7: Commit**

```bash
git add dashboard/family_plan.py tests/test_family_plan_billing_store.py
git commit -m "feat(family-plan): billing store helpers (due/mark_charged/mark_failed/record_charge + charges ledger)"
```

---

### Task 2: Subscribe + idempotent fulfillment (`app.py`)

**Files:**
- Modify: `app.py` (add the subscribe route, `_fulfill_family_plan`, the `/family-plan/return` route, and one line in `webhook_stripe`)
- Test: `tests/test_family_plan_subscribe_api.py` (create)

**Interfaces:**
- Consumes: `family_plan.PLAN`, `family_plan.activate`, `family_plan.record_charge`, `family_plan.init_family_plan_table` (Task 1); `stripe_pay.create_checkout_session(amount_cents, *, customer_email, description, metadata, success_url, cancel_url, save_card=True)`, `stripe_pay.get_session`, `stripe_pay.get_payment_intent`; `portal_identity.resolve_identity`; `subscriptions.add_months`; `_db_lock`, `LOG_DB`, `PUBLIC_BASE_URL`, `_STRIPE_ACTIVE`, `_family_plan_enabled`, `_client_login_enabled`, `send_evox_email`.
- Produces: `POST /api/portal/<token>/family-plan/subscribe` -> `{ok, url}`; `_fulfill_family_plan(session_id) -> dict`; `GET /family-plan/return`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_family_plan_subscribe_api.py`:

```python
import sqlite3
from unittest import mock
import app as appmod
from dashboard import family_plan as fp


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _tok(email="care@x.com", name="Care"):
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx); fp.init_family_plan_table(cx)
        token, _ = cp.upsert_portal(cx, email, name, {}); cx.commit()
    return token


def test_subscribe_returns_checkout_url():
    tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True), \
         mock.patch.object(appmod, "_family_plan_enabled", return_value=True), \
         mock.patch("dashboard.stripe_pay.create_checkout_session",
                    return_value={"id": "cs_1", "url": "https://stripe/cs_1"}):
        r = _client().post(f"/api/portal/{tok}/family-plan/subscribe")
    assert r.status_code == 200 and r.get_json()["url"] == "https://stripe/cs_1"


def test_subscribe_flag_off_404():
    tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True), \
         mock.patch.object(appmod, "_family_plan_enabled", return_value=False):
        r = _client().post(f"/api/portal/{tok}/family-plan/subscribe")
    assert r.status_code == 404


def test_subscribe_stripe_inactive_503():
    tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", False), \
         mock.patch.object(appmod, "_family_plan_enabled", return_value=True):
        r = _client().post(f"/api/portal/{tok}/family-plan/subscribe")
    assert r.status_code == 503


def test_fulfill_activates_and_charges_once():
    _tok("f@x.com")
    fake_session = {"metadata": {"kind": "family_plan", "email": "f@x.com"},
                    "payment_intent": "pi_1"}
    fake_pi = {"status": "succeeded", "customer": "cus_1", "payment_method": "pm_1"}
    with mock.patch("dashboard.stripe_pay.get_session", return_value=fake_session), \
         mock.patch("dashboard.stripe_pay.get_payment_intent", return_value=fake_pi), \
         mock.patch.object(appmod, "send_evox_email"):
        appmod._fulfill_family_plan("cs_evt_1")
        appmod._fulfill_family_plan("cs_evt_1")   # webhook + return double-delivery
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "f@x.com")
        assert s["status"] == "active" and s["stripe_customer_id"] == "cus_1"
        assert s["next_charge_at"] and s["source"] == "stripe"
        n = cx.execute("SELECT COUNT(*) FROM family_sub_charges "
                       "WHERE caregiver_email='f@x.com'").fetchone()[0]
    assert n == 1                                  # charged exactly once


def test_fulfill_ignores_unpaid():
    with mock.patch("dashboard.stripe_pay.get_session",
                    return_value={"metadata": {"kind": "family_plan", "email": "u@x.com"},
                                  "payment_intent": "pi_x"}), \
         mock.patch("dashboard.stripe_pay.get_payment_intent",
                    return_value={"status": "requires_payment_method",
                                  "customer": None, "payment_method": None}):
        appmod._fulfill_family_plan("cs_evt_2")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        assert fp.get(cx, "u@x.com") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan_subscribe_api.py -v`
Expected: FAIL — 404 on the subscribe route (not registered) / `AttributeError: _fulfill_family_plan`.

- [ ] **Step 3: Add the subscribe route + fulfillment + return route**

In `app.py`, add near the coach subscription routes (after `_fulfill_coach_sub` / `coach_subscribe_return`, ~line 19192):

```python
@app.route("/api/portal/<token>/family-plan/subscribe", methods=["POST"])
def portal_family_plan_subscribe(token):
    """Start a Stripe Checkout for the $147/mo Family Plan (caregiver-initiated
    from their portal). Vaults the card; month 1 is charged now, months 2..N by
    the monthly cron off the vaulted card. Entitlement is covers() (unchanged)."""
    if not _family_plan_enabled():
        return jsonify({"error": "not_found"}), 404
    if not _STRIPE_ACTIVE:
        return jsonify({"error": "unavailable"}), 503
    from dashboard import portal_identity as _pi, family_plan as _fp
    sess_cookie = request.cookies.get("rm_portal_session", "")
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        ident = _pi.resolve_identity(cx, token=token, session_token=sess_cookie,
                                     client_login_enabled=_client_login_enabled())
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        email = ident.email
    from dashboard import stripe_pay as _sp
    base = PUBLIC_BASE_URL.rstrip("/")
    sess = _sp.create_checkout_session(
        _fp.PLAN["amount_cents"], customer_email=email, description=_fp.PLAN["label"],
        metadata={"kind": "family_plan", "email": email},
        success_url=f"{base}/family-plan/return?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/portal/{token}", save_card=True)
    return jsonify({"ok": True, "url": sess.get("url")})


def _fulfill_family_plan(session_id):
    """Activate a caregiver's paid Family Plan from a paid+vaulted checkout,
    idempotently (claim-then-activate on family_sub_grants(session_id) PRIMARY
    KEY). Callable from the /family-plan/return redirect AND the webhook so a
    closed tab still gets fulfilled. Re-fetches the session + PaymentIntent;
    only proceeds on a succeeded payment WITH a vaulted customer + method.
    Never raises."""
    try:
        from dashboard import stripe_pay as _sp, family_plan as _fp, subscriptions as _subs
        sess = _sp.get_session(session_id)
        md = sess.get("metadata") or {}
        if md.get("kind") != "family_plan":
            return {"ok": False, "reason": "not_family_plan"}
        email = (md.get("email") or "").strip().lower()
        pi_id = sess.get("payment_intent")
        if not (email and pi_id):
            return {"ok": False, "reason": "incomplete"}
        pi = _sp.get_payment_intent(pi_id)
        if pi.get("status") != "succeeded":
            return {"ok": False, "reason": "unpaid"}
        customer, pm = pi.get("customer"), pi.get("payment_method")
        if not (customer and pm):
            return {"ok": False, "reason": "no_card"}
        from datetime import date as _date
        next_charge = _subs.add_months(_date.today().isoformat(), 1)
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            cx.execute("CREATE TABLE IF NOT EXISTS family_sub_grants "
                       "(session_id TEXT PRIMARY KEY, email TEXT, created_at TEXT)")
            _fp.init_family_plan_table(cx)
            claimed = cx.execute(
                "INSERT OR IGNORE INTO family_sub_grants (session_id,email,created_at) VALUES (?,?,?)",
                (session_id, email, _fp._now())).rowcount == 1
            cx.commit()
            if not claimed:
                return {"ok": True, "reason": "already_fulfilled"}
            _fp.activate(cx, email, next_charge_at=next_charge, customer_id=customer,
                         payment_method_id=pm, source="stripe")
            _fp.record_charge(cx, caregiver_email=email,
                              amount_cents=_fp.PLAN["amount_cents"], pi_id=pi_id,
                              status="succeeded")
        try:
            html = ("<p>Your Family Plan is active. Everyone in your household with "
                    "sharing on now has their full analysis unlocked. You can cancel "
                    "any time from your portal.</p>")
            send_evox_email(email, "", "Your Family Plan is active", html, html, b"")
        except Exception:
            app.logger.exception("family plan confirmation failed")
        return {"ok": True}
    except Exception:
        app.logger.exception("family plan fulfill failed for %s", session_id)
        return {"ok": False, "reason": "error"}


@app.route("/family-plan/return")
def family_plan_return():
    sid = request.args.get("session_id", "")
    if sid:
        _fulfill_family_plan(sid)
    return redirect(f"{PUBLIC_BASE_URL.rstrip('/')}/")
```

- [ ] **Step 4: Wire the webhook dispatch**

In `app.py`, in `webhook_stripe` (~line 24677), add `_fulfill_family_plan(session_id)` to the fulfiller sequence, right after `_fulfill_coach_sub(session_id)`:

```python
                _fulfill_coach_sub(session_id)
                _fulfill_family_plan(session_id)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan_subscribe_api.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_family_plan_subscribe_api.py
git commit -m "feat(family-plan): portal-initiated Stripe checkout + idempotent fulfillment + webhook"
```

---

### Task 3: Self-cancel route (`app.py`)

**Files:**
- Modify: `app.py` (add the cancel route)
- Test: `tests/test_family_plan_subscribe_api.py` (append one test)

**Interfaces:**
- Consumes: `family_plan.set_status`, `family_plan.activate`, `family_plan.get`, `portal_identity.resolve_identity`, `_db_lock`.
- Produces: `POST /api/portal/<token>/family-plan/cancel` -> `{ok}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_family_plan_subscribe_api.py`:

```python
def test_cancel_sets_cancelled_and_stops_cover():
    tok = _tok("cxl2@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        fp.activate(cx, "cxl2@x.com", next_charge_at="2026-08-01",
                    customer_id="c", payment_method_id="p"); cx.commit()
    r = _client().post(f"/api/portal/{tok}/family-plan/cancel")
    assert r.get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        assert fp.get(cx, "cxl2@x.com")["status"] == "cancelled"
        assert fp.is_active(cx, "cxl2@x.com") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan_subscribe_api.py::test_cancel_sets_cancelled_and_stops_cover -v`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Add the cancel route**

In `app.py`, after `family_plan_return` (Task 2):

```python
@app.route("/api/portal/<token>/family-plan/cancel", methods=["POST"])
def portal_family_plan_cancel(token):
    """Caregiver self-cancel. Stops covers() for the household at the next read;
    no refund, no proration (the current paid cycle simply is not renewed)."""
    from dashboard import portal_identity as _pi, family_plan as _fp
    sess_cookie = request.cookies.get("rm_portal_session", "")
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _fp.init_family_plan_table(cx)
        ident = _pi.resolve_identity(cx, token=token, session_token=sess_cookie,
                                     client_login_enabled=_client_login_enabled())
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        _fp.set_status(cx, ident.email, "cancelled")
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan_subscribe_api.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_family_plan_subscribe_api.py
git commit -m "feat(family-plan): caregiver self-cancel route"
```

---

### Task 4: Monthly charge cron + bounded dunning (`app.py`)

**Files:**
- Modify: `app.py` (add the charge cron route)
- Create: `scripts/run_family_plan_cron.py` (cron entry, mirrors `run_subscriptions_cron.py`)
- Test: `tests/test_family_plan_charge_cron.py` (create)

**Interfaces:**
- Consumes: `family_plan.due`, `family_plan.record_charge`, `family_plan.mark_charged`, `family_plan.mark_failed`, `family_plan.set_status`, `family_plan.get`, `family_plan.PLAN`; `stripe_pay.charge_off_session`; `subscriptions.add_months`; `_db_lock`, `LOG_DB`, `CONSOLE_SECRET`, `GLEN_CONSULT_EMAIL`, `send_evox_email`.
- Produces: `POST /api/cron/family-plan/charge` -> `{charged, failed, cancelled}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_family_plan_charge_cron.py`:

```python
import sqlite3
from unittest import mock
import requests
import app as appmod
from dashboard import family_plan as fp


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(email, next_at, *, source="stripe", fail_count=0, status="active"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        fp.activate(cx, email, next_charge_at=next_at, customer_id="cus",
                    payment_method_id="pm", source=source)
        if fail_count or status != "active":
            cx.execute("UPDATE family_subscriptions SET fail_count=?, status=? "
                       "WHERE caregiver_email=?", (fail_count, status, email.lower()))
        cx.commit()


def _hdr():
    return {"X-Console-Key": appmod.CONSOLE_SECRET}


def test_cron_requires_key():
    assert _client().post("/api/cron/family-plan/charge").status_code == 401


def test_cron_charges_due_and_advances():
    _seed("due@x.com", "2026-01-01")
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": "pi_ok", "status": "succeeded"}):
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["charged"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "due@x.com")
        assert s["next_charge_at"] > "2026-01-01" and s["last_charged_at"]


def test_cron_skips_future():
    _seed("future@x.com", "2099-01-01")
    with mock.patch("dashboard.stripe_pay.charge_off_session") as charge:
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["charged"] == 0 and not charge.called


def test_cron_never_charges_a_comp():
    _seed("comp@x.com", None, source="comp")
    with mock.patch("dashboard.stripe_pay.charge_off_session") as charge:
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["charged"] == 0 and not charge.called


def test_cron_failed_charge_goes_past_due_no_advance():
    _seed("fail@x.com", "2026-01-01")
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": None, "status": "failed"}), \
         mock.patch.object(appmod, "send_evox_email"):
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["failed"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "fail@x.com")
        assert s["status"] == "past_due" and s["fail_count"] == 1
        assert s["next_charge_at"] == "2026-01-01"


def test_cron_third_failure_cancels_and_stops_cover():
    _seed("dead@x.com", "2026-01-01", fail_count=2)   # already failed twice
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": None, "status": "failed"}), \
         mock.patch.object(appmod, "send_evox_email"):
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["failed"] == 1 and d["cancelled"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "dead@x.com")
        assert s["status"] == "cancelled" and fp.is_active(cx, "dead@x.com") is False


def test_cron_retries_a_past_due_sub_and_recovers_on_success():
    _seed("grace@x.com", "2026-01-01", fail_count=1, status="past_due")  # prior failure, in grace
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": "pi_ok", "status": "succeeded"}):
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["charged"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "grace@x.com")
        assert s["status"] == "active" and s["fail_count"] == 0      # recovered
        assert s["next_charge_at"] > "2026-01-01"                    # advanced


def test_cron_one_exception_does_not_abort_batch():
    _seed("boom@x.com", "2026-01-01")     # due first, raises
    _seed("ok@x.com", "2026-01-02")       # due second, succeeds
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    side_effect=[requests.Timeout("timed out"),
                                 {"id": "pi_ok2", "status": "succeeded"}]), \
         mock.patch.object(appmod, "send_evox_email"):
        resp = _client().post("/api/cron/family-plan/charge", headers=_hdr())
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["charged"] == 1 and d["failed"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        assert fp.get(cx, "boom@x.com")["next_charge_at"] == "2026-01-01"   # not advanced
        assert fp.get(cx, "ok@x.com")["next_charge_at"] > "2026-01-02"      # advanced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan_charge_cron.py -v`
Expected: FAIL — 404 (cron route not registered) on every test except `test_cron_requires_key`.

- [ ] **Step 3: Add the charge cron route**

In `app.py`, after the coach charge cron (`coach_subscriptions_charge_cron`, ~line 19256):

```python
@app.route("/api/cron/family-plan/charge", methods=["POST"])
def family_plan_charge_cron():
    """Monthly charge cron for paid Family Plans. Charges each due, non-comp,
    active plan off the vaulted card. On success: record + advance next_charge_at
    one month (the only way the date moves, so a same-day re-run cannot
    double-charge). On failure: record, mark past_due, notify; after 3 consecutive
    failures cancel the plan (past_due still entitles mid-cycle, so grace is
    bounded). Comped plans (next_charge_at NULL / source='comp') are never due."""
    if request.headers.get("X-Console-Key") != CONSOLE_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import family_plan as _fp, stripe_pay as _sp, subscriptions as _subs
    from datetime import date as _date
    today = _date.today().isoformat()
    charged = failed = cancelled = 0
    with sqlite3.connect(LOG_DB) as rcx:
        rcx.row_factory = sqlite3.Row
        _fp.init_family_plan_table(rcx)
        due_rows = _fp.due(rcx, today)
    for sub in due_rows:
        email = sub["caregiver_email"]
        try:
            res = _sp.charge_off_session(
                sub["stripe_customer_id"], sub["payment_method_id"], sub["amount_cents"],
                description=_fp.PLAN["label"],
                metadata={"kind": "family_plan_cycle", "email": email})
            ok = res.get("status") == "succeeded"
            pi_id = res.get("id") or ""
        except Exception:
            app.logger.exception("family plan charge raised for %s", email)
            ok = False
            pi_id = ""
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _fp.init_family_plan_table(cx)
            _fp.record_charge(cx, caregiver_email=email, amount_cents=sub["amount_cents"],
                              pi_id=pi_id, status="succeeded" if ok else "failed")
            if ok:
                _fp.mark_charged(cx, email, _subs.add_months(today, 1))
            else:
                _fp.mark_failed(cx, email)
                row = _fp.get(cx, email)
                if row and int(row.get("fail_count") or 0) >= 3:
                    _fp.set_status(cx, email, "cancelled")
                    was_cancelled = True
                else:
                    was_cancelled = False
        if ok:
            charged += 1
        else:
            failed += 1
            if was_cancelled:
                cancelled += 1
            html = (f"<p>We could not process this month's Family Plan charge for {email}. "
                    f"Please update the card on file from your portal to keep the plan active.</p>")
            try:
                send_evox_email(email, "", "Your Family Plan payment did not go through",
                                html, html, b"")
            except Exception:
                app.logger.exception("family plan failure notify (member) failed for %s", email)
            try:
                send_evox_email(GLEN_CONSULT_EMAIL, "Glen", f"Family Plan charge failed: {email}",
                                html, html, b"")
            except Exception:
                app.logger.exception("family plan failure notify (glen) failed for %s", email)
    return jsonify({"charged": charged, "failed": failed, "cancelled": cancelled})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan_charge_cron.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Create the cron entry script**

Create `scripts/run_family_plan_cron.py` (mirrors `run_subscriptions_cron.py`; posts from the cron container to the web endpoint where `LOG_DB` lives):

```python
#!/usr/bin/env python3
"""Monthly Family Plan charge cron entry point.

Posts to the web service's /api/cron/family-plan/charge endpoint, which runs the
charge scheduler inside the web container (where the persistent disk + chat_log.db
live). Render cron containers do NOT share the web service's disk, so the scheduler
cannot run here directly.

Required env vars on the cron service:
  WEB_URL       — base URL of the web service (no trailing slash)
                  default: https://glen-knowledge-chat.onrender.com
  CONSOLE_SECRET — shared secret matching the web service's CONSOLE_SECRET
                   (sent as X-Console-Key)
"""
import os
import sys
import json
import urllib.request
import urllib.error

WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
KEY = os.environ.get("CONSOLE_SECRET", "")

if not KEY:
    print("ERROR: CONSOLE_SECRET not set on cron service", flush=True)
    sys.exit(1)


def main():
    url = f"{WEB_URL}/api/cron/family-plan/charge"
    req = urllib.request.Request(
        url, method="POST", data=b"{}",
        headers={"X-Console-Key": KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}: {body}", flush=True)
            data = json.loads(body)
            print(f"Family plan cron: charged={data.get('charged', 0)} "
                  f"failed={data.get('failed', 0)} cancelled={data.get('cancelled', 0)}",
                  flush=True)
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code}: {e.read().decode('utf-8', errors='replace')}", flush=True)
        sys.exit(4)
    except urllib.error.URLError as e:
        print(f"URLError: {e}", flush=True)
        sys.exit(5)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add app.py scripts/run_family_plan_cron.py tests/test_family_plan_charge_cron.py
git commit -m "feat(family-plan): monthly charge cron with bounded dunning + cron entry script"
```

**Ops note (not a code step):** a Render `cron_job` must invoke `python scripts/run_family_plan_cron.py` monthly (e.g. `0 17 1 * *`), with `CONSOLE_SECRET` + `WEB_URL` set — mirroring the coach cron's Render wiring. This is a Render dashboard change for Glen/ops, not part of the merge.

---

### Task 5: Caregiver portal UI — Subscribe + Cancel (`static/client-portal.html`)

**Files:**
- Modify: `static/client-portal.html` (the Family Plan block, ~line 992-1001, plus a small handler)

**Interfaces:**
- Consumes: the existing `o.family_plan` payload block (`{price_cents, value_cents, label, active}`) and the page's `token` / `seg` variable used by other portal POSTs.
- Produces: a **Subscribe** button (inactive state) and a **Cancel** link (active state).

- [ ] **Step 1: Replace the "Just reply to arrange it" wiring with buttons**

In `static/client-portal.html`, in the Family Plan block, replace the inactive branch (currently ending `... Just reply to arrange it.</p>`) and add a Cancel affordance to the active branch. Keep prices from the payload; no em dashes, no ALL CAPS:

```javascript
    const fpl = o.family_plan;
    if(fpl && fpl.active){
      html += `<p><strong>${esc(fpl.label)}</strong> — you're on it. Everyone in your household with sharing on has their full analysis unlocked. <button type="button" class="linklike" data-fp-cancel="1">Cancel plan</button></p>`;
    } else if(fpl){
      const fpPrice = "$" + Math.round(fpl.price_cents/100);
      const fpValue = "$" + Math.round(fpl.value_cents/100);
      html += `<p><strong>${esc(fpl.label)}</strong> — full analysis for everyone in your household. <strong>${fpPrice}/month</strong> (a ${fpValue}/month value). <button type="button" class="btn" data-fp-subscribe="1">Start the Family Plan</button></p>`;
    } else {
      html += `<p class="muted">No monthly subscription, and nothing you need to sign up for.</p>`;
    }
```

Note: the active-state copy is trimmed to "full analysis for everyone in your household" (what `covers()` actually delivers) — the four-benefit copy is held until those benefits ship (see spec go-live gate).

- [ ] **Step 2: Add the click handlers**

Near the other portal button handlers (where `seg`/`token` is in scope), add a delegated handler:

```javascript
document.addEventListener("click", async (e) => {
  const sub = e.target.closest("[data-fp-subscribe]");
  const cxl = e.target.closest("[data-fp-cancel]");
  if(sub){
    sub.disabled = true;
    try{
      const r = await fetch(`/api/portal/${encodeURIComponent(seg)}/family-plan/subscribe`,
                            {method:"POST", credentials:"same-origin"});
      const j = await r.json();
      if(j && j.url){ window.location.href = j.url; } else { sub.disabled = false; }
    }catch(_){ sub.disabled = false; }
  } else if(cxl){
    if(!window.confirm("Cancel the Family Plan? Household members lose their unlocked analysis at the end of the cycle.")) return;
    cxl.disabled = true;
    try{
      await fetch(`/api/portal/${encodeURIComponent(seg)}/family-plan/cancel`,
                  {method:"POST", credentials:"same-origin"});
      window.location.reload();
    }catch(_){ cxl.disabled = false; }
  }
});
```

(Use the same token variable the surrounding code uses — `seg` in the render scope, or `token` if that is what is in scope at the handler site. Match the neighboring `fetch("/api/portal/" + encodeURIComponent(seg) + ...)` calls.)

- [ ] **Step 3: Render-verify locally (no JS unit harness in this repo)**

Serve the app against prod data on a spare port and render a caregiver's portal in headless Chrome to confirm the Subscribe button shows for a caregiver without a plan, and the Cancel affordance shows when `family_plan.active` is true (per [[reference_portal_render_verify]]). Confirm the button POSTs to `/api/portal/<token>/family-plan/subscribe` (Network tab) and that no console errors fire. This is a manual verification step — there is no JS test runner here; the route behaviour itself is covered by Tasks 2-4.

- [ ] **Step 4: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(family-plan): portal Subscribe + Cancel wiring (analysis-only copy)"
```

---

### Task 6: Full-suite regression + branch check

- [ ] **Step 1: Run the family + subscription + portal suites**

Run: `cd ~/deploy-chat && python -m pytest tests/test_family_plan.py tests/test_family_plan_api.py tests/test_family_plan_billing_store.py tests/test_family_plan_subscribe_api.py tests/test_family_plan_charge_cron.py tests/test_portal_family_unlock.py tests/test_portal_options_family_plan.py tests/test_coach_subscribe_api.py tests/test_coach_sub_cron.py -v`
Expected: PASS (coach suites confirm the shared webhook/cron changes did not regress the sibling).

- [ ] **Step 2: Diff the FAILED set against main's baseline**

Per [[feedback_suite_green_not_task_green]], run the broader suite and compare the sorted FAILED list to main's baseline (not counts). Any NEW failure is this branch's; a pre-existing failure on main is not.

Run: `cd ~/deploy-chat && python -m pytest -q 2>&1 | tail -30`

- [ ] **Step 3: Re-fetch origin/main and confirm no drift**

Per [[feedback_refetch_main_mid_session]], before opening the PR: `git fetch origin && git log --oneline origin/main -1` and rebase/merge if main advanced.

---

## Self-Review

**Spec coverage:**
- Store additions (`due`/`mark_charged`/`mark_failed`/`record_charge` + `family_sub_charges`) → Task 1. ✓
- Subscribe route + idempotent fulfillment (return + webhook) → Task 2. ✓
- Cancel → Task 3. ✓
- Monthly charge cron + bounded dunning + comp-never-charged → Task 4. ✓
- Caregiver portal surface → Task 5. ✓
- Money-path invariants (first-charge-once, no mass-charge, no double-charge, comp excluded, dunning) → asserted in Task 1/2/4 tests. ✓
- Go-live copy gate → Task 5 trims the active-state copy to delivered value; the full copy pass + benefit wiring stay deferred (spec). ✓
- Karin comp → operational, not a code task (called out in the spec + handoff below). ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every test step shows the assertions.

**Type consistency:** `caregiver_email` used consistently (not `member_email`, which is the coach column); `record_charge` has no `tier` (family is single-tier); statuses `active`/`past_due`/`cancelled` consistent across store, cron, and tests; `PLAN["amount_cents"]`/`PLAN["label"]` used verbatim; `_fp.get` (not `get_sub`) matches the existing family module.

---

## Handoff notes

- **Karin (operational, do now, independent of this build):** `POST /api/console/family-plan {caregiver_email: "permanentlyyours@hawaii.rr.com", source: "comp"}` so her household is covered by one comped entitlement instead of per-member `mark-paid`.
- **Go-live gate (blocking real charges):** do not point real caregivers at the Subscribe button until the plan copy matches delivered value. Task 5 trims the portal copy to the analysis-only benefit; the four-benefit marketing copy and the three ancillary benefits (family shipment, member pricing, group coaching) stay deferred. Test-mode Stripe validation is fine before then.
- **Render cron_job** for `scripts/run_family_plan_cron.py` is a dashboard change (Task 4 ops note), not part of the merge.
