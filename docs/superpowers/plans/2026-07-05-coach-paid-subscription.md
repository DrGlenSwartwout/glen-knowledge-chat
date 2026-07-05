# Paid Coaching Subscription (coaching arc, slice 2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Month-to-month paid coaching: a member subscribes to Rae ($100/mo, one EVOX credit per cycle) or Dr. Glen ($200/mo, one Causal Biofield entitlement per cycle); card on file charges month 1 at signup and a monthly cron recharges + regrants; cancel anytime keeps the current cycle.

**Architecture:** Mirror the existing **Continuous Care monthly** pattern (proven, in-tree). Subscribe = `create_checkout_session(amount, save_card=True)` — Stripe's hosted page charges month 1 AND vaults the card. A `_fulfill_coach_sub(session_id)` webhook/return handler (claim-then-create idempotency on `coach_sub_grants(session_id)`) verifies the succeeded PaymentIntent, creates a `coach_subscriptions` row (`next_charge_at = today + 1 month`), and grants the cycle service. A monthly cron charges due rows off the vaulted card and regrants. New self-contained store `dashboard/coach_subscriptions.py`. No raw card data touches the app.

**Tech Stack:** Python 3 / Flask (single `app.py` + `dashboard/*.py`), sqlite (`chat_log.db`, `?` placeholders, `_db_lock`), Stripe via `dashboard/stripe_pay.py` (raw HTTPS), `dashboard/subscriptions.py:add_months` for date math.

## Global Constraints

- **Money path (load-bearing):** month 1 is charged exactly ONCE — on Stripe's hosted checkout; `_fulfill_coach_sub` is idempotent (claim-then-create on `coach_sub_grants(session_id)` PRIMARY KEY, like `continuous_care_grants`) so a webhook+return double-delivery cannot double-create/double-grant. The cron only charges rows with `next_charge_at <= today` and advances `next_charge_at` by a month ONLY on a successful charge (same locked write) so a re-run does not double-charge. Signup seeds `next_charge_at = add_months(today, 1)` so the first cron run never re-charges new subscribers.
- **Card capture is Stripe-hosted:** the app stores only the Stripe `customer` id + `payment_method` id (from the succeeded PaymentIntent), never a card number. A subscription is created only when the checkout PaymentIntent is `succeeded` AND has a vaulted `customer` + `payment_method` (no un-chargeable subscription).
- **Tiers:** `rae` = 10000 cents ($100), service = one EVOX credit (`evox.add_session_credits(cx, email, 1)`); `glen` = 20000 cents ($200), service = Causal Biofield entitlement (`consult.set_consult_ready(cx, email, True)`). One included service per successful cycle (use-it-or-lose-it).
- **Failed charge:** increment `fail_count`, set `status='past_due'`, do NOT advance `next_charge_at` or grant; notify member + Glen. A `past_due` sub is skipped by the cron.
- **Cancel:** `status='canceled'`; cron skips; member keeps the already-granted current cycle (no refund/proration).
- **Copy:** no em dashes, no ALL CAPS; server strings via `textContent`.
- sqlite writes under `with _db_lock, sqlite3.connect(LOG_DB)`; emails lowercased.
- Go-live gate: `_STRIPE_ACTIVE` (already true on prod).
- DRY, YAGNI, TDD, frequent commits.

**Repo facts the implementer needs (mirror these exactly):**
- `dashboard/stripe_pay.py`: `create_checkout_session(amount_cents, *, customer_email, description, metadata, success_url, cancel_url, save_card=False) -> {id,url}` (with `save_card=True` → charges + vaults); `get_session(session_id) -> {id, payment_status, metadata, payment_intent, setup_intent}`; `get_payment_intent(pi_id) -> {..., status, customer, payment_method}`; `charge_off_session(customer_id, payment_method_id, amount_cents, *, description, metadata) -> {id, status}`; `_STRIPE_ACTIVE`.
- The sibling to copy: `_fulfill_continuous_care_monthly(session_id)` (app.py ~7916) — the claim-then-create idempotency + `get_session`→`get_payment_intent`→verify-succeeded→customer/pm flow. The `/webhook/stripe` fan-out calls every `_fulfill_*`; each re-fetches and no-ops on a non-matching `metadata.kind`. Add `_fulfill_coach_sub` to that dispatch AND call it from the subscribe return route (redirect + webhook both fulfill).
- `dashboard/subscriptions.py:add_months(yyyy_mm_dd, n) -> str` (month-end safe). `/api/cron/charge-subscriptions` (app.py:26421) is the recurring-charge cron to mirror.
- Grants: `dashboard/evox.py:add_session_credits(cx, email, n) -> int`; `dashboard/consult.py:set_consult_ready(cx, email, ready)`.
- `_evox_ident(cx, token)` (member); `send_evox_email(to, name, subject, html, text, ics_bytes)` + `GLEN_CONSULT_EMAIL`; `CONSOLE_SECRET`; `PUBLIC_BASE_URL`; `_db_lock`; `LOG_DB`; the EVOX-reminders cron's `X-Console-Key == CONSOLE_SECRET` auth idiom.

**Testing note (READ FIRST):**
- Pure/store test (Task 1) — plain `python3 -m pytest tests/test_coach_subscriptions_store.py -q`.
- Route/cron tests (Tasks 2-3) `import app`; override DATA_DIR:
  ```
  export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <paths> -q
  ```

---

### Task 1: Subscription store (`dashboard/coach_subscriptions.py`)

**Files:**
- Create: `dashboard/coach_subscriptions.py`
- Test: `tests/test_coach_subscriptions_store.py`

**Interfaces:**
- Consumes: `dashboard/subscriptions.py:add_months`.
- Produces: `TIERS`; `init_sub_tables(cx)`; `get_sub(cx, email) -> dict|None`; `create_sub(cx, *, email, tier, customer_id, payment_method_id, next_charge_at) -> None` (upsert on email); `set_status(cx, email, status)`; `mark_charged(cx, email, next_charge_at)` (advance + reset fail_count + last_charged_at=now); `mark_failed(cx, email)` (fail_count+=1, status='past_due'); `record_charge(cx, *, email, tier, amount_cents, pi_id, status)`; `due(cx, today) -> [dict]` (status='active' AND next_charge_at <= today).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_subscriptions_store.py
import sqlite3
from dashboard import coach_subscriptions as _cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _cs.init_sub_tables(cx)
    return cx


def test_tiers():
    assert _cs.TIERS["rae"]["amount_cents"] == 10000
    assert _cs.TIERS["glen"]["amount_cents"] == 20000
    assert _cs.TIERS["rae"]["service"] == "evox" and _cs.TIERS["glen"]["service"] == "biofield"


def test_create_and_get():
    cx = _cx()
    _cs.create_sub(cx, email="M@x.com", tier="rae", customer_id="cus_1",
                   payment_method_id="pm_1", next_charge_at="2026-08-05")
    s = _cs.get_sub(cx, "m@x.com")
    assert s["tier"] == "rae" and s["status"] == "active" and s["next_charge_at"] == "2026-08-05"
    assert s["stripe_customer_id"] == "cus_1" and s["payment_method_id"] == "pm_1"


def test_due_only_active_and_past():
    cx = _cx()
    _cs.create_sub(cx, email="due@x.com", tier="rae", customer_id="c", payment_method_id="p",
                   next_charge_at="2026-07-01")            # past → due
    _cs.create_sub(cx, email="future@x.com", tier="glen", customer_id="c", payment_method_id="p",
                   next_charge_at="2026-12-01")            # future → not due
    _cs.create_sub(cx, email="cx@x.com", tier="rae", customer_id="c", payment_method_id="p",
                   next_charge_at="2026-07-01"); _cs.set_status(cx, "cx@x.com", "canceled")
    emails = [d["member_email"] for d in _cs.due(cx, "2026-07-15")]
    assert emails == ["due@x.com"]                          # not future, not canceled


def test_mark_charged_advances_and_resets():
    cx = _cx()
    _cs.create_sub(cx, email="m@x.com", tier="rae", customer_id="c", payment_method_id="p",
                   next_charge_at="2026-07-01")
    _cs.mark_failed(cx, "m@x.com")
    assert _cs.get_sub(cx, "m@x.com")["fail_count"] == 1
    assert _cs.get_sub(cx, "m@x.com")["status"] == "past_due"
    _cs.mark_charged(cx, "m@x.com", "2026-08-01")
    s = _cs.get_sub(cx, "m@x.com")
    assert s["next_charge_at"] == "2026-08-01" and s["fail_count"] == 0 and s["last_charged_at"]


def test_record_charge_ledger():
    cx = _cx()
    _cs.record_charge(cx, email="m@x.com", tier="rae", amount_cents=10000, pi_id="pi_1",
                      status="succeeded")
    row = cx.execute("SELECT * FROM coach_sub_charges").fetchone()
    assert row["pi_id"] == "pi_1" and row["amount_cents"] == 10000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_coach_subscriptions_store.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/coach_subscriptions.py
"""Paid coaching subscription store (arc slice 2b). Pure sqlite. One subscription
per member (Rae $100/mo or Glen $200/mo). Recurring billing = card on file +
a monthly cron; this module holds state only. Money-path correctness (idempotent
first charge, no cron double-charge) lives in the routes/cron, guarded by
next_charge_at advancing only on success."""

TIERS = {
    "rae":  {"amount_cents": 10000, "service": "evox",     "label": "Rae"},
    "glen": {"amount_cents": 20000, "service": "biofield", "label": "Dr. Glen"},
}

_DDL = """
CREATE TABLE IF NOT EXISTS coach_subscriptions (
    member_email TEXT PRIMARY KEY,
    tier TEXT,
    amount_cents INTEGER,
    stripe_customer_id TEXT,
    payment_method_id TEXT,
    status TEXT,
    started_at TEXT,
    next_charge_at TEXT,
    last_charged_at TEXT,
    fail_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_coachsub_due ON coach_subscriptions(status, next_charge_at);
CREATE TABLE IF NOT EXISTS coach_sub_charges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_email TEXT,
    tier TEXT,
    amount_cents INTEGER,
    pi_id TEXT,
    status TEXT,
    charged_at TEXT
);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_sub_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def get_sub(cx, email):
    row = cx.execute("SELECT * FROM coach_subscriptions WHERE member_email=?",
                     (_lc(email),)).fetchone()
    return dict(row) if row else None


def create_sub(cx, *, email, tier, customer_id, payment_method_id, next_charge_at):
    amount = TIERS.get(tier, {}).get("amount_cents", 0)
    cx.execute(
        "INSERT INTO coach_subscriptions (member_email,tier,amount_cents,stripe_customer_id,"
        "payment_method_id,status,started_at,next_charge_at,fail_count) "
        "VALUES (?,?,?,?,?, 'active', ?, ?, 0) "
        "ON CONFLICT(member_email) DO UPDATE SET tier=excluded.tier, amount_cents=excluded.amount_cents, "
        "stripe_customer_id=excluded.stripe_customer_id, payment_method_id=excluded.payment_method_id, "
        "status='active', next_charge_at=excluded.next_charge_at, fail_count=0",
        (_lc(email), tier, amount, customer_id, payment_method_id, _now(), next_charge_at))
    cx.commit()


def set_status(cx, email, status):
    cx.execute("UPDATE coach_subscriptions SET status=? WHERE member_email=?",
               (status, _lc(email)))
    cx.commit()


def mark_charged(cx, email, next_charge_at):
    cx.execute("UPDATE coach_subscriptions SET next_charge_at=?, last_charged_at=?, "
               "fail_count=0, status='active' WHERE member_email=?",
               (next_charge_at, _now(), _lc(email)))
    cx.commit()


def mark_failed(cx, email):
    cx.execute("UPDATE coach_subscriptions SET fail_count=fail_count+1, status='past_due' "
               "WHERE member_email=?", (_lc(email),))
    cx.commit()


def record_charge(cx, *, email, tier, amount_cents, pi_id, status):
    cx.execute("INSERT INTO coach_sub_charges (member_email,tier,amount_cents,pi_id,status,charged_at) "
               "VALUES (?,?,?,?,?,?)", (_lc(email), tier, amount_cents, pi_id, status, _now()))
    cx.commit()


def due(cx, today):
    rows = cx.execute("SELECT * FROM coach_subscriptions WHERE status='active' "
                      "AND next_charge_at <= ? ORDER BY next_charge_at", (today,)).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_coach_subscriptions_store.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/coach_subscriptions.py tests/test_coach_subscriptions_store.py
git commit -m "feat(coaching): paid subscription store"
```

---

### Task 2: Subscribe + fulfillment + cancel (`app.py`)

**Files:**
- Modify: `app.py` (subscribe route, `_grant_cycle_service`, `_fulfill_coach_sub` + webhook/return wiring, cancel route)
- Test: `tests/test_coach_subscribe_api.py`

**Interfaces:**
- Consumes: `dashboard/coach_subscriptions.py` (all), `stripe_pay.create_checkout_session`/`get_session`/`get_payment_intent`, `dashboard/subscriptions.py:add_months`, `evox.add_session_credits`, `consult.set_consult_ready`, `_evox_ident`, `_STRIPE_ACTIVE`, `send_evox_email`, `GLEN_CONSULT_EMAIL`, `_db_lock`, `LOG_DB`, `PUBLIC_BASE_URL`.
- Produces: `POST /api/community/coach-subscribe`; `_grant_cycle_service(cx, email, tier)`; `_fulfill_coach_sub(session_id)`; `POST /api/community/coach-subscribe/cancel`.

**Contract:**
- `POST /api/community/coach-subscribe {tier}` (member portal-token; 400 bad tier; 503 if not `_STRIPE_ACTIVE`): `create_checkout_session(TIERS[tier].amount_cents, customer_email=email, description="Coaching with <label>", metadata={kind:"coach_sub", tier, email}, success_url=<base>/coach-subscribe/return?session_id={CHECKOUT_SESSION_ID}, cancel_url=<base>/portal/..., save_card=True)`; return `{ok, url}`.
- `_fulfill_coach_sub(session_id)` (called from the return route AND added to `/webhook/stripe`): `get_session`; kind-guard `metadata.kind=="coach_sub"`; `get_payment_intent(session.payment_intent)`; proceed only if `status=="succeeded"` AND `customer` AND `payment_method`; **claim-then-create** on `coach_sub_grants(session_id PRIMARY KEY)` (INSERT OR IGNORE, rowcount==1 means first delivery); only when claimed: `create_sub(next_charge_at=add_months(today,1))` + `record_charge(status="succeeded")` + `_grant_cycle_service` + confirmation email. Never raises.
- `_grant_cycle_service(cx, email, tier)`: `rae` → `evox.add_session_credits(cx, email, 1)`; `glen` → `consult.set_consult_ready(cx, email, True)`.
- `POST /api/community/coach-subscribe/cancel` (member portal-token): `set_status("canceled")`; `{ok}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_subscribe_api.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_subscriptions as _cs


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _tok(email="m@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx); _cs.init_sub_tables(cx)
        t = _ev.ensure_portal_token(cx, email, "Mel"); cx.commit()
    return t


def test_subscribe_returns_checkout_url():
    c = _client(); tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True), \
         mock.patch("dashboard.stripe_pay.create_checkout_session",
                    return_value={"id": "cs_1", "url": "https://stripe/cs_1"}):
        r = c.post(f"/api/community/coach-subscribe?token={tok}", json={"tier": "rae"})
    assert r.get_json()["url"] == "https://stripe/cs_1"


def test_subscribe_bad_tier_400():
    c = _client(); tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True):
        r = c.post(f"/api/community/coach-subscribe?token={tok}", json={"tier": "nope"})
    assert r.status_code == 400


def test_fulfill_creates_sub_grants_once():
    _tok()
    fake_session = {"metadata": {"kind": "coach_sub", "tier": "rae", "email": "m@x.com"},
                    "payment_intent": "pi_1"}
    fake_pi = {"status": "succeeded", "customer": "cus_1", "payment_method": "pm_1"}
    with mock.patch("dashboard.stripe_pay.get_session", return_value=fake_session), \
         mock.patch("dashboard.stripe_pay.get_payment_intent", return_value=fake_pi), \
         mock.patch("dashboard.evox.add_session_credits", return_value=1) as grant, \
         mock.patch.object(appmod, "send_evox_email"):
        appmod._fulfill_coach_sub("cs_evt_1")
        appmod._fulfill_coach_sub("cs_evt_1")   # webhook + return double-delivery
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        s = _cs.get_sub(cx, "m@x.com")
        assert s["tier"] == "rae" and s["status"] == "active"
        n = cx.execute("SELECT COUNT(*) FROM coach_sub_charges WHERE member_email='m@x.com'").fetchone()[0]
    assert grant.call_count == 1 and n == 1        # granted + charged exactly once


def test_fulfill_ignores_unpaid():
    with mock.patch("dashboard.stripe_pay.get_session",
                    return_value={"metadata": {"kind": "coach_sub", "tier": "glen", "email": "u@x.com"},
                                  "payment_intent": "pi_x"}), \
         mock.patch("dashboard.stripe_pay.get_payment_intent",
                    return_value={"status": "requires_payment_method", "customer": None, "payment_method": None}):
        appmod._fulfill_coach_sub("cs_evt_2")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        assert _cs.get_sub(cx, "u@x.com") is None   # no sub on an unpaid session


def test_cancel_sets_canceled():
    c = _client(); tok = _tok("cxl@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        _cs.create_sub(cx, email="cxl@x.com", tier="rae", customer_id="c",
                       payment_method_id="p", next_charge_at="2026-08-01"); cx.commit()
    r = c.post(f"/api/community/coach-subscribe/cancel?token={tok}")
    assert r.get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        assert _cs.get_sub(cx, "cxl@x.com")["status"] == "canceled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_subscribe_api.py -q`
Expected: FAIL — routes / `_fulfill_coach_sub` missing.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near the other community routes). Mirror `_fulfill_continuous_care_monthly` for the fulfillment shape:

```python
def _grant_cycle_service(cx, email, tier):
    """Grant this cycle's included service: Rae -> 1 EVOX credit; Glen -> Causal
    Biofield entitlement. Best-effort; never raises."""
    try:
        if tier == "rae":
            from dashboard import evox as _ev
            _ev.add_session_credits(cx, email, 1)
        elif tier == "glen":
            from dashboard import consult as _cn
            _cn.set_consult_ready(cx, email, True)
    except Exception:
        app.logger.exception("coach sub grant failed for %s/%s", email, tier)


@app.route("/api/community/coach-subscribe", methods=["POST"])
def community_coach_subscribe():
    from dashboard import coach_subscriptions as _cs
    body = request.get_json(force=True) or {}
    tier = (body.get("tier") or "").strip()
    if tier not in _cs.TIERS:
        return jsonify({"error": "bad_tier"}), 400
    if not _STRIPE_ACTIVE:
        return jsonify({"error": "unavailable"}), 503
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        email = ident.email
    from dashboard import stripe_pay as _sp
    base = PUBLIC_BASE_URL.rstrip("/")
    sess = _sp.create_checkout_session(
        _cs.TIERS[tier]["amount_cents"], customer_email=email,
        description=f"Coaching with {_cs.TIERS[tier]['label']}",
        metadata={"kind": "coach_sub", "tier": tier, "email": email},
        success_url=f"{base}/coach-subscribe/return?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/", save_card=True)
    return jsonify({"ok": True, "url": sess.get("url")})


def _fulfill_coach_sub(session_id):
    """Create a paid coaching subscription from a paid+vaulted coach_sub checkout,
    idempotently (claim-then-create on coach_sub_grants(session_id)). Callable from
    the /coach-subscribe/return redirect AND the webhook. Never raises."""
    try:
        from dashboard import stripe_pay as _sp, coach_subscriptions as _cs, subscriptions as _subs
        sess = _sp.get_session(session_id)
        md = sess.get("metadata") or {}
        if md.get("kind") != "coach_sub":
            return {"ok": False, "reason": "not_coach_sub"}
        email = (md.get("email") or "").strip().lower()
        tier = (md.get("tier") or "").strip()
        pi_id = sess.get("payment_intent")
        if not (email and tier in _cs.TIERS and pi_id):
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
            cx.execute("CREATE TABLE IF NOT EXISTS coach_sub_grants "
                       "(session_id TEXT PRIMARY KEY, email TEXT, created_at TEXT)")
            _cs.init_sub_tables(cx)
            claimed = cx.execute(
                "INSERT OR IGNORE INTO coach_sub_grants (session_id,email,created_at) VALUES (?,?,?)",
                (session_id, email, _cs._now())).rowcount == 1
            cx.commit()
            if not claimed:
                return {"ok": True, "reason": "already_fulfilled"}
            _cs.create_sub(cx, email=email, tier=tier, customer_id=customer,
                           payment_method_id=pm, next_charge_at=next_charge)
            _cs.record_charge(cx, email=email, tier=tier,
                              amount_cents=_cs.TIERS[tier]["amount_cents"], pi_id=pi_id,
                              status="succeeded")
            _grant_cycle_service(cx, email, tier)
        try:
            html = (f"<p>Your monthly coaching with {_cs.TIERS[tier]['label']} is active. "
                    f"This cycle includes your included session. You can cancel any time from "
                    f"your portal.</p>")
            send_evox_email(email, "", "Your coaching subscription is active", html, html, b"")
        except Exception:
            app.logger.exception("coach sub confirmation failed")
        return {"ok": True}
    except Exception:
        app.logger.exception("coach sub fulfill failed for %s", session_id)
        return {"ok": False, "reason": "error"}


@app.route("/coach-subscribe/return")
def coach_subscribe_return():
    sid = request.args.get("session_id", "")
    if sid:
        _fulfill_coach_sub(sid)
    return redirect(f"{PUBLIC_BASE_URL.rstrip('/')}/")


@app.route("/api/community/coach-subscribe/cancel", methods=["POST"])
def community_coach_subscribe_cancel():
    from dashboard import coach_subscriptions as _cs
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_sub_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        _cs.set_status(cx, ident.email, "canceled")
        return jsonify({"ok": True})
```

Then wire `_fulfill_coach_sub` into the `/webhook/stripe` handler's `_fulfill_*` dispatch (grep `_fulfill_continuous_care_monthly(` in the webhook route and add `_fulfill_coach_sub(sid)` alongside it, guarded the same way — each fulfiller no-ops on a non-matching kind).

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_subscribe_api.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_coach_subscribe_api.py
git commit -m "feat(coaching): subscribe checkout + idempotent fulfillment + cancel"
```

---

### Task 3: Monthly charge cron (`app.py`)

**Files:**
- Modify: `app.py` (add the recurring charge cron)
- Test: `tests/test_coach_sub_cron.py`

**Interfaces:**
- Consumes: `dashboard/coach_subscriptions.py` (`due`, `mark_charged`, `mark_failed`, `record_charge`, `TIERS`), `stripe_pay.charge_off_session`, `dashboard/subscriptions.py:add_months`, `_grant_cycle_service`, `CONSOLE_SECRET`, `send_evox_email`, `GLEN_CONSULT_EMAIL`, `_db_lock`, `LOG_DB`.
- Produces: `POST /api/cron/coach-subscriptions/charge`.

**Contract:** `X-Console-Key == CONSOLE_SECRET` (else 401). For each `due(cx, today)` sub: `charge_off_session(customer, pm, amount, ...)`; on `status=="succeeded"` → `record_charge(succeeded)` + `_grant_cycle_service` + `mark_charged(next_charge_at=add_months(today,1))`; else → `record_charge(failed)` + `mark_failed` + best-effort notify. Returns `{charged, failed}`. Only charges `next_charge_at <= today` and advances on success → no double-charge on a re-run; a future-dated sub is never charged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coach_sub_cron.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_subscriptions as _cs


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(email, tier, next_at):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        _cs.create_sub(cx, email=email, tier=tier, customer_id="cus", payment_method_id="pm",
                       next_charge_at=next_at); cx.commit()


def _hdr():
    return {"X-Console-Key": appmod.CONSOLE_SECRET}


def test_cron_requires_key():
    assert _client().post("/api/cron/coach-subscriptions/charge").status_code == 401


def test_cron_charges_due_grants_and_advances():
    c = _client(); _seed("due@x.com", "rae", "2026-01-01")   # far-past → due
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": "pi_ok", "status": "succeeded"}), \
         mock.patch("dashboard.evox.add_session_credits", return_value=1) as grant:
        d = c.post("/api/cron/coach-subscriptions/charge", headers=_hdr()).get_json()
    assert d["charged"] == 1 and grant.called
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        s = _cs.get_sub(cx, "due@x.com")
        assert s["next_charge_at"] > "2026-01-01" and s["last_charged_at"]   # advanced


def test_cron_skips_future():
    c = _client(); _seed("future@x.com", "glen", "2099-01-01")   # future → not due
    with mock.patch("dashboard.stripe_pay.charge_off_session") as charge:
        d = c.post("/api/cron/coach-subscriptions/charge", headers=_hdr()).get_json()
    assert d["charged"] == 0 and not charge.called                # never charged early


def test_cron_failed_charge_past_due_no_grant():
    c = _client(); _seed("fail@x.com", "rae", "2026-01-01")
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": None, "status": "failed"}), \
         mock.patch("dashboard.evox.add_session_credits") as grant, \
         mock.patch.object(appmod, "send_evox_email"):
        d = c.post("/api/cron/coach-subscriptions/charge", headers=_hdr()).get_json()
    assert d["failed"] == 1 and not grant.called
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        s = _cs.get_sub(cx, "fail@x.com")
        assert s["status"] == "past_due" and s["fail_count"] == 1
        assert s["next_charge_at"] == "2026-01-01"                 # NOT advanced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_sub_cron.py -q`
Expected: FAIL — route 404.

- [ ] **Step 3: Write minimal implementation**

```python
@app.route("/api/cron/coach-subscriptions/charge", methods=["POST"])
def coach_subscriptions_charge_cron():
    if request.headers.get("X-Console-Key") != CONSOLE_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import coach_subscriptions as _cs, stripe_pay as _sp, subscriptions as _subs
    from datetime import date as _date
    today = _date.today().isoformat()
    charged = failed = 0
    with sqlite3.connect(LOG_DB) as rcx:
        rcx.row_factory = sqlite3.Row
        _cs.init_sub_tables(rcx)
        due_rows = _cs.due(rcx, today)
    for sub in due_rows:
        email = sub["member_email"]; tier = sub["tier"]
        res = _sp.charge_off_session(sub["stripe_customer_id"], sub["payment_method_id"],
                                     sub["amount_cents"],
                                     description=f"Coaching with {_cs.TIERS.get(tier,{}).get('label','')}",
                                     metadata={"kind": "coach_sub_cycle", "tier": tier, "email": email})
        ok = res.get("status") == "succeeded"
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _cs.init_sub_tables(cx)
            _cs.record_charge(cx, email=email, tier=tier, amount_cents=sub["amount_cents"],
                              pi_id=res.get("id") or "", status="succeeded" if ok else "failed")
            if ok:
                _grant_cycle_service(cx, email, tier)
                _cs.mark_charged(cx, email, _subs.add_months(today, 1))
            else:
                _cs.mark_failed(cx, email)
        if ok:
            charged += 1
        else:
            failed += 1
            try:
                html = (f"<p>We could not process this month's coaching charge for {email}. "
                        f"Please update the card on file to keep coaching active.</p>")
                send_evox_email(email, "", "Your coaching payment did not go through", html, html, b"")
                send_evox_email(GLEN_CONSULT_EMAIL, "Glen", f"Coaching charge failed: {email}",
                                html, html, b"")
            except Exception:
                app.logger.exception("coach sub failure notify failed for %s", email)
    return jsonify({"charged": charged, "failed": failed})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_coach_sub_cron.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_coach_sub_cron.py
git commit -m "feat(coaching): monthly subscription charge cron"
```

---

### Task 4: Member subscription surface (`static/client-portal.html`)

**Files:**
- Modify: `static/client-portal.html` (the slice-2 upsell block + a subscription-status card)
- Test: manual JS parse check.

**Interfaces:**
- Consumes: `POST /api/community/coach-subscribe {tier}` → `{ok, url}`; `POST /api/community/coach-subscribe/cancel`.

**Design note:** the slice-2 upsell offer currently has two "I'm interested" buttons (Rae/Glen) posting `coaching-interest`. Change those to **Subscribe** buttons that `POST /api/community/coach-subscribe?token=...` `{tier:"rae"|"glen"}` and, on `{ok, url}`, `window.location = url` (redirect to Stripe). Keep the plain-language copy: "Rae, $100 a month, includes one EVOX session each month" / "Dr. Glen, $200 a month, includes one Causal Biofield Analysis each month". Wrap new JS in `<!-- BEGIN coach-subscribe script -->` / `<!-- END coach-subscribe script -->`. (A full "Your coaching subscription" status card with the next-charge date + Cancel can read a small status endpoint; for this slice, a Cancel action is enough — add a quiet "Cancel coaching" link that `POST`s the cancel route and shows "Your coaching is canceled." on success. Server strings via textContent; no em dashes, no ALL CAPS.)

- [ ] **Step 1: Convert the upsell buttons to Subscribe + add Cancel**

Read the slice-2 upsell block in `static/client-portal.html`. Change the two interest buttons to Subscribe (redirect to the returned Stripe url), and add a quiet "Cancel coaching" action posting the cancel route.

- [ ] **Step 2: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re; h=open('static/client-portal.html').read(); print('\n;\n'.join(re.findall(r'<script>(.*?)</script>', h, re.S)))")`
Expected: no output (clean parse).

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(coaching): subscribe + cancel on the upsell offer"
```

---

## Definition of Done

- A member subscribes to Rae ($100) or Glen ($200) from the upsell; month 1 is charged on Stripe's hosted page and vaults the card; `_fulfill_coach_sub` creates the subscription once (idempotent) and grants the cycle service (EVOX credit / Biofield entitlement); a monthly cron recharges + regrants and advances the next date only on success; a failed charge goes past_due without a grant; cancel deactivates and the cron skips.
- The money path holds: month 1 charged once, no cron double-charge, no first-run mass-charge, no un-chargeable subscription.
- All new tests pass; EVOX/consult/Community/coaching slices 1-2 untouched (this reuses their grant helpers + Stripe).

## Deferred (not in this plan)

- The Render cron_job wiring for `/api/cron/coach-subscriptions/charge` (a go-live op, like the EVOX-reminders cron — set up after merge).
- Slice 3: the 1:1 coaching thread (the paid coaching conversation) + report/block.
- A full subscription-status card (next-charge date, this cycle's service used) via a status endpoint; annual/prepay tiers; richer dunning; carryover.
