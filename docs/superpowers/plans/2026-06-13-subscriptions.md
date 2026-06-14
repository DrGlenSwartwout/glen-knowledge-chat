# Subscriptions "Subscribe & Grow" — Implementation Plan (Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Recurring orders where Stripe vaults the card and OUR daily scheduler charges off-session each cycle through the existing `_price_cart()` (Plan 2) at an escalating loyalty tier (5/10/15%), with skip/pause/cadence/cancel self-serve — all behind `SUBSCRIPTIONS_ENABLED` (default off).

**Architecture:** New `dashboard/subscriptions.py` (table + CRUD + tier math, pure). New Stripe primitives in `dashboard/stripe_pay.py` (save-card checkout + off-session charge). A `/reorder/subscribe` setup flow captures the Stripe customer + payment-method on checkout-return and writes a subscription row. A daily `/api/cron/charge-subscriptions` endpoint (CRON_SECRET-gated, like the existing crons) does heads-up emails + charges due subs via `_price_cart(subscriber_tier_pct=tier)` → order + QBO invoice + receipt, with basic dunning. A magic-link `/subscription` portal (reuses the `rm_reorder_email` cookie) does skip/pause/resume/change-cadence/cancel.

**Tech Stack:** Python 3.11, Flask, sqlite (`LOG_DB`), Stripe REST (raw HTTP, matching the existing `stripe_pay.py`), QBO, `dashboard/pricing`+`_price_cart`, pytest.

**Run tests:** pure modules (Tasks 1-2) → `~/.venvs/deploy-chat311/bin/python -m pytest <path>`; route/app tests → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path>` (ignore the 2 known pre-existing failures).

**Depends on (already in main):** `dashboard/pricing.compute`, `dashboard/points`, `app._price_cart` / `_engine_item` / `_shipping_line` / `CheckoutError`, the orders discount/points/shipping columns, the reorder magic-link auth (`auth_tokens`, `_hash_token`, `send_magic_link_email`, `_reorder_email_from_cookie`, `rm_reorder_email` cookie), the cron pattern (`/cron/*` + `X-Cron-Secret`, `scripts/run_*_cron.py`).

**Map facts:** `stripe_pay.create_checkout_session(amount_cents, *, customer_email, description, metadata, success_url, cancel_url)` POSTs form params to `api.stripe.com/v1/checkout/sessions` (mode=payment); `get_session(id)` returns `{id,payment_status,amount_total,metadata,payment_intent}`. No Customer/save-card/off-session today. `/begin/checkout-return` reads the session metadata and records the QBO payment. `_price_cart(cart, *, ship, coupon_pct, subscriber_tier_pct, channel)` returns `{priced, qbo_lines, items_rec, discount_cents, points_redeemed_cents, shipping_cents}`.

---

### Task 1: Stripe vault primitives (save-card + off-session charge)

**Files:**
- Modify: `dashboard/stripe_pay.py`
- Test: `tests/test_stripe_vault.py`

First READ `dashboard/stripe_pay.py` to find its HTTP helper (the function that POSTs to `api.stripe.com` with the Bearer key + form-encoded body) and reuse it. Tests stub that helper — never hit real Stripe.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stripe_vault.py
from dashboard import stripe_pay

class _Resp:
    def __init__(self, d): self._d = d
    def json(self): return self._d

def test_checkout_session_save_card_params(monkeypatch):
    captured = {}
    def fake_post(path, params):           # match the real helper's (path, params) shape
        captured["path"] = path; captured["params"] = params
        return {"id": "cs_1", "url": "https://stripe/x"}
    monkeypatch.setattr(stripe_pay, "_post", fake_post)   # adjust name to the real helper
    stripe_pay.create_checkout_session(
        7000, customer_email="a@x.com", description="d", metadata={"k": "v"},
        success_url="s", cancel_url="c", save_card=True)
    p = captured["params"]
    assert p["mode"] == "payment"
    assert p["customer_creation"] == "always"
    assert p["payment_intent_data[setup_future_usage]"] == "off_session"

def test_charge_off_session_params(monkeypatch):
    captured = {}
    def fake_post(path, params):
        captured["path"] = path; captured["params"] = params
        return {"id": "pi_1", "status": "succeeded"}
    monkeypatch.setattr(stripe_pay, "_post", fake_post)
    out = stripe_pay.charge_off_session("cus_1", "pm_1", 5000,
                                        description="cycle", metadata={"sub": "9"})
    assert captured["path"].endswith("/payment_intents")
    assert captured["params"]["off_session"] == "true"
    assert captured["params"]["confirm"] == "true"
    assert captured["params"]["customer"] == "cus_1"
    assert captured["params"]["payment_method"] == "pm_1"
    assert out["status"] == "succeeded"

def test_charge_off_session_card_declined(monkeypatch):
    def fake_post(path, params):
        return {"error": {"type": "card_error", "code": "card_declined",
                          "decline_code": "insufficient_funds"}}
    monkeypatch.setattr(stripe_pay, "_post", fake_post)
    out = stripe_pay.charge_off_session("cus_1", "pm_1", 5000, description="x", metadata={})
    assert out["status"] == "failed"
    assert out["decline_code"] == "insufficient_funds"
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_stripe_vault.py -v`
Expected: FAIL (save_card kwarg / charge_off_session missing; adjust the `_post` name to the real helper if needed).

- [ ] **Step 3: Implement**

In `dashboard/stripe_pay.py`:
- Add `save_card=False` to `create_checkout_session`; when True add to the params: `"customer_creation": "always"` and `"payment_intent_data[setup_future_usage]": "off_session"`.
- Add `get_payment_intent(pi_id)` → GET `/payment_intents/{pi_id}` returning `{id, customer, payment_method, status}` (so the setup return can read the vaulted customer + pm).
- Add:
```python
def charge_off_session(customer_id, payment_method_id, amount_cents, *, description, metadata):
    """Charge a vaulted card off-session. Returns {id, status, decline_code?, error?}.
    status: 'succeeded' | 'requires_action' | 'failed'."""
    params = {
        "amount": str(int(amount_cents)), "currency": "usd",
        "customer": customer_id, "payment_method": payment_method_id,
        "off_session": "true", "confirm": "true", "description": description or "",
    }
    for k, v in (metadata or {}).items():
        params[f"metadata[{k}]"] = str(v)
    resp = _post("/payment_intents", params)          # match the real helper name
    err = resp.get("error")
    if err:
        return {"id": None, "status": "failed",
                "decline_code": err.get("decline_code") or err.get("code"),
                "error": err.get("message")}
    return {"id": resp.get("id"), "status": resp.get("status")}
```

- [ ] **Step 4: Run to verify it passes** — `... pytest tests/test_stripe_vault.py -v` → PASS (3).
- [ ] **Step 5: Commit** — `git commit -m "feat(stripe): save-card checkout + off-session charge primitives"`

---

### Task 2: Subscriptions table + model + tier math

**Files:**
- Create: `dashboard/subscriptions.py`
- Test: `tests/test_subscriptions_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subscriptions_model.py
import sqlite3
from dashboard import subscriptions as subs

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx); return cx

def test_tier_for_escalates_and_caps():
    assert subs.tier_for(0) == 5
    assert subs.tier_for(1) == 10
    assert subs.tier_for(2) == 15
    assert subs.tier_for(9) == 15

def test_create_and_get():
    cx = _cx()
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="cus_1",
                      stripe_payment_method_id="pm_1",
                      items=[{"slug":"x","qty":1}], cadence_months=1,
                      ship_address={"state":"CA"}, next_charge_date="2026-07-01")
    s = subs.get(cx, sid)
    assert s["status"] == "active" and s["order_count"] == 0
    assert subs.get_active_by_email(cx, "a@x.com")[0]["id"] == sid

def test_list_due_respects_status_skip_and_date():
    cx = _cx()
    a = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                    items=[], cadence_months=1, ship_address={}, next_charge_date="2026-07-01")
    subs.create(cx, email="b@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                items=[], cadence_months=1, ship_address={}, next_charge_date="2026-09-01")
    due = subs.list_due(cx, as_of="2026-07-15")
    assert [d["id"] for d in due] == [a]            # only the past-due active one
    subs.set_skip_next(cx, a, True)
    assert subs.list_due(cx, as_of="2026-07-15") == []   # skip hides it

def test_advance_after_charge_increments_and_moves_date():
    cx = _cx()
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                      items=[], cadence_months=2, ship_address={}, next_charge_date="2026-07-01")
    subs.advance_after_charge(cx, sid)
    s = subs.get(cx, sid)
    assert s["order_count"] == 1
    assert s["next_charge_date"] == "2026-09-01"     # +2 months

def test_skip_consumes_without_incrementing_tier():
    cx = _cx()
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                      items=[], cadence_months=1, ship_address={}, next_charge_date="2026-07-01")
    subs.set_skip_next(cx, sid, True)
    subs.consume_skip(cx, sid)                        # advances date, clears flag, no order_count++
    s = subs.get(cx, sid)
    assert s["order_count"] == 0 and s["skip_next"] == 0
    assert s["next_charge_date"] == "2026-08-01"

def test_cancel_resets_tier():
    cx = _cx()
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                      items=[], cadence_months=1, ship_address={}, next_charge_date="2026-07-01")
    subs.advance_after_charge(cx, sid)
    subs.set_status(cx, sid, "cancelled")
    s = subs.get(cx, sid)
    assert s["status"] == "cancelled" and s["order_count"] == 0   # reset on cancel
```

- [ ] **Step 2: Run to verify it fails** — module missing.

- [ ] **Step 3: Implement** `dashboard/subscriptions.py`:
- `SUBSCRIBE_TIERS = [5, 10, 15]`; `tier_for(n) = SUBSCRIBE_TIERS[min(int(n), len-1)]`.
- A pure date helper `add_months(yyyy_mm_dd, n)` (no external deps; handle month overflow, clamp day to month end).
- `init_subscriptions_table(cx)`: columns `id, email, stripe_customer_id, stripe_payment_method_id, items_json, cadence_months, status DEFAULT 'active', order_count DEFAULT 0, next_charge_date, ship_address_json, skip_next DEFAULT 0, last_notified_date, created_at, updated_at, cancelled_at` + index on `(status, next_charge_date)` and on `email`.
- `create(...)` returns the new id (status 'active', order_count 0). `get(cx, sid)` → dict. `get_active_by_email(cx, email)` → list. `list_due(cx, as_of)` → active, `skip_next=0`, `next_charge_date <= as_of`, ordered by date.
- `advance_after_charge(cx, sid)`: `order_count += 1`, `next_charge_date = add_months(next_charge_date, cadence_months)`, touch updated_at.
- `consume_skip(cx, sid)`: `next_charge_date = add_months(...)`, `skip_next = 0` (no order_count change).
- `set_skip_next(cx, sid, bool)`, `set_status(cx, sid, status)` (when status=='cancelled' also set order_count=0 + cancelled_at), `set_cadence(cx, sid, months)` (recompute next_charge_date from today is NOT needed; just store), `set_next_charge_date(cx, sid, date)`.

- [ ] **Step 4: Run to verify it passes** — PASS (6).
- [ ] **Step 5: Commit** — `feat(subscriptions): table + CRUD + escalating tier math`

---

### Task 3: Subscribe setup flow (vault the card at checkout, write the subscription)

**Files:**
- Modify: `app.py` (a `/reorder/subscribe` route + capture in `/begin/checkout-return`)
- Test: `tests/test_subscribe_setup.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subscribe_setup.py
import app as appmod

def _setup(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug":s,"name":"Brain Boost","price_cents":7000,"qty_pricing":True,"qbo_item_id":"27"} if s=="brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id":"INV1","TotalAmt":50.0,"DocNumber":"1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    cap = {}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session",
        lambda *a, **k: cap.update(k) or {"id":"cs_1","url":"https://stripe/setup"})
    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    return cap

def test_subscribe_creates_setup_session_with_save_card(monkeypatch):
    cap = _setup(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/reorder/subscribe", json={"items":[{"slug":"brain-boost","qty":1}],
               "cadence_months":1, "address":{"state":"CA","country":"US","name":"A"}})
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://stripe/setup"
    assert cap["save_card"] is True                      # vaults the card
    # first order priced at the 5% tier (tier_for(0))
    # (invoice stub fixed; assert the metadata carries the kind + cadence for the return handler)
    assert cap["metadata"]["kind"] == "subscribe"
    assert cap["metadata"]["cadence_months"] == "1"

def test_subscribe_disabled_when_flag_off(monkeypatch):
    cap = _setup(monkeypatch); monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "false")
    c = appmod.app.test_client()
    r = c.post("/reorder/subscribe", json={"items":[{"slug":"brain-boost","qty":1}],
               "cadence_months":1, "address":{"state":"CA","country":"US"}})
    assert r.status_code == 400
```

- [ ] **Step 2: Run to verify it fails** — route missing.

- [ ] **Step 3: Implement**
- `_subscriptions_enabled()` helper: `os.environ.get("SUBSCRIPTIONS_ENABLED","").strip().lower() in ("1","true","yes","on")`.
- `/reorder/subscribe` (POST, authed via `_reorder_email_from_cookie`): if not `_subscriptions_enabled()` → 400. Read items + cadence_months (1/2/3) + address. Price the FIRST order via `_price_cart(cart, ship=ship, subscriber_tier_pct=subs.tier_for(0))` (5%). Build the first invoice exactly like the engine reorder path (list lines + shipping line + discount_cents). Create a Stripe checkout session with `save_card=True` and `metadata={"kind":"subscribe","cadence_months":str(cadence),"email":email,"invoice_id":inv_id,"customer_id":qbo_cid,"items":json.dumps(cart),"ship":json.dumps(ship)}` (items+ship small; if too big for Stripe metadata, stash them in a `pending_subscriptions` row keyed by session id and put only that key in metadata — implement the stash if the JSON exceeds ~450 chars). Return `{ok, stripe_url}`.
- In `/begin/checkout-return`: when `md.get("kind") == "subscribe"` and `payment_status == "paid"`: retrieve the PaymentIntent via `stripe_pay.get_payment_intent(sess["payment_intent"])` to get `customer` + `payment_method`; then `subs.create(cx, email=md["email"], stripe_customer_id=pi["customer"], stripe_payment_method_id=pi["payment_method"], items=json.loads(md["items"]), cadence_months=int(md["cadence_months"]), ship_address=json.loads(md["ship"]), next_charge_date=add_months(today, cadence))`; record the first order (it was already invoiced) + send the setup-confirmation email. Keep this in a try/except so it never breaks the return redirect.

- [ ] **Step 4: Run to verify it passes** — PASS (2).
- [ ] **Step 5: Commit** — `feat(subscriptions): /reorder/subscribe setup flow vaults card + creates subscription on return`

---

### Task 4: Daily charge scheduler (heads-up + charge + dunning)

**Files:**
- Modify: `app.py` (a `/api/cron/charge-subscriptions` endpoint)
- Create: `scripts/run_subscriptions_cron.py`
- Modify: `render.yaml` (add ONE cron service OR fold the curl into an existing daily script — DO NOT exceed the cron-service count that previously broke the Blueprint; prefer folding into an existing daily cron script that already curls endpoints)
- Test: `tests/test_subscriptions_cron.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subscriptions_cron.py
import os, sqlite3, app as appmod
from dashboard import subscriptions as subs

def _seed_due(monkeypatch):
    # one due sub; stub pricing + stripe + invoice + order + email
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    monkeypatch.setattr(appmod, "_price_cart", lambda cart, **k: {
        "priced": {"total_cents": 5000, "get_cents": 0}, "qbo_lines": [{"name":"X","amount":50.0,"qty":1}],
        "items_rec": [{"name":"X","qty":1,"desc":"X"}], "discount_cents": 350,
        "points_redeemed_cents": 0, "shipping_cents": 0})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id":"C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id":"INV","TotalAmt":50.0})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    charged = {}
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: charged.update({"amount": a[2]}) or {"id":"pi_1","status":"succeeded"})
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))
    return charged

def test_cron_charges_due_subscription_and_advances(monkeypatch, tmp_path):
    charged = _seed_due(monkeypatch)
    # use the app's LOG_DB-backed subscriptions; create one due today
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="cus_1", stripe_payment_method_id="pm_1",
                      items=[{"slug":"x","qty":1}], cadence_months=1, ship_address={"state":"CA"},
                      next_charge_date="2000-01-01"); cx.commit()
    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions",
               headers={"X-Cron-Secret": os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET","")})
    assert r.status_code == 200
    body = r.get_json(); assert body["charged"] >= 1
    s = subs.get(cx, sid); assert s["order_count"] == 1     # advanced
```

(If `CRON_SECRET`/`CONSOLE_SECRET` is unavailable in the test env, the implementer may monkeypatch the secret check to pass — note it.)

- [ ] **Step 2: Run to verify it fails** — endpoint missing.

- [ ] **Step 3: Implement**
- `_send_subscription_email(to_email, kind, data)` — thin wrapper building subject/body per kind in (`heads_up`, `receipt`, `payment_failed`, `setup_confirm`) and sending via the existing `_send_full_report_email` (or the SMTP path). Glen voice, "In wellness, Dr. Glen".
- `/api/cron/charge-subscriptions` (POST, `X-Cron-Secret` gated exactly like `/cron/personal-send`): support `?dry_run=1`. Two passes:
  1. **Heads-up:** active subs with `next_charge_date` within `lead_days` (3) and `last_notified_date != next_charge_date` → `_send_subscription_email(..., "heads_up", ...)`, set `last_notified_date`.
  2. **Charge:** `subs.list_due(cx, as_of=today)`:
     - if `skip_next` → `subs.consume_skip` (handled by list_due excluding skip; so flip: process skip BEFORE list_due, or have a `list_skip_due` + consume). Simplest: iterate active due-or-skip; for skip → `consume_skip`, continue.
     - price via `_price_cart(items, ship=ship_address, subscriber_tier_pct=subs.tier_for(order_count))`.
     - `res = stripe_pay.charge_off_session(stripe_customer_id, stripe_payment_method_id, total_cents, description=..., metadata={"sub":sid})`.
     - on `succeeded` → build QBO invoice (list lines + shipping + discount) + `_ingest_order(source="subscription", ...)` + `subs.advance_after_charge` + `_send_subscription_email(..., "receipt", ...)`.
     - on `failed` → increment a dunning counter (store on the row, e.g. reuse `last_notified_date`/add a `failed_count` column — add the column in this task's migration), `_send_subscription_email(..., "payment_failed", ...)`; after 3 fails set status `past_due` (effectively paused). Do NOT advance.
     - on `requires_action` → treat as failed for v1 + a payment_failed email noting authentication needed.
  - Return `{ok, charged, skipped, failed, notified}`.
- `scripts/run_subscriptions_cron.py`: stdlib-only; curls `{WEB_URL}/api/cron/charge-subscriptions` with `X-Cron-Secret` (mirror `scripts/run_personal_email_cron.py`).
- render.yaml: add the daily schedule (e.g. `0 16 * * *` = 6 AM HST) — but FIRST check the current cron-service count; if adding a service risks the Blueprint limit, instead add the curl into an existing daily multi-endpoint cron script and note it in the plan output.

- [ ] **Step 4: Run to verify it passes** — PASS.
- [ ] **Step 5: Commit** — `feat(subscriptions): daily charge scheduler + heads-up + basic dunning`

---

### Task 5: Manage-plan portal

**Files:**
- Modify: `app.py` (`/subscription` page + `/api/subscription/*` routes)
- Create: `static/subscription.html`
- Test: `tests/test_subscription_portal.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subscription_portal.py
import sqlite3, app as appmod
from dashboard import subscriptions as subs

def test_portal_requires_auth_cookie(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "")
    c = appmod.app.test_client()
    assert c.get("/api/subscription/details").status_code == 401

def test_portal_actions(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                      items=[{"slug":"x","qty":1}], cadence_months=1, ship_address={}, next_charge_date="2030-01-01")
    cx.commit()
    c = appmod.app.test_client()
    assert c.get("/api/subscription/details").get_json()["subscriptions"][0]["id"] == sid
    assert c.post("/api/subscription/skip", json={"id": sid}).status_code == 200
    assert subs.get(cx, sid)["skip_next"] == 1
    assert c.post("/api/subscription/pause", json={"id": sid}).status_code == 200
    assert subs.get(cx, sid)["status"] == "paused"
    assert c.post("/api/subscription/cancel", json={"id": sid}).status_code == 200
    assert subs.get(cx, sid)["status"] == "cancelled"
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**
- All `/api/subscription/*` routes: 401 if no `_reorder_email_from_cookie()`. Every action must verify the target subscription's `email` matches the cookie email (no cross-account edits).
- `GET /api/subscription/details` → `{subscriptions: [...]}` (the caller's, with next_charge_date, current tier `subs.tier_for(order_count)`, items, cadence, status, skip_next).
- `POST /api/subscription/skip {id}` → `set_skip_next(True)`; `/resume-skip` → False.
- `POST /api/subscription/pause {id}` → status 'paused'; `/resume {id}` → 'active'.
- `POST /api/subscription/cancel {id}` → status 'cancelled'.
- `POST /api/subscription/cadence {id, cadence_months}` → validate in (1,2,3), `set_cadence`.
- `GET /subscription` → serve `static/subscription.html` (mirrors `reorder.html`: sign-in state via the reorder magic link, then a "Your plan" view listing subscriptions + Skip next / Pause / Resume / Change cadence / Cancel buttons calling the APIs; card updates show a "contact us to update your card" note for v1).

- [ ] **Step 4: Run to verify it passes** — PASS (2).
- [ ] **Step 5: Commit** — `feat(subscriptions): magic-link manage-plan portal`

---

### Task 6: Full suite green + flag doc

**Files:** Create `docs/subscriptions.md`

- [ ] **Step 1:** Run all Plan 3 tests + the Plan 1/2 suites; all green.
- [ ] **Step 2:** Write `docs/subscriptions.md`: the `SUBSCRIPTIONS_ENABLED` flag (requires `PRICING_ENGINE_CHECKOUT` + `STRIPE_ACTIVE` + `STRIPE_SECRET_KEY`), the vault+scheduler model, the escalating tier, skip/pause/cancel, the daily cron + secret, and the go-live checklist (set the three flags; confirm the Stripe key; run `?dry_run=1` first).
- [ ] **Step 3:** Commit.

---

## Self-review
- **Spec coverage:** vault+scheduler billing (Tasks 1,3,4); cadence 1/2/3 + escalating 5/10/15 + tier reset on cancel / hold on skip-pause (Task 2,4,5); heads-up + receipt + payment-failed emails (Task 4); skip/pause/cadence/cancel portal (Task 5); prices each cycle through `_price_cart` (Task 4) so volume/floors/points all apply; flag-gated + dry-run (Tasks 3-6).
- **Deferred:** full SCA/3DS off-session confirm flow (v1 treats requires_action as failed+notify); in-portal card update (v1 = contact us / future Stripe Customer Portal); auto-points-redeem on subscription orders; rewards-tier referral attribution + cash-out (Plan 4); begin-funnel subscribe entry.
- **Risk:** charges real cards off-session → entire system behind `SUBSCRIPTIONS_ENABLED` (default off); always `?dry_run=1` first; the cron is CRON_SECRET-gated; never advance a subscription on a failed charge.
- **Type consistency:** `tier_for(n)->int`, `subs.create(...)->id`, `subs.list_due(cx,as_of)->rows`, `advance_after_charge`/`consume_skip`/`set_skip_next`/`set_status`/`set_cadence`, `charge_off_session(cus,pm,amount,*,description,metadata)->{id,status,...}`, `_send_subscription_email(to,kind,data)` used identically across tasks.

## Next
Plan 4 — Rewards tiers + points earning at payment-return + referral attribution + cash-out review task; plus begin-funnel checkout on `_price_cart` and the Products-console per-SKU floor UI.
