# Repoint Paid tier at continuous-care — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repoint the program page's Paid tier from the half-wired `live-group` checkout to the complete continuous-care-monthly flow, via a new portal-token wrapper route, shipped dark behind a new flag.

**Architecture:** Extract the continuous-care Stripe-session construction into a shared helper (single source of truth on the money path). Add a token wrapper route `POST /portal/offer/continuous-care/checkout` that resolves the member from their portal token and starts that checkout (term 12), returning `stripe_url`. Gate the Paid tier on a new `PROGRAM_PAID_LIVE_ENABLED` flag. Fulfillment (return handler, webhook, idempotency, immediate grant) already exists and is untouched.

**Tech Stack:** Python 3 / Flask (`app.py`), sqlite3, vanilla-JS static page, pytest. Spec: `docs/superpowers/specs/2026-07-10-portal-paid-continuous-care-repoint-design.md`.

## Global Constraints

- **Money-path safety:** the existing `POST /continuous-care/checkout` externally observable behavior must stay identical after the helper extraction (same `{"ok":true,"url":...}` success, same `{"ok":false,"error":"invalid"|"unavailable"|"checkout_failed"}` cases). Verified by a regression test.
- **No dead buy buttons:** the wrapper route 404s unless `_program_paid_live_enabled()` AND `CONTINUOUS_CARE_MONTHLY_ENABLED` AND `_STRIPE_ACTIVE`; the Paid tier renders `available` only when `_program_paid_live_enabled() and CONTINUOUS_CARE_MONTHLY_ENABLED`.
- **Ships dark:** `PROGRAM_PAID_LIVE_ENABLED` defaults off (truthy tuple `("1","true","yes","on")`); presence-checked, never print its value. Durable flip = `doppler secrets set` (Doppler is source; Render is pruned).
- **Copy rules:** no em dashes, no ALL CAPS words, no "Hook:" label. The continuous-care description string is `f"Remedy Match Continuous Care - {term_months} month (monthly)"` (hyphen with spaces) — copy verbatim.
- **Term = 12** hardcoded for the portal wrapper.
- **Test commands:** app-importing tests → `doppler run -p remedy-match -c dev -- python3 -m pytest tests/<f> -q`; pure `program_tiers` test → bare `python3 -m pytest tests/<f> -q`.

---

## File Structure

- **Modify** `app.py` — add `_continuous_care_checkout_session(email, term_months)` helper; refactor `continuous_care_checkout` (`app.py:3396`) to call it; add `_program_paid_live_enabled()` (near `_portal_program_page_enabled`, `app.py:15263`); add wrapper route `POST /portal/offer/continuous-care/checkout` (after the live-group route, `app.py:19595`); change the Paid gate in `api_portal_program` (`app.py:16198`).
- **Modify** `dashboard/program_tiers.py` — rename param `paid_enabled`→`paid_live`; repoint Paid `checkout_path`; add term note copy.
- **Create** `tests/test_continuous_care_repoint.py` — existing-route regression + wrapper-route tests (doppler).
- **Modify** `tests/test_program_tiers.py` — param rename + new checkout_path/copy assertions (bare).
- **Modify** `tests/test_program_page_routes.py` — update Paid-gate tests to the new flag.

---

### Task 1: Shared continuous-care session helper + refactor existing route

**Files:**
- Modify: `app.py` (add helper near `continuous_care_checkout` at `app.py:3396`; refactor that route's body)
- Test: `tests/test_continuous_care_repoint.py` (new)

**Interfaces:**
- Produces: `_continuous_care_checkout_session(email: str, term_months: int) -> dict` (the Stripe session dict; caller reads `.get("url")`).

- [ ] **Step 1: Write the failing regression test**

```python
# tests/test_continuous_care_repoint.py
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_existing_continuous_care_checkout_unchanged(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    from dashboard import stripe_pay
    seen = {}
    def _fake(amount, **kw):
        seen["amount"] = amount
        seen["metadata"] = kw.get("metadata")
        return {"url": "https://stripe.test/existing"}
    monkeypatch.setattr(stripe_pay, "create_checkout_session", _fake)
    r = c.post("/continuous-care/checkout", json={"email": "a@x.com", "term_months": 12})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["url"] == "https://stripe.test/existing"
    assert seen["metadata"]["kind"] == "continuous_care_monthly"
    assert seen["metadata"]["term_months"] == "12"


def test_existing_continuous_care_checkout_invalid(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    r = c.post("/continuous-care/checkout", json={"email": "", "term_months": 12})
    assert r.get_json() == {"ok": False, "error": "invalid"}
```

- [ ] **Step 2: Run to verify it passes on current code (baseline), then we refactor without breaking it**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_continuous_care_repoint.py -q`
Expected: PASS (2 passed) — this test pins the CURRENT behavior before refactor. (If it fails now, the test is wrong — fix the test to match current behavior before touching app.py.)

- [ ] **Step 3: Extract the helper and refactor the route**

Add the helper just above `continuous_care_checkout` (`app.py:3396`):

```python
def _continuous_care_checkout_session(email, term_months):
    """Create the continuous-care Stripe checkout session (mode=payment: $99 now +
    card vault). Single source of truth for metadata / save_card / success+cancel
    URLs shared by /continuous-care/checkout and the portal wrapper. Returns the
    session dict (read .get("url"))."""
    from dashboard import stripe_pay as _sp, prepay as _pp
    base = PUBLIC_BASE_URL.rstrip("/")
    return _sp.create_checkout_session(
        _pp.MONTHLY_ANCHOR_CENTS, customer_email=email,
        description=f"Remedy Match Continuous Care - {term_months} month (monthly)",
        metadata={"email": email, "kind": "continuous_care_monthly",
                  "term_months": str(term_months)},
        success_url=f"{base}/continuous-care/return?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/",
        save_card=True)
```

Then replace the inline `try/except` construction inside `continuous_care_checkout` (currently `app.py:3405-3425`, the block from `from dashboard import stripe_pay as _sp, prepay as _pp` through the `except`) so the body reads:

```python
    if not email or term_months not in (6, 12):
        return jsonify({"ok": False, "error": "invalid"}), 200
    try:
        sess = _continuous_care_checkout_session(email, term_months)
        return jsonify({"ok": True, "url": sess.get("url")})
    except Exception as e:
        print(f"[continuous-care] checkout failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "checkout_failed"}), 200
```

(Keep the earlier lines of the route unchanged: the `CONTINUOUS_CARE_MONTHLY_ENABLED and _STRIPE_ACTIVE` guard returning `{"ok":false,"error":"unavailable"},200`, and the `email`/`term_months` parsing from the request body. Only the Stripe-construction block moves into the helper. The now-unused local `from dashboard import stripe_pay as _sp, prepay as _pp` import in the route body should be removed since the helper owns it.)

- [ ] **Step 4: Run the regression test to verify behavior is unchanged**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_continuous_care_repoint.py -q`
Expected: PASS (2 passed) — same responses, now via the helper.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_continuous_care_repoint.py
git commit -m "refactor(cc): extract shared _continuous_care_checkout_session helper"
```

---

### Task 2: Flag helper + portal-token wrapper route

**Files:**
- Modify: `app.py` (add `_program_paid_live_enabled()` near `app.py:15263`; add wrapper route after the live-group route, `app.py:19595`)
- Test: `tests/test_continuous_care_repoint.py` (append)

**Interfaces:**
- Consumes: `_continuous_care_checkout_session` (Task 1); existing `resolve_identity`, `_client_login_enabled`, `CONTINUOUS_CARE_MONTHLY_ENABLED`, `_STRIPE_ACTIVE`, `LOG_DB`.
- Produces: `_program_paid_live_enabled() -> bool`; route `POST /portal/offer/continuous-care/checkout` returning `{"ok":true,"stripe_url":...}`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_continuous_care_repoint.py`)**

```python
def _seed_portal(appmod, email, name="Test"):
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx)
        token, _id = cp.upsert_portal(cx, email, name, {})
    return token


def test_cc_wrapper_404_when_flag_off(client, monkeypatch):
    c, appmod = client
    monkeypatch.delenv("PROGRAM_PAID_LIVE_ENABLED", raising=False)
    r = c.post("/portal/offer/continuous-care/checkout?token=whatever")
    assert r.status_code == 404


def test_cc_wrapper_returns_stripe_url(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PROGRAM_PAID_LIVE_ENABLED", "1")
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    from dashboard import stripe_pay
    captured = {}
    def _fake(amount, **kw):
        captured["metadata"] = kw.get("metadata")
        return {"url": "https://stripe.test/cc"}
    monkeypatch.setattr(stripe_pay, "create_checkout_session", _fake)
    tok = _seed_portal(appmod, "cc@x.com")
    r = c.post(f"/portal/offer/continuous-care/checkout?token={tok}")
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://stripe.test/cc"
    assert captured["metadata"]["term_months"] == "12"
    assert captured["metadata"]["email"] == "cc@x.com"


def test_cc_wrapper_404_for_bad_token(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PROGRAM_PAID_LIVE_ENABLED", "1")
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    r = c.post("/portal/offer/continuous-care/checkout?token=nope")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_continuous_care_repoint.py -q`
Expected: the 3 new tests FAIL (route/flag not defined → 404 for all, so the stripe_url test fails; bad-token test may pass trivially).

- [ ] **Step 3: Add the flag helper**

After `_portal_program_page_enabled()` (`app.py:15263`):

```python
def _program_paid_live_enabled() -> bool:
    """Whether the program page's Paid tier is a live, sellable Join. Dark by default."""
    return os.environ.get("PROGRAM_PAID_LIVE_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")
```

- [ ] **Step 4: Add the wrapper route**

After `portal_group_join_checkout` (i.e. after `app.py:19595`, before `portal_group_join_return`):

```python
@app.route("/portal/offer/continuous-care/checkout", methods=["POST"])
def portal_continuous_care_checkout():
    """Portal-token wrapper: start the continuous-care-monthly checkout for the
    member resolved from their portal token. Dark until PROGRAM_PAID_LIVE_ENABLED."""
    if not (_program_paid_live_enabled() and CONTINUOUS_CARE_MONTHLY_ENABLED and _STRIPE_ACTIVE):
        return jsonify({"error": "not found"}), 404
    token = request.args.get("token", "") or (request.get_json(silent=True) or {}).get("token", "")
    sess_cookie = request.cookies.get("rm_portal_session", "")
    from dashboard import portal_identity as _pi
    with sqlite3.connect(LOG_DB) as cx:
        ident = _pi.resolve_identity(cx, token=token, session_token=sess_cookie,
                                     client_login_enabled=_client_login_enabled())
    if ident is None:
        return jsonify({"error": "not found"}), 404
    try:
        sess = _continuous_care_checkout_session(ident.email, 12)
    except Exception:
        app.logger.exception("continuous-care portal checkout failed")
        return jsonify({"error": "Could not start checkout. Please reach out and we'll help."}), 502
    return jsonify({"ok": True, "stripe_url": sess.get("url", "")})
```

- [ ] **Step 5: Run to verify all pass**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_continuous_care_repoint.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_continuous_care_repoint.py
git commit -m "feat(program): flag + portal-token continuous-care checkout wrapper"
```

---

### Task 3: Repoint Paid tier + update the endpoint caller

**Files:**
- Modify: `dashboard/program_tiers.py` (param rename, checkout_path, copy)
- Modify: `app.py` (`api_portal_program` caller at `app.py:16195-16199`)
- Test: `tests/test_program_tiers.py` (bare), `tests/test_program_page_routes.py` (doppler)

**Interfaces:**
- Consumes: `_program_paid_live_enabled` + `CONTINUOUS_CARE_MONTHLY_ENABLED` (from Task 2) and the wrapper route path (Task 2).
- Produces: `program_blocks(*, paid_owned, family_owned, paid_live, family_enabled)` with Paid pointing at `/portal/offer/continuous-care/checkout`.

- [ ] **Step 1: Write/adjust the failing pure tests (`tests/test_program_tiers.py`)**

Rename `paid_enabled=`→`paid_live=` in every existing call in this file, and update the Paid assertions. Replace the paid-related tests with:

```python
def test_paid_available_points_at_continuous_care():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_live=True, family_enabled=True))
    assert t["paid"]["state"] == "available"
    assert t["paid"]["checkout_path"] == "/portal/offer/continuous-care/checkout"
    assert t["paid"]["cta_kind"] == "checkout_post"
    assert any("12" in b for b in t["paid"]["benefits"])  # term note present


def test_paid_coming_soon_when_not_live():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_live=False, family_enabled=True))
    assert t["paid"]["state"] == "coming_soon"


def test_paid_owned_wins():
    t = _by_key(pt.program_blocks(
        paid_owned=True, family_owned=False,
        paid_live=True, family_enabled=True))
    assert t["paid"]["state"] == "owned"
```

(Also update `test_cta_kinds_and_family_has_no_checkout_route`, `test_free_is_always_owned`, `test_family_*`, `test_current_tier_key_*` — anywhere they call `program_blocks(...)` with `paid_enabled=`, change to `paid_live=`. Grep the file for `paid_enabled` and replace all.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_program_tiers.py -q`
Expected: FAIL — `program_blocks() got an unexpected keyword argument 'paid_live'` (and checkout_path mismatch).

- [ ] **Step 3: Update `dashboard/program_tiers.py`**

Rename the signature param and Paid state input, repoint the path, add the term note:

```python
def program_blocks(*, paid_owned, family_owned, paid_live, family_enabled):
```

In the `paid` dict: change `"checkout_path"` to `"/portal/offer/continuous-care/checkout"`, and change the benefits list to include the honest term note (append one bullet), and set state via `paid_live`:

```python
    paid = {
        "key": "paid",
        "name": "Guided membership",
        "benefits": [
            "Live group coaching with Dr. Glen",
            "Your protocol re-matched as you progress",
            "Your AI ally and Terrain Restore support",
            "Billed $99 per month for 12 months; your first month is charged today",
        ],
        "price_cents": _po.MEMBERSHIP_PRICE_CENTS,
        "value_cents": None,
        "period": "/mo",
        "cta_label": "Join",
        "checkout_path": "/portal/offer/continuous-care/checkout",
        "cta_kind": "checkout_post",
        "state": _state(paid_owned, paid_live),
    }
```

- [ ] **Step 4: Update the endpoint caller (`app.py:16195-16199`)**

Change the `program_blocks` call so Paid gates on the new flag:

```python
    tiers = _pt.program_blocks(
        paid_owned=paid_owned,
        family_owned=family_owned,
        paid_live=(_program_paid_live_enabled() and CONTINUOUS_CARE_MONTHLY_ENABLED),
        family_enabled=_family_plan_enabled(),
    )
```

- [ ] **Step 5: Update the endpoint tests (`tests/test_program_page_routes.py`)**

The gate changed, so update the affected tests:
- In `test_api_program_returns_tiers_for_free_client`: it currently sets `SUBSCRIPTIONS_ENABLED`/`PORTAL_OFFERS_ENABLED` and asserts `paid` `available`. Change it to reflect the new gate — either (a) assert `paid` is now `"coming_soon"` (drop the offers/subscriptions env sets), or (b) to keep asserting `available`, set `monkeypatch.setenv("PROGRAM_PAID_LIVE_ENABLED","1")` and `monkeypatch.setattr(appmod,"CONTINUOUS_CARE_MONTHLY_ENABLED",True)` and assert `checkout_path == "/portal/offer/continuous-care/checkout"`. Use (b) so a live-Paid path stays covered.
- The earlier `test_api_program_paid_coming_soon_when_offers_off` (added when Paid was gated on offers) is now obsolete — rewrite it as `test_api_program_paid_coming_soon_when_not_live`: with `PROGRAM_PAID_LIVE_ENABLED` unset, seed a client, assert `paid` `state == "coming_soon"`.
- Grep `tests/test_program_page_routes.py` for `PORTAL_OFFERS_ENABLED` and `SUBSCRIPTIONS_ENABLED` in the paid-tier assertions and reconcile them to the new flag.

- [ ] **Step 6: Run both test files**

Run: `python3 -m pytest tests/test_program_tiers.py -q && doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_program_page_routes.py tests/test_continuous_care_repoint.py -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add dashboard/program_tiers.py app.py tests/test_program_tiers.py tests/test_program_page_routes.py
git commit -m "feat(program): Paid tier points at continuous-care, gated on PROGRAM_PAID_LIVE_ENABLED"
```

- [ ] **Step 8: Render-verify (controller, after all tasks)** — with the flag off, Paid shows "coming soon"; with `PROGRAM_PAID_LIVE_ENABLED=1` + continuous-care on, Paid shows the Join button whose `data-checkout` is `/portal/offer/continuous-care/checkout`. ([[feedback_render_the_page_not_the_payload]])

---

## Self-Review

**Spec coverage:**
- New wrapper route + guard + identity + return stripe_url → Task 2. ✓
- Shared helper, existing route unchanged → Task 1 (+ regression test). ✓
- Paid repoint + param rename + copy → Task 3. ✓
- Endpoint gate on new flag → Task 3 Step 4. ✓
- New flag, ships dark → Task 2 Step 3. ✓
- 12-month term → Task 2 route (`_continuous_care_checkout_session(ident.email, 12)`). ✓
- Fulfillment untouched → nothing modifies return/webhook/grants. ✓

**Placeholder scan:** none. Test code is concrete. Copy string matches the verbatim existing description.

**Type consistency:** `program_blocks(..., paid_live, ...)` used identically in Task 3 module + endpoint + both test files. `_continuous_care_checkout_session(email, term_months) -> dict` produced in Task 1, consumed in Task 2. Wrapper returns `{"ok":true,"stripe_url":...}`; the program page JS reads `stripe_url` (matches). Existing route keeps `{"ok":true,"url":...}` (unchanged).
