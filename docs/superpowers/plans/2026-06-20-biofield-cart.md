# Begin #4c — Biofield Reveal Match-to-Order Cart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a member order their Biofield-reveal matched remedies as one cart — select remedies + quantities, see the live volume-discounted total, and check out in one Stripe session — reusing the existing cart-checkout engine, behind `BIOFIELD_CART_ENABLED` (dark).

**Architecture:** Two new token-scoped endpoints on the reveal (`order-preview`, `order-checkout`) reuse a cart-checkout core (`_checkout_cart`) extracted DRY out of `reorder_checkout`. Both recompute the member's server-side visible matched set (`_biofield_visible_slugs`) and reject any slug not in it. The front-end adds a checkbox + qty stepper per remedy and a sticky total bar. Stripe metadata `kind="reorder"` reuses the existing `/begin/checkout-return` handler unchanged.

**Tech Stack:** Python 3.11 / Flask (single `app.py`), SQLite (`LOG_DB`), the `dashboard.pricing` engine, `dashboard.stripe_pay`, QBO invoice helpers, pytest. Front-end is vanilla JS in `static/begin-biofield.html`.

## Global Constraints

- No emoji, no em dashes (applies to all code, copy, comments, commit messages).
- Behind `BIOFIELD_CART_ENABLED` (default off). When off: the reveal payload omits cart affordances and both new endpoints return `{ok:false}`. No behavior change when off.
- Checkout additionally requires the existing `PRICING_ENGINE_CHECKOUT` + `_STRIPE_ACTIVE` + membership — no new money path; it rides the engine already live in `/reorder/checkout`.
- Anti-bypass: the visible matched set is recomputed server-side on every call; a slug not currently visible to that member is never priced or ordered.
- Surface is the Biofield reveal ONLY. Do not touch `/begin/match` chat, the $1 trial (#4b), the per-remedy single Order button, the billing/pricing engine, the daily cron, or `/begin/checkout-return`.
- v1 does NOT surface points-redemption or referral-code entry in the reveal cart (the helper accepts them; the endpoints pass defaults).
- Reuse, never duplicate: extract `_checkout_cart` so `reorder_checkout` and the new endpoint share one code path. Existing reorder tests must stay green.
- Test runner: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Tests skip if app is not importable. Mock Stripe / qb / GHL; tmp `LOG_DB` via `monkeypatch.setattr(app_module, "LOG_DB", db)`; toggle the flag via `monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True/False, raising=False)`.

---

## Critical files

- `app.py`
  - Near `BIOFIELD_TRIAL_ENABLED` (~2580): add `BIOFIELD_CART_ENABLED`.
  - Near the biofield helpers (~1425-1476): add `_biofield_unlock_flags(row, email)` + `_biofield_visible_slugs(row, email)`; add `slug` to `_biofield_remedy_payload`'s return.
  - `begin_biofield_reveal` (~1515-1556): use `_biofield_unlock_flags` (behavior-preserving) and add `cart_enabled` to both payloads.
  - Near `reorder_checkout` (~10454-10525): add `_resolve_ship_address` + `_checkout_cart`; rewrite the engine-path body of `reorder_checkout` to call `_checkout_cart`.
  - Near the reveal routes (after `begin_biofield_reveal_top`, ~1572): add `begin_biofield_order_preview` + `begin_biofield_order_checkout`.
- `static/begin-biofield.html`: cart affordances (checkbox + qty stepper per remedy) + the sticky total bar wired to the two endpoints; CSS for the bar.
- `tests/test_biofield_cart.py` (new): all #4c endpoint + helper tests.

---

## Task 1: Visible-set helpers + flag + payload `cart_enabled`

**Files:**
- Modify: `app.py` (add `BIOFIELD_CART_ENABLED`; add `_biofield_unlock_flags` + `_biofield_visible_slugs`; add `slug` to `_biofield_remedy_payload`; refactor `begin_biofield_reveal` to use the flags helper + add `cart_enabled`)
- Test: `tests/test_biofield_cart.py`

**Interfaces:**
- Consumes: `_active_membership_for_email(email)`, `is_member(session_id="", email="")`, `_db_lock`, `LOG_DB`, `dashboard.biofield_reveals` (`init_free_unlocks`, `free_unlock_reveal_id`), `_biofield_verify_token(th)`, `_hash_token(s)`.
- Produces:
  - `BIOFIELD_CART_ENABLED: bool`
  - `_biofield_unlock_flags(row: dict, email: str) -> dict` with keys `paid, first_approved, top_unlocked, free_available` (never raises).
  - `_biofield_visible_slugs(row: dict, email: str) -> list[str]` — paid: every matched slug; else top remedy slug if `top_unlocked`; else `[]` (never raises).
  - `_biofield_remedy_payload(r)` now also returns `slug`.
  - The reveal payload (member path) now includes `"cart_enabled": BIOFIELD_CART_ENABLED`.

- [ ] **Step 1: Write the failing tests**

Add to a new file `tests/test_biofield_cart.py` (the `_load_app`, `_fresh`, `_approved_reveal` helpers mirror `tests/test_biofield_trial.py` exactly):

```python
# tests/test_biofield_cart.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals, subscriptions
    with sqlite3.connect(db) as cx:
        biofield_reveals.init_table(cx)
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        cx.execute(
            "CREATE TABLE IF NOT EXISTS auth_tokens "
            "(token_hash TEXT PRIMARY KEY, email TEXT NOT NULL, purpose TEXT NOT NULL, "
            "extra TEXT, created_at TEXT NOT NULL, expires_at TEXT NOT NULL, consumed_at TEXT)"
        )
        cx.commit()
    return db


def _approved_reveal(app_module, db, email="t@x.com"):
    import secrets as _s
    from datetime import datetime, timezone, timedelta
    from dashboard import biofield_reveals as br
    token = "tk_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        rid, _ = br.upsert(cx, email, "2026-06-20", {"greeting": "Hi", "body": "b"},
                           [{"name": "Top", "slug": "top", "meaning": "m"},
                            {"name": "Deep1", "slug": "deep1", "meaning": "m2"},
                            {"name": "Deep2", "slug": "deep2", "meaning": "m3"}], "s")
        br.set_token(cx, rid, th)
        br.approve_first(cx, rid, "glen")
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def _row(app_module, db, token):
    th = app_module._hash_token(token)
    valid, row = app_module._biofield_verify_token(th)
    assert valid and row is not None
    return row


def test_visible_slugs_paid_returns_all(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    row = _row(app_module, db, _approved_reveal(app_module, db))
    assert app_module._biofield_visible_slugs(row, "t@x.com") == ["top", "deep1", "deep2"]


def test_visible_slugs_free_top_only(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    token = _approved_reveal(app_module, db)
    # Claim the one-time free top unlock for this member so top_unlocked is true.
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        br.init_free_unlocks(cx)
        rid = br.get_by_token_hash(cx, app_module._hash_token(token))["id"]
        br.record_free_unlock(cx, "t@x.com", rid)
        cx.commit()
    row = _row(app_module, db, token)
    assert app_module._biofield_visible_slugs(row, "t@x.com") == ["top"]


def test_visible_slugs_free_locked_returns_empty(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    row = _row(app_module, db, _approved_reveal(app_module, db))
    assert app_module._biofield_visible_slugs(row, "t@x.com") == []


def test_reveal_payload_cart_enabled(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    token = _approved_reveal(app_module, db)
    html = app_module.app.test_client().get(f"/begin/biofield/{token}").get_data(as_text=True)
    assert '"cart_enabled": true' in html
    assert '"slug": "top"' in html  # remedy payload now carries slug
```

(The free-unlock writer is `dashboard.biofield_reveals.record_free_unlock(cx, email, reveal_id)` — the same function `begin_biofield_reveal_top` calls to record the one-time unlock.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py -v`
Expected: FAIL (`AttributeError: module 'app' has no attribute '_biofield_visible_slugs'`, and `cart_enabled` not in payload).

- [ ] **Step 3: Add the flag**

In `app.py`, immediately after the `BIOFIELD_TRIAL_ENABLED = ...` line (~2580), add:

```python
BIOFIELD_CART_ENABLED = os.environ.get("BIOFIELD_CART_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 4: Add `slug` to the remedy payload**

In `app.py`, in `_biofield_remedy_payload` (~1425), change the returned dict to include the slug:

```python
def _biofield_remedy_payload(r):
    """Return {name, meaning, slug, buy_url, page_url} for any remedy dict. Never raises."""
    try:
        slug = (r.get("slug") or "").strip()
        buy_url = f"/begin/buy/{slug}" if slug else "/begin/match"
        page_url = f"/begin/product/{slug}" if slug else "/begin/match"
        return {"name": r.get("name", ""), "meaning": r.get("meaning", ""),
                "slug": slug, "buy_url": buy_url, "page_url": page_url}
    except Exception:
        return None
```

- [ ] **Step 5: Add the unlock-flags + visible-slugs helpers**

In `app.py`, directly after `_biofield_verify_token` (~1477), add:

```python
def _biofield_unlock_flags(row, email):
    """Compute (paid, first_approved, top_unlocked, free_available) for a reveal row + member.
    Single source of truth for reveal visibility. Never raises."""
    from dashboard import biofield_reveals as _br
    email = (email or "").strip().lower()
    first_approved = bool(row.get("first_approved"))
    try:
        paid = bool(_active_membership_for_email(email))
    except Exception:
        paid = False
    fu_rid = None
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _br.init_free_unlocks(cx)
            fu_rid = _br.free_unlock_reveal_id(cx, email)
    except Exception:
        fu_rid = None
    top_unlocked = bool(first_approved and fu_rid == row.get("id"))
    free_available = bool(first_approved and fu_rid is None)
    return {"paid": paid, "first_approved": first_approved,
            "top_unlocked": top_unlocked, "free_available": free_available}


def _biofield_visible_slugs(row, email):
    """Slugs this member may order from the reveal: paid -> all matched remedies;
    else the top remedy if unlocked; else []. Never raises."""
    try:
        remedies = row.get("remedies") or []
        if not remedies:
            return []
        flags = _biofield_unlock_flags(row, email)
        if flags["paid"]:
            return [(r.get("slug") or "").strip() for r in remedies if (r.get("slug") or "").strip()]
        if flags["top_unlocked"]:
            s = (remedies[0].get("slug") or "").strip()
            return [s] if s else []
        return []
    except Exception:
        return []
```

- [ ] **Step 6: Refactor `begin_biofield_reveal` to use the flags helper + add `cart_enabled`**

In `app.py` (~1515-1556), replace the inline free-unlock lookup + paid block with the helper, and add `cart_enabled` to both payloads. Replace the code from `# Member path: compute unlock state` down to the end of the `else:` payload with:

```python
    # Member path: compute unlock state (single source of truth)
    flags = _biofield_unlock_flags(row, email)
    first_approved = flags["first_approved"]
    top_unlocked = flags["top_unlocked"]
    free_available = flags["free_available"]
    paid = flags["paid"]

    if paid:
        all_remedies = row.get("remedies") or []
        payload = {
            "interpretation": row.get("interpretation") or {},
            "blurred_count": 0,
            "first_approved": first_approved,
            "free_available": False,
            "top_unlocked": True,
            "paid": True,
            "trial_enabled": BIOFIELD_TRIAL_ENABLED,
            "cart_enabled": BIOFIELD_CART_ENABLED,
            "remedies": [_biofield_remedy_payload(r) for r in all_remedies],
        }
    else:
        payload = {
            "interpretation": row.get("interpretation") or {},
            "blurred_count": len(row.get("remedies") or []) - (1 if top_unlocked else 0),
            "first_approved": first_approved,
            "free_available": free_available,
            "top_unlocked": top_unlocked,
            "top": _biofield_top_payload(row) if top_unlocked else None,
            "paid": False,
            "trial_enabled": BIOFIELD_TRIAL_ENABLED,
            "cart_enabled": BIOFIELD_CART_ENABLED,
        }
```

(Leave the `_record_entry_unlock("biofield", email)` line and the `__REVEAL__` injection below it untouched.)

- [ ] **Step 7: Run the tests to verify they pass + no reveal regressions**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py tests/test_biofield_trial.py -v`
Expected: PASS (all new tests green; the #4a/#4b reveal+trial tests still green — the refactor is behavior-preserving).

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_biofield_cart.py
git commit -m "feat: begin #4c visible-set helpers + BIOFIELD_CART_ENABLED + cart_enabled payload"
```

---

## Task 2: Extract `_checkout_cart` (DRY) from `reorder_checkout`

**Files:**
- Modify: `app.py` (add `_resolve_ship_address` + `_checkout_cart`; rewrite the engine-path body of `reorder_checkout` to call `_checkout_cart`)
- Test: `tests/test_biofield_cart.py`

**Interfaces:**
- Consumes: `_price_cart`, `_resolve_checkout_coupon_pct`, `qb` (`find_or_create_customer`, `create_invoice`), `_ingest_order`, `_record_referral_if_any`, `_shipping_line`, `_stripe_checkout_url_for_reorder`, `_STRIPE_ACTIVE`, `CheckoutError`, `_bos_orders.list_orders_by_email`, `dashboard.points`.
- Produces:
  - `_resolve_ship_address(email: str, body_address: dict) -> dict` — `body_address` if non-empty, else the member's last order address, else `{}` (never raises).
  - `_checkout_cart(email, cart, *, ship, points_to_redeem_cents=0, referral_code=None) -> dict` returning `{"out": {invoice_id, doc_number, customer_id, total}, "stripe_url": str}`. Raises `CheckoutError` for a pricing problem or an empty priced cart.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_biofield_cart.py`:

```python
def test_checkout_cart_builds_invoice_and_stripe(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    # Fake the pricing + QBO + stripe layers so the helper is exercised in isolation.
    monkeypatch.setattr(app_module, "_price_cart", lambda cart, **kw: {
        "priced": {"lines": [], "subtotal_cents": 5000, "discount_cents": 0,
                   "points_redeemed_cents": 0, "get_cents": 0, "total_cents": 5000},
        "qbo_lines": [{"name": "Top", "amount": 50.0, "qty": 1}],
        "items_rec": [{"name": "Top", "qty": 1, "desc": "Top"}],
        "subtotal_list_cents": 5000, "discount_cents": 0,
        "points_redeemed_cents": 0, "shipping_cents": 1300})
    monkeypatch.setattr(app_module, "_resolve_checkout_coupon_pct", lambda code, email: (None, None))
    monkeypatch.setattr(app_module.qb, "find_or_create_customer", lambda email, name: {"Id": "C1"})
    monkeypatch.setattr(app_module.qb, "create_invoice",
        lambda cust, lines, **kw: {"Id": "INV1", "DocNumber": "1001", "TotalAmt": 63.0})
    monkeypatch.setattr(app_module, "_ingest_order", lambda **kw: None)
    monkeypatch.setattr(app_module, "_record_referral_if_any", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_stripe_checkout_url_for_reorder",
        lambda out, email: "https://stripe.test/sess")
    res = app_module._checkout_cart("t@x.com", [{"slug": "top", "qty": 1}], ship={"name": "T", "country": "US"})
    assert res["stripe_url"] == "https://stripe.test/sess"
    assert res["out"] == {"invoice_id": "INV1", "doc_number": "1001",
                          "customer_id": "C1", "total": 63.0}


def test_checkout_cart_empty_raises(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_price_cart", lambda cart, **kw: {
        "priced": {"lines": [], "subtotal_cents": 0, "discount_cents": 0,
                   "points_redeemed_cents": 0, "get_cents": 0, "total_cents": 0},
        "qbo_lines": [], "items_rec": [], "subtotal_list_cents": 0,
        "discount_cents": 0, "points_redeemed_cents": 0, "shipping_cents": 0})
    monkeypatch.setattr(app_module, "_resolve_checkout_coupon_pct", lambda code, email: (None, None))
    with pytest.raises(app_module.CheckoutError):
        app_module._checkout_cart("t@x.com", [], ship={"country": "US"})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py::test_checkout_cart_builds_invoice_and_stripe -v`
Expected: FAIL (`module 'app' has no attribute '_checkout_cart'`).

- [ ] **Step 3: Add `_resolve_ship_address` + `_checkout_cart`**

In `app.py`, immediately above `def reorder_checkout` (~10454), add:

```python
def _resolve_ship_address(email, body_address):
    """Ship-to: the request address if given, else the member's last order address, else {}."""
    ship = body_address or {}
    if not ship:
        try:
            with sqlite3.connect(LOG_DB) as cx:
                cx.row_factory = sqlite3.Row
                prior = _bos_orders.list_orders_by_email(cx, email, limit=1)
            if prior:
                ship = prior[0].get("address") or {}
        except Exception:
            ship = {}
    return ship or {}


def _checkout_cart(email, cart, *, ship, points_to_redeem_cents=0, referral_code=None):
    """Price a cart through the engine, create the QBO invoice, ingest the order, and mint the
    Stripe URL (metadata kind=reorder -> recorded by /begin/checkout-return). Returns
    {out, stripe_url}. Raises CheckoutError for a pricing problem or an empty priced cart."""
    requested_redeem = int(points_to_redeem_cents or 0)
    if requested_redeem > 0:
        from dashboard import points as _pts_co
        with sqlite3.connect(LOG_DB) as _cx_pts:
            _pts_co.init_points_table(_cx_pts)
            _bal = _pts_co.balance(_cx_pts, email)
        requested_redeem = min(requested_redeem, _bal)
    _ref_pct, _ref_ctx = _resolve_checkout_coupon_pct(referral_code, email)
    pc = _price_cart(cart, ship=ship, coupon_pct=_ref_pct, points_to_redeem_cents=requested_redeem)
    if not pc["qbo_lines"]:
        raise CheckoutError("Your cart is empty or those items are no longer available.")
    cust = qb.find_or_create_customer(email, ship.get("name", ""))
    inv = qb.create_invoice(
        cust,
        pc["qbo_lines"] + _shipping_line(pc["shipping_cents"]),
        allow_online_pay=True,
        email_to=email,
        discount_cents=pc["discount_cents"] + pc["points_redeemed_cents"])
    _ingest_order(source="reorder", external_ref=inv.get("Id"), email=email,
                  name=ship.get("name", ""), items=pc["items_rec"],
                  total_cents=int(round(float(inv.get("TotalAmt") or 0) * 100)),
                  address=ship, channel="retail",
                  get_cents=pc["priced"].get("get_cents", 0),
                  discount_cents=pc["discount_cents"],
                  points_redeemed_cents=pc["points_redeemed_cents"],
                  shipping_cents=pc["shipping_cents"])
    _record_referral_if_any(_ref_ctx, email, inv.get("Id"))
    out = {"invoice_id": inv.get("Id"), "doc_number": inv.get("DocNumber"),
           "customer_id": cust.get("Id"), "total": inv.get("TotalAmt")}
    stripe_url = _stripe_checkout_url_for_reorder(out, email) if _STRIPE_ACTIVE else ""
    return {"out": out, "stripe_url": stripe_url}
```

- [ ] **Step 4: Rewrite the engine path of `reorder_checkout` to call the helper**

In `app.py`, replace the engine-path block inside `reorder_checkout` (the `if os.environ.get("PRICING_ENGINE_CHECKOUT", ...)` body, ~10473-10525) with:

```python
    if os.environ.get("PRICING_ENGINE_CHECKOUT", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            ship = _resolve_ship_address(email, body_address or {})
            requested_redeem = (body.get("points_to_redeem_cents") if isinstance(body, dict) else 0) or 0
            referral_code = body.get("referral_code") if isinstance(body, dict) else None
            try:
                res = _checkout_cart(email, cart, ship=ship,
                                     points_to_redeem_cents=requested_redeem,
                                     referral_code=referral_code)
            except CheckoutError as e:
                return jsonify({"ok": False, "error": str(e)}), 400
            out, stripe_url = res["out"], res["stripe_url"]
            _pe = {"payment_error": _CARD_UNAVAILABLE} if (_STRIPE_ACTIVE and not stripe_url) else {}
            return jsonify({"ok": True, "stripe_url": stripe_url, **out, **_pe})
        except Exception as e:
            app.logger.exception("reorder checkout (engine) failed")
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
```

- [ ] **Step 5: Run the new helper tests + the full reorder suite (no regressions)**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py -k checkout_cart -v && doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k reorder -v`
Expected: PASS (the two helper tests, and every existing `reorder` test still green through the extracted path).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_biofield_cart.py
git commit -m "refactor: extract _checkout_cart from reorder_checkout (DRY for begin #4c)"
```

---

## Task 3: `POST /begin/biofield/<token>/order-preview`

**Files:**
- Modify: `app.py` (add `begin_biofield_order_preview` after `begin_biofield_reveal_top`, ~1572)
- Test: `tests/test_biofield_cart.py`

**Interfaces:**
- Consumes: `BIOFIELD_CART_ENABLED`, `_hash_token`, `_biofield_verify_token`, `_biofield_visible_slugs`, `_resolve_ship_address`, `_price_cart`.
- Produces: `POST /begin/biofield/<token>/order-preview` returning `{ok, lines:[{slug,name,qty,list_cents,line_total_cents,savings_cents}], subtotal_cents, shipping_cents, savings_cents, total_cents}` (never raises; flag off -> `{ok:false}`; bad token -> 404).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_biofield_cart.py`:

```python
def test_preview_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", False, raising=False)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/order-preview", json={"items": [{"slug": "top", "qty": 1}]})
    assert r.get_json().get("ok") is False


def test_preview_prices_visible_set(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})  # paid -> all visible
    seen = {}
    def _fake_price(cart, **kw):
        seen["cart"] = cart
        return {"priced": {"lines": [{"slug": "top", "name": "Top", "qty": 1,
                                      "list_cents": 6997, "line_total_cents": 5997}],
                           "subtotal_cents": 5997, "discount_cents": 1000,
                           "points_redeemed_cents": 0, "get_cents": 0, "total_cents": 5997},
                "shipping_cents": 1300}
    monkeypatch.setattr(app_module, "_price_cart", _fake_price)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/order-preview",
                                          json={"items": [{"slug": "top", "qty": 1}]})
    body = r.get_json()
    assert body["ok"] is True
    assert body["subtotal_cents"] == 5997 and body["shipping_cents"] == 1300
    assert body["total_cents"] == 5997 + 1300 and body["savings_cents"] == 1000
    assert body["lines"][0] == {"slug": "top", "name": "Top", "qty": 1,
                                "list_cents": 6997, "line_total_cents": 5997, "savings_cents": 1000}


def test_preview_rejects_invisible_slug(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)  # free -> nothing visible
    captured = {}
    monkeypatch.setattr(app_module, "_price_cart",
        lambda cart, **kw: captured.setdefault("cart", cart) or {
            "priced": {"lines": [], "subtotal_cents": 0, "discount_cents": 0,
                       "points_redeemed_cents": 0, "get_cents": 0, "total_cents": 0},
            "shipping_cents": 0})
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/order-preview",
                                          json={"items": [{"slug": "deep1", "qty": 1}]})
    body = r.get_json()
    # No visible slugs -> empty cart, _price_cart not called, zeroed totals.
    assert body["ok"] is True and body["lines"] == [] and body["total_cents"] == 0
    assert "cart" not in captured


def test_preview_bad_token_404(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    r = app_module.app.test_client().post("/begin/biofield/nope/order-preview", json={"items": []})
    assert r.status_code == 404 and r.get_json().get("ok") is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py -k preview -v`
Expected: FAIL (404 route not found / endpoint missing).

- [ ] **Step 3: Add the endpoint**

In `app.py`, directly after the `begin_biofield_reveal_top` route ends (~the function returning the one-time free unblock), add:

```python
@app.route("/begin/biofield/<token>/order-preview", methods=["POST"])
def begin_biofield_order_preview(token):
    """Live volume-priced preview of the member's matched-set cart. Never raises."""
    if not BIOFIELD_CART_ENABLED:
        return jsonify({"ok": False}), 200
    try:
        th = _hash_token((token or "").strip())
        valid, row = _biofield_verify_token(th)
        if not valid or row is None:
            return jsonify({"ok": False, "error": "invalid"}), 404
        email = (row.get("email") or "").strip().lower()
        body = request.get_json(silent=True) or {}
        visible = set(_biofield_visible_slugs(row, email))
        items = []
        for it in (body.get("items") or []):
            s = (it.get("slug") or "").strip()
            if s and s in visible:
                items.append({"slug": s, "qty": max(1, min(int(it.get("qty") or 1), 99))})
        if not items:
            return jsonify({"ok": True, "lines": [], "subtotal_cents": 0,
                            "shipping_cents": 0, "savings_cents": 0, "total_cents": 0})
        ship = _resolve_ship_address(email, {})
        pc = _price_cart(items, ship=ship)
        priced = pc["priced"]
        lines = [{"slug": ln.get("slug"), "name": ln.get("name"), "qty": ln.get("qty"),
                  "list_cents": int(ln.get("list_cents") or 0),
                  "line_total_cents": int(ln.get("line_total_cents") or 0),
                  "savings_cents": int(ln.get("list_cents") or 0) - int(ln.get("line_total_cents") or 0)}
                 for ln in priced.get("lines", [])]
        subtotal = int(priced.get("subtotal_cents") or 0)
        shipping = int(pc.get("shipping_cents") or 0)
        return jsonify({"ok": True, "lines": lines, "subtotal_cents": subtotal,
                        "shipping_cents": shipping,
                        "savings_cents": int(priced.get("discount_cents") or 0),
                        "total_cents": subtotal + shipping})
    except Exception as e:
        print(f"[biofield-cart] preview failed: {e!r}", flush=True)
        return jsonify({"ok": False}), 200
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py -k preview -v`
Expected: PASS (all four preview tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_biofield_cart.py
git commit -m "feat: begin #4c order-preview endpoint (visible-set priced preview)"
```

---

## Task 4: `POST /begin/biofield/<token>/order-checkout`

**Files:**
- Modify: `app.py` (add `begin_biofield_order_checkout` next to the preview route)
- Test: `tests/test_biofield_cart.py`

**Interfaces:**
- Consumes: `BIOFIELD_CART_ENABLED`, `_hash_token`, `_biofield_verify_token`, `is_member`, `_biofield_visible_slugs`, `_resolve_ship_address`, `_checkout_cart`, `_STRIPE_ACTIVE`, `_CARD_UNAVAILABLE`, `CheckoutError`.
- Produces: `POST /begin/biofield/<token>/order-checkout` returning `{ok, stripe_url, invoice_id, doc_number, customer_id, total}` on success; 403 `{need_optin:true}` for a non-member; 400 for an empty/all-invisible cart; `{ok:false}` flag-off; 404 bad token. Wrapped, never 500s on an unexpected error path (returns 500 JSON only as the final catch).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_biofield_cart.py`:

```python
def test_checkout_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", False, raising=False)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/order-checkout", json={"items": [{"slug": "top", "qty": 1}]})
    assert r.get_json().get("ok") is False


def test_checkout_non_member_403(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": False)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/order-checkout", json={"items": [{"slug": "top", "qty": 1}]})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_checkout_empty_cart_400(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)  # free -> nothing visible
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    token = _approved_reveal(app_module, db)
    # 'deep1' is not visible to a free member -> filtered out -> empty -> 400
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/order-checkout", json={"items": [{"slug": "deep1", "qty": 1}]})
    assert r.status_code == 400 and r.get_json().get("ok") is False


def test_checkout_member_returns_stripe_url(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})  # paid -> all visible
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    passed = {}
    def _fake_checkout(email, cart, *, ship, **kw):
        passed["cart"] = cart
        return {"out": {"invoice_id": "INV9", "doc_number": "9", "customer_id": "C9", "total": 120.0},
                "stripe_url": "https://stripe.test/checkout"}
    monkeypatch.setattr(app_module, "_checkout_cart", _fake_checkout)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/order-checkout",
        json={"items": [{"slug": "top", "qty": 2}, {"slug": "deep1", "qty": 1}, {"slug": "evil", "qty": 9}]})
    body = r.get_json()
    assert body["ok"] is True and body["stripe_url"] == "https://stripe.test/checkout"
    assert body["invoice_id"] == "INV9"
    # 'evil' is not in the matched set -> dropped; visible slugs only.
    assert {c["slug"] for c in passed["cart"]} == {"top", "deep1"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py -k "checkout_flag_off or non_member or empty_cart or returns_stripe" -v`
Expected: FAIL (route missing).

- [ ] **Step 3: Add the endpoint**

In `app.py`, directly after `begin_biofield_order_preview`, add:

```python
@app.route("/begin/biofield/<token>/order-checkout", methods=["POST"])
def begin_biofield_order_checkout(token):
    """Check out the member's matched-set cart in one Stripe session (kind=reorder)."""
    if not BIOFIELD_CART_ENABLED:
        return jsonify({"ok": False}), 200
    try:
        th = _hash_token((token or "").strip())
        valid, row = _biofield_verify_token(th)
        if not valid or row is None:
            return jsonify({"ok": False, "error": "invalid"}), 404
        email = (row.get("email") or "").strip().lower()
        _sid = (request.cookies.get("amg_session") or "").strip()
        if not is_member(_sid, email):
            return jsonify({"ok": False, "need_optin": True,
                            "error": "Please agree to our Terms to continue your order."}), 403
        body = request.get_json(silent=True) or {}
        visible = set(_biofield_visible_slugs(row, email))
        items = []
        for it in (body.get("items") or []):
            s = (it.get("slug") or "").strip()
            if s and s in visible:
                items.append({"slug": s, "qty": max(1, min(int(it.get("qty") or 1), 99))})
        if not items:
            return jsonify({"ok": False,
                            "error": "Your cart is empty or those items are no longer available."}), 400
        ship = _resolve_ship_address(email, body.get("address") or {})
        try:
            res = _checkout_cart(email, items, ship=ship)
        except CheckoutError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        out, stripe_url = res["out"], res["stripe_url"]
        _pe = {"payment_error": _CARD_UNAVAILABLE} if (_STRIPE_ACTIVE and not stripe_url) else {}
        return jsonify({"ok": True, "stripe_url": stripe_url, **out, **_pe})
    except Exception as e:
        app.logger.exception("biofield order checkout failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py -v`
Expected: PASS (every #4c test green).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_biofield_cart.py
git commit -m "feat: begin #4c order-checkout endpoint (matched-set one-session checkout)"
```

---

## Task 5: Front-end — checkboxes + qty steppers + sticky total bar

**Files:**
- Modify: `static/begin-biofield.html` (cart affordances in `renderRemedyInto` usage + a sticky bar + CSS + the preview/checkout wiring)
- Test: `tests/test_biofield_cart.py` (serve assertions only; the live interaction is a manual visual pass)

**Interfaces:**
- Consumes: the reveal payload fields `cart_enabled` (bool), `paid` (bool), `remedies[]` / `top` (each with `slug`, `name`, `meaning`, `buy_url`, `page_url`), and the two endpoints `/begin/biofield/<token>/order-preview` + `/order-checkout`. The token is already in `window.location.pathname` (`/begin/biofield/<token>`).
- Produces: a member-visible cart on the reveal when `cart_enabled` is true.

- [ ] **Step 1: Write the failing serve test**

Add to `tests/test_biofield_cart.py`:

```python
def test_reveal_html_has_cart_wiring(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    token = _approved_reveal(app_module, db)
    html = app_module.app.test_client().get(f"/begin/biofield/{token}").get_data(as_text=True)
    # The page ships the cart JS (endpoint paths) and the sticky bar element id.
    assert "order-preview" in html and "order-checkout" in html
    assert "bf-cart-bar" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py::test_reveal_html_has_cart_wiring -v`
Expected: FAIL (`order-preview` / `bf-cart-bar` not in the served HTML).

- [ ] **Step 3: Add the CSS for the sticky bar**

In `static/begin-biofield.html`, inside the `<style>` block, add:

```css
  .bf-cart-row{ display:flex; align-items:center; gap:10px; margin-top:10px; }
  .bf-cart-row input[type=number]{ width:64px; padding:6px; }
  #bf-cart-bar{ position:sticky; bottom:0; left:0; right:0; margin-top:24px;
    background:var(--card,#fff); border-top:1px solid var(--border,#ddd);
    padding:14px 18px; display:none; align-items:center; justify-content:space-between; gap:14px; }
  #bf-cart-bar.show{ display:flex; }
  #bf-cart-bar .bf-cart-total{ font-weight:600; }
  #bf-cart-bar .bf-cart-save{ color:#2e7d32; font-size:0.9em; }
  #bf-cart-bar button{ padding:10px 18px; cursor:pointer; }
  #bf-cart-bar button[disabled]{ opacity:0.5; cursor:default; }
```

- [ ] **Step 4: Add the cart affordances + the sticky bar + wiring**

In `static/begin-biofield.html`, make these edits inside the member-render script (the function that reads `var data = reveal;`):

(a) Extend `renderRemedyInto` to optionally add a checkbox + qty stepper when the cart is enabled and the remedy has a slug. Replace the function with:

```javascript
      var cartItems = {};  // slug -> qty (only checked items)

      function renderRemedyInto(container, rem) {
        var nameLink = document.createElement("a");
        nameLink.className = "remedy-name-link";
        nameLink.setAttribute("href", rem.page_url || "/begin/match");
        nameLink.setAttribute("target", "_blank");
        nameLink.setAttribute("rel", "noopener");
        nameLink.textContent = rem.name || "";
        container.appendChild(nameLink);

        if (rem.meaning) {
          var meaningEl = document.createElement("p");
          meaningEl.className = "remedy-meaning";
          meaningEl.textContent = rem.meaning;
          container.appendChild(meaningEl);
        }

        var orderBtn = document.createElement("a");
        orderBtn.className = "buy-btn";
        orderBtn.setAttribute("href", rem.buy_url || "/begin/match");
        orderBtn.textContent = "Order";
        container.appendChild(orderBtn);

        if (data.cart_enabled && rem.slug) {
          var row = document.createElement("div");
          row.className = "bf-cart-row";
          var cb = document.createElement("input");
          cb.type = "checkbox"; cb.checked = true;
          var qty = document.createElement("input");
          qty.type = "number"; qty.min = "1"; qty.value = "1";
          var lbl = document.createElement("span");
          lbl.textContent = "Add to my order";
          cartItems[rem.slug] = 1;
          function sync() {
            if (cb.checked) { cartItems[rem.slug] = Math.max(1, parseInt(qty.value || "1", 10)); }
            else { delete cartItems[rem.slug]; }
            refreshCartBar();
          }
          cb.addEventListener("change", sync);
          qty.addEventListener("input", sync);
          row.appendChild(cb); row.appendChild(qty); row.appendChild(lbl);
          container.appendChild(row);
        }
      }
```

(b) After the remedy cards are appended to `root` (after the `if (data.paid) { ... } else { ... }` block), add the sticky bar + its logic:

```javascript
      // -- Matched-set cart (sticky bar) --
      if (data.cart_enabled) {
        var token = (window.location.pathname.split("/")[3] || "");
        var bar = document.createElement("div");
        bar.id = "bf-cart-bar";
        var info = document.createElement("div");
        var totalEl = document.createElement("span"); totalEl.className = "bf-cart-total";
        var saveEl = document.createElement("span"); saveEl.className = "bf-cart-save";
        info.appendChild(totalEl); info.appendChild(document.createTextNode(" "));
        info.appendChild(saveEl);
        var btn = document.createElement("button");
        btn.textContent = "Order my remedies";
        bar.appendChild(info); bar.appendChild(btn);
        document.body.appendChild(bar);

        var previewTimer = null;
        function items() {
          var out = [];
          for (var s in cartItems) { out.push({ slug: s, qty: cartItems[s] }); }
          return out;
        }
        window.refreshCartBar = function () {
          var its = items();
          if (!its.length) { bar.classList.remove("show"); return; }
          bar.classList.add("show");
          if (previewTimer) { clearTimeout(previewTimer); }
          previewTimer = setTimeout(function () {
            fetch("/begin/biofield/" + token + "/order-preview", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ items: its })
            }).then(function (r) { return r.json(); }).then(function (p) {
              if (!p || !p.ok) { return; }
              var n = (p.lines || []).length;
              totalEl.textContent = "Order " + n + " remedies - $" + (p.total_cents / 100).toFixed(2);
              saveEl.textContent = p.savings_cents > 0 ? ("You save $" + (p.savings_cents / 100).toFixed(2) + " ordering together") : "";
            }).catch(function () {});
          }, 250);
        };
        btn.addEventListener("click", function () {
          var its = items();
          if (!its.length) { return; }
          btn.disabled = true;
          fetch("/begin/biofield/" + token + "/order-checkout", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ items: its })
          }).then(function (r) { return r.json(); }).then(function (res) {
            if (res && res.ok && res.stripe_url) { window.location.href = res.stripe_url; }
            else { btn.disabled = false; alert((res && res.error) || "Could not start checkout."); }
          }).catch(function () { btn.disabled = false; });
        });
        refreshCartBar();
      }
```

(NOTE on the token index: the path is `/begin/biofield/<token>`, so `pathname.split("/")` is `["", "begin", "biofield", "<token>"]` -> index 3. Verify against the served path before finishing.)

- [ ] **Step 5: Run the serve test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py -v`
Expected: PASS (the serve assertion plus the whole #4c suite).

- [ ] **Step 6: Commit**

```bash
git add static/begin-biofield.html tests/test_biofield_cart.py
git commit -m "feat: begin #4c reveal cart UI - checkboxes + qty steppers + sticky total bar"
```

---

## Verification

- Per task: the named `pytest` target passes (doppler + venv).
- Full sweep after Task 5: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_cart.py tests/test_biofield_trial.py -v` and a `-k reorder` regression run — all green (the DRY extraction did not change reorder behavior).
- Final Opus whole-branch review (focus: the visible-set anti-bypass holds on BOTH endpoints; the `_checkout_cart` extraction is behavior-identical to the old reorder engine path; flag-off is fully dark on both endpoints + the payload; preview never raises and never creates an invoice; checkout is wrapped and never 500s except the final catch; XSS-safe front-end — `textContent`/`setAttribute` only, no `innerHTML` of dynamic data; no emoji, no em dashes).
- Manual visual pass (live, after flag flip): the checkboxes + qty steppers render per remedy, the sticky bar shows the live volume-discounted total + savings line, "Order my remedies" goes to Stripe, and the per-remedy single Order button still works.
- Ship via PR + merge to `main` (auto-deploys dark behind `BIOFIELD_CART_ENABLED`); gentle `/begin/biofield/<token>` + `order-preview` probe per the warm-up rule (flag off -> `{ok:false}`); update memory.

## Build order
Task 1 (helpers + flag + payload) -> Task 2 (DRY `_checkout_cart`) -> Task 3 (preview) -> Task 4 (checkout) -> Task 5 (front-end). Tasks 3 and 4 both depend on Tasks 1 and 2; Task 5 depends on all prior. Go-live = flip `BIOFIELD_CART_ENABLED=true` in Doppler `remedy-match/prd` (with `PRICING_ENGINE_CHECKOUT` + `STRIPE_ACTIVE` already on), after the manual visual pass.
