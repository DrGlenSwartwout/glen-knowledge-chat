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
