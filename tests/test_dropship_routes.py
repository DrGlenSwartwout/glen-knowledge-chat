"""Tests for /api/practitioner/dropship/quote and /api/practitioner/dropship/checkout.

Monkeypatches: _practitioner_session_pid, _pp.portal_data, appmod._dropship.build_dropship_order,
_ingest_order, _STRIPE_ACTIVE.
"""

import app as appmod


def _auth(monkeypatch):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "p1")
    monkeypatch.setattr(appmod._pp, "portal_data",
        lambda pid: {"modules_completed": 0, "email": "doc@x.com", "name": "Doc",
                     "wholesale_unlocked": True,
                     "cart": [{"slug": "brain-boost", "qty": 6}]})


def test_dropship_quote_requires_auth(monkeypatch):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: None)
    assert appmod.app.test_client().post(
        "/api/practitioner/dropship/quote", json={}).status_code == 401


def test_dropship_quote_uses_canonical_practitioner_quote(monkeypatch):
    _auth(monkeypatch)
    captured = {}
    def fake_quote(items, practitioner):
        captured.update({"items": items, "practitioner": practitioner})
        return {"lines": [{"slug": "brain-boost", "qty": 6,
                           "unit_cents": 4000, "line_cents": 24000}],
                "subtotal_cents": 24000}
    monkeypatch.setattr(appmod._dropship, "quote_dropship_cart", fake_quote)

    r = appmod.app.test_client().get("/api/practitioner/dropship/quote")

    assert r.status_code == 200
    assert r.get_json()["subtotal_cents"] == 24000
    assert captured["practitioner"] == {"id": "p1", "modules_completed": 0}


def test_dropship_checkout_ships_to_patient(monkeypatch):
    _auth(monkeypatch)
    monkeypatch.setattr(appmod, "_price_cart",
                        lambda *a, **k: {"shipping_cents": 1300})
    monkeypatch.setattr(appmod._dropship, "build_dropship_order",
        lambda *a, **k: {"ok": True, "invoice_id": "INV", "total": 352.60,
                         "customer_id": "C1", "source": "dropship",
                         "ship_to": k.get("patient_ship"), "method": "zelle",
                         "get_cents": 0, "shipping_cents": k.get("shipping_cents")})
    cap = {}
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: cap.update(kw))
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)
    # stub cart_clear so it doesn't hit a real DB
    monkeypatch.setattr(appmod._pp, "cart_clear", lambda pid: None)
    r = appmod.app.test_client().post(
        "/api/practitioner/dropship/checkout",
        json={"method": "zelle",
              "patient_address": {"name": "Pat", "state": "CA", "country": "US",
                                  "street": "1 Main St", "city": "Los Angeles", "zip": "90001"}})
    assert r.status_code == 200
    assert cap["source"] == "dropship"
    assert cap["address"]["name"] == "Pat"
    assert cap["shipping_cents"] == 1300


def test_dropship_checkout_requires_patient_address(monkeypatch):
    _auth(monkeypatch)
    monkeypatch.setattr(appmod._dropship, "build_dropship_order",
        lambda *a, **k: {"ok": True, "invoice_id": "INV", "total": 100.0,
                         "customer_id": "C1", "source": "dropship",
                         "ship_to": k.get("patient_ship"), "method": "zelle",
                         "get_cents": 0})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)
    monkeypatch.setattr(appmod._pp, "cart_clear", lambda pid: None)
    # POST without patient_address → 400
    r = appmod.app.test_client().post(
        "/api/practitioner/dropship/checkout",
        json={"method": "zelle"})
    assert r.status_code == 400
    data = r.get_json()
    assert data.get("ok") is False
    assert "patient_address" in (data.get("error") or "")


def test_dropship_checkout_blocks_practitioner_as_recipient_without_confirmation(monkeypatch):
    _auth(monkeypatch)
    monkeypatch.setattr(appmod._pp, "cart_clear", lambda pid: None)
    r = appmod.app.test_client().post(
        "/api/practitioner/dropship/checkout",
        json={"method": "zelle",
              "patient_address": {"name": "  DOC ", "state": "CA", "country": "US",
                                  "street": "1 Main St", "city": "Los Angeles",
                                  "zip": "90001"}})
    assert r.status_code == 400
    assert r.get_json()["code"] == "recipient_name_matches_practitioner"


def test_dropship_checkout_allows_confirmed_practitioner_recipient(monkeypatch):
    _auth(monkeypatch)
    monkeypatch.setattr(appmod, "_price_cart",
                        lambda *a, **k: {"shipping_cents": 1300})
    monkeypatch.setattr(appmod._dropship, "build_dropship_order",
        lambda *a, **k: {"ok": True, "invoice_id": "INV", "total": 100.0,
                         "customer_id": "C1", "source": "dropship",
                         "ship_to": k.get("patient_ship"), "method": "zelle",
                         "get_cents": 0, "shipping_cents": k.get("shipping_cents")})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)
    monkeypatch.setattr(appmod._pp, "cart_clear", lambda pid: None)
    r = appmod.app.test_client().post(
        "/api/practitioner/dropship/checkout",
        json={"method": "zelle", "recipient_name_confirmed": True,
              "patient_address": {"name": "Doc", "state": "CA", "country": "US",
                                  "street": "1 Main St", "city": "Los Angeles",
                                  "zip": "90001"}})
    assert r.status_code == 200
