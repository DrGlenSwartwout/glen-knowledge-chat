"""Tests for POST /api/client/<code>/checkout — patient-paid dispensary checkout.

Three scenarios:
  1. Unknown dispensary code → 404
  2. Patient not a member (consent gate) → 403 with need_optin
  3. Happy path (Stripe inactive) → 200; order recorded with source="dispensary"
     and the patient's address.
"""

import importlib
import pytest
import app as appmod


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with minimum env stubs so the app doesn't call real services."""
    appmod.app.config["TESTING"] = True
    appmod.app.config["SECRET_KEY"] = "test"
    # Never call real Stripe, QBO, etc. from the route.
    return appmod.app.test_client()


# ── helpers ──────────────────────────────────────────────────────────────────

_VALID_BODY = {
    "email": "patient@example.com",
    "name": "Pat Patient",
    "method": "zelle",
    "address": {"street": "123 Main St", "city": "Portland", "state": "OR",
                "zip": "97201", "country": "US"},
    "items": [{"slug": "brain-boost", "qty": 1}],
}

_BUILD_CLIENT_ORDER_OK = {
    "ok": True,
    "invoice_id": "INV-001",
    "total": 70.0,
    "customer_id": "CUST-1",
    "margin_cents": 1340,
    "ship_to": {"name": "Pat Patient", "street": "123 Main St",
                "city": "Portland", "state": "OR", "zip": "97201", "country": "US"},
    "source": "dispensary",
    "get_cents": 0,
}


# ── Test 1: unknown code → 404 ────────────────────────────────────────────────

def test_unknown_dispensary_code_returns_404(client, monkeypatch):
    """When the dispensary code has no matching practitioner, return 404."""
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: None)

    resp = client.post(
        "/api/client/BADCODE/checkout",
        json=_VALID_BODY,
        content_type="application/json",
    )
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["ok"] is False


# ── Test 2: consent gate — not a member → 403 ────────────────────────────────

def test_not_a_member_returns_403(client, monkeypatch):
    """Patient who hasn't agreed to ToS is refused with need_optin."""
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: "p1")
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid, **kw: {"modules_completed": 0})
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: False)

    resp = client.post(
        "/api/client/ABC123/checkout",
        json=_VALID_BODY,
        content_type="application/json",
    )
    assert resp.status_code == 403
    data = resp.get_json()
    assert data["ok"] is False
    assert data.get("need_optin") is True


# ── Test 3: happy path (Stripe inactive) → 200 ───────────────────────────────

def test_happy_path_zelle_returns_200_and_records_order(client, monkeypatch):
    """Full happy path with Stripe inactive: zelle pay_instructions returned,
    order recorded via _ingest_order with source='dispensary' and patient address."""
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: "p1")
    monkeypatch.setattr(appmod._pp, "portal_data",
                        lambda pid, **kw: {"modules_completed": 2, "dispensary_code": "ABC123"})
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: True)
    monkeypatch.setattr(appmod._dropship, "build_client_order",
                        lambda items, prac, *, patient, method: dict(_BUILD_CLIENT_ORDER_OK))
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)

    captured = {}

    def _fake_ingest_order(*, source, external_ref, email="", name="",
                           total_cents=0, address=None, channel="retail",
                           get_cents=0, **kw):
        captured["source"] = source
        captured["external_ref"] = external_ref
        captured["email"] = email
        captured["address"] = address
        captured["get_cents"] = get_cents

    monkeypatch.setattr(appmod, "_ingest_order", _fake_ingest_order)

    resp = client.post(
        "/api/client/ABC123/checkout",
        json=_VALID_BODY,
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["invoice_id"] == "INV-001"
    assert data["margin_cents"] == 1340

    # _ingest_order was called with source="dispensary" and the patient's address.
    assert captured["source"] == "dispensary"
    assert captured["external_ref"] == "INV-001"
    assert captured["email"] == "patient@example.com"
    assert (captured["address"] or {}).get("state") == "OR"
