# tests/test_portal_item_reorder.py
"""Task 6b: per-item portal reorder checkout. POST /api/portal/<token>/checkout
now accepts an OPTIONAL JSON body {items:[{slug, qty}]} that charges exactly
those items (through the SAME repertoire-aware _portal_priced_lines path Task
5b wired up), guarded so a client can only reorder a slug that's actually in
their own portal-channel purchase history — never an arbitrary catalog SKU —
and can never override the server-computed price. Absent body => unchanged
practitioner-curated-cart behavior. Mirrors the fixture/helper patterns in
tests/test_portal_reorder_module.py and tests/test_client_portal_routes.py."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "REPERTOIRE_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_portal(appmod, email="client@example.com", name="Client", content=None):
    from dashboard import client_portal as cp
    content = content or {"greeting": "hi", "video": {}, "layers": []}
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, content)
    cx.close()
    return token


def _seed_order(appmod, *, source, email, slugs_qty, status="done", days_ago=1,
                 unit_cents=6997, external_ref=None):
    """Insert an orders row directly so tests control created_at/source
    precisely. slugs_qty: list of (slug, qty) tuples."""
    from dashboard.orders import init_orders_table
    created = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    items = [{"slug": s, "qty": q, "name": s, "unit_cents": unit_cents}
             for s, q in slugs_qty]
    ref = external_ref or f"o-{source}-{email}-{days_ago}-{slugs_qty[0][0]}"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        init_orders_table(cx)
        cx.execute(
            "INSERT INTO orders (created_at, source, external_ref, channel, email, "
            "items_json, total_cents, status) VALUES (?,?,?,?,?,?,?,?)",
            (created, source, ref, "retail", email,
             json.dumps(items), unit_cents * sum(q for _, q in slugs_qty), status))
        cx.commit()


def _seed_active_membership(appmod, email, *, source="founding"):
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.init_membership_tables(cx)
        cx.execute(
            "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source) "
            "VALUES (?,?,?,?,?,?)",
            (f"mem_{email}", email, datetime.utcnow().isoformat() + "Z", expires,
             "test", source))
        cx.commit()


def _mock_checkout(appmod, monkeypatch, stripe_url="https://checkout.stripe/x",
                    capture_ingest=None):
    """Wires the QBO invoice + Stripe URL mocks shared by every test below.
    Returns the dict `captured["lines"]` will be written into."""
    captured = {}
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    def _fake_invoice(cust, lines, **kw):
        captured["lines"] = lines
        total = sum(l["amount"] * l["qty"] for l in lines)
        return {"Id": "INV1", "DocNumber": "1001", "TotalAmt": total}
    monkeypatch.setattr(qbo_billing, "create_invoice", _fake_invoice)
    if capture_ingest is not None:
        monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: capture_ingest.update(kw))
    else:
        monkeypatch.setattr(appmod, "_ingest_order", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder",
                        lambda out, email: stripe_url)
    return captured


# ── (a) posting a specific entitled item charges exactly that item ──────────

def test_item_reorder_charges_clicked_item_at_member_price(client, monkeypatch):
    c, appmod = client
    email = "reorder-member@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["neuro-magnesium"])
    # Two distinct SKUs in the client's real history — only one gets clicked.
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("neuro-magnesium", 1)], days_ago=5)
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=6)

    ingested = {}
    captured = _mock_checkout(appmod, monkeypatch, stripe_url="https://checkout.stripe/item",
                              capture_ingest=ingested)

    expected_unit = appmod._price_cart(
        [{"slug": "neuro-magnesium", "qty": 1}],
        ship={"country": "US", "state": "TX"}, email=email,
    )["priced"]["lines"][0]["line_total_cents"]
    assert expected_unit < 6997  # member repertoire price must actually be discounted

    r = c.post(f"/api/portal/{tok}/checkout",
               json={"items": [{"slug": "neuro-magnesium", "qty": 2}]})
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://checkout.stripe/item"
    assert len(captured["lines"]) == 1
    assert captured["lines"][0]["qty"] == 2
    # "amount" is the PER-UNIT price (qty is separate) — line total is amount*qty.
    assert round(captured["lines"][0]["amount"] * 100) == expected_unit
    # Only the clicked SKU was ingested as the order — not the other history item.
    assert len(ingested["items"]) == 1
    assert ingested["items"][0]["slug"] == "neuro-magnesium"


# ── (b) no body => unchanged curated-cart behavior ───────────────────────────

def test_item_reorder_absent_body_charges_curated_cart(client, monkeypatch):
    c, appmod = client
    tok = _seed_portal(appmod, content={
        "greeting": "hi", "video": {}, "layers": [],
        "reorder_items": [{"slug": "nous-energy", "qty": 1, "price_cents": 2500}],
    })
    captured = _mock_checkout(appmod, monkeypatch, stripe_url="https://checkout.stripe/curated")

    r = c.post(f"/api/portal/{tok}/checkout")  # no body at all
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://checkout.stripe/curated"
    assert captured["lines"][0]["amount"] == 25.0  # curated special price, unchanged


# ── (c) posting a slug NOT in the client's entitlement is rejected ──────────

def test_item_reorder_rejects_unentitled_slug(client, monkeypatch):
    c, appmod = client
    email = "guard@example.com"
    tok = _seed_portal(appmod, email)
    # This client has only ever bought nous-energy through the portal channel.
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=5)

    invoice_called = {"n": 0}
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    def _fake_invoice(cust, lines, **kw):
        invoice_called["n"] += 1
        return {"Id": "INV1", "DocNumber": "1001", "TotalAmt": 100}
    monkeypatch.setattr(qbo_billing, "create_invoice", _fake_invoice)
    ingest_called = {"n": 0}
    monkeypatch.setattr(appmod, "_ingest_order",
                        lambda *a, **k: ingest_called.__setitem__("n", ingest_called["n"] + 1))
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder",
                        lambda out, email: "https://checkout.stripe/x")

    # neuro-magnesium was never purchased by this client — not entitled to reorder it.
    r = c.post(f"/api/portal/{tok}/checkout",
               json={"items": [{"slug": "neuro-magnesium", "qty": 1}]})
    assert r.status_code == 400
    assert invoice_called["n"] == 0
    assert ingest_called["n"] == 0


# ── security: a client-posted price is never trusted ─────────────────────────

def test_item_reorder_ignores_client_posted_price(client, monkeypatch):
    c, appmod = client
    email = "priceguard@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=5, unit_cents=6997)
    captured = _mock_checkout(appmod, monkeypatch)

    r = c.post(f"/api/portal/{tok}/checkout",
               json={"items": [{"slug": "nous-energy", "qty": 1, "price_cents": 1}]})
    assert r.status_code == 200
    # 1 cent posted price must be ignored — server prices it from the catalog.
    assert captured["lines"][0]["amount"] != 0.01
    assert captured["lines"][0]["amount"] == 69.97
