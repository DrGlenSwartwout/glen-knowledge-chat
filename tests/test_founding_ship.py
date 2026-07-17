"""Task 6: charge-on-ship admin action — test suite.

Tests:
1. _ship_founding_reservation charges, activates, and ingests order with source="reorder"
2. Route POST /api/founding/ship skips already-active reservations (idempotency)
3. Route rejects missing/wrong X-Console-Key
"""
import sqlite3
import app as appmod
from dashboard import subscriptions as subs


# ---------------------------------------------------------------------------
# Helper: build an in-memory DB with the founding schema
# ---------------------------------------------------------------------------

def _build_cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_founding_columns(cx)
    return cx


# ---------------------------------------------------------------------------
# Test 1: unit-level — _ship_founding_reservation
# ---------------------------------------------------------------------------

def test_ship_charges_and_activates(monkeypatch):
    """Paid-only (Pattern II): no QBO invoice is created; the charge amount is
    subtotal+shipping (never GET), and the order is keyed on the Stripe charge id."""
    cx = _build_cx()
    sid = subs.create_founding_reservation(
        cx, email="f@x.com", stripe_customer_id="cus_1",
        stripe_payment_method_id="pm_1",
        items=[{"slug": "neuro-magnesium", "qty": 1}],
        ship_address={"state": "HI"},
        founding_slug="neuro-magnesium",
    )
    sub = subs.get(cx, sid)

    monkeypatch.setattr(appmod, "_price_cart", lambda items, *, ship, subscriber_tier_pct=None, **k: {
        "qbo_lines": [{"name": "Neuro Magnesium", "amount": 80.0, "qty": 1}],
        "shipping_cents": 600, "discount_cents": 0, "points_redeemed_cents": 0,
        "priced": {"total_cents": 8600, "get_cents": 0, "subtotal_cents": 8600},
    })
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "pi_1"})

    def _boom_invoice(*a, **k):
        raise AssertionError("paid-only: _ship_founding_reservation must not call create_invoice")

    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", _boom_invoice)
    orders = []
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: orders.append(kw))

    res = appmod._ship_founding_reservation(cx, sub)

    assert res["charged"] is True
    assert res["amount_cents"] == 9200  # subtotal(8600) + shipping(600), GET excluded
    row = subs.get(cx, sid)
    assert row["founding_state"] == "active" and row["order_count"] == 1
    assert row["next_charge_date"] != "2999-01-01"
    assert orders and orders[0]["source"] == "reorder"   # qualifies for delivery->coaching
    assert orders[0]["external_ref"] == "pi_1"  # keyed on the Stripe charge id, not an invoice id


# ---------------------------------------------------------------------------
# Test 2: route-level — idempotency (active reservation not re-charged)
# ---------------------------------------------------------------------------

def test_ship_route_skips_already_active(monkeypatch, tmp_path):
    """An already-active founding reservation must not be re-charged when the
    ship action is triggered again. The route uses list_founding_pending which
    only returns 'pending' rows, so the route naturally skips active ones.
    This test genuinely POSTs to the route and asserts charge call counts."""
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")

    # Seed the schema and both rows into the real file DB the route will read
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_founding_columns(cx)

    sid_active = subs.create_founding_reservation(
        cx, email="active@x.com", stripe_customer_id="cus_A",
        stripe_payment_method_id="pm_A",
        items=[{"slug": "neuro-magnesium", "qty": 1}],
        ship_address={"state": "HI"},
        founding_slug="neuro-magnesium",
    )
    subs.mark_founding_active(cx, sid_active, next_charge_date="2026-07-24")

    sid_pending = subs.create_founding_reservation(
        cx, email="pending@x.com", stripe_customer_id="cus_B",
        stripe_payment_method_id="pm_B",
        items=[{"slug": "neuro-magnesium", "qty": 1}],
        ship_address={"state": "HI"},
        founding_slug="neuro-magnesium",
    )
    cx.commit()
    cx.close()

    # Track which sub IDs were charged
    charged_ids = []

    def _fake_ship(cx2, sub):
        charged_ids.append(sub["id"])
        subs.mark_founding_active(cx2, sub["id"], next_charge_date="2026-07-24")
        return {"charged": True, "sub_id": sub["id"], "amount_cents": 8600}

    monkeypatch.setattr(appmod, "_ship_founding_reservation", _fake_ship)

    # POST to the real route — it opens LOG_DB itself via sqlite3.connect(LOG_DB)
    client = appmod.app.test_client()
    r = client.post(
        "/api/founding/ship",
        headers={"X-Console-Key": "test-secret"},
        json={"slug": "neuro-magnesium"},
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True

    # Only the pending sub should have been charged — active was never in the pending list
    assert charged_ids == [sid_pending], (
        f"Expected only sid_pending={sid_pending} charged; got {charged_ids}"
    )


# ---------------------------------------------------------------------------
# Test 3: route auth — wrong/missing X-Console-Key rejected
# ---------------------------------------------------------------------------

def test_ship_route_rejects_bad_key(monkeypatch):
    """POST /api/founding/ship must return 403 when CONSOLE_SECRET is set
    and the request provides a wrong or missing key."""
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "secret-abc")
    client = appmod.app.test_client()

    # Missing key
    r = client.post("/api/founding/ship", json={"slug": "neuro-magnesium"})
    assert r.status_code == 403

    # Wrong key
    r = client.post("/api/founding/ship",
                    headers={"X-Console-Key": "wrong"},
                    json={"slug": "neuro-magnesium"})
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Test 4: route auth — correct key accepted (smoke test)
# ---------------------------------------------------------------------------

def test_ship_route_accepts_correct_key(monkeypatch):
    """POST /api/founding/ship with correct key returns 200 ok (empty results
    is fine — no pending reservations in the live DB during test)."""
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "secret-abc")

    # Stub _ship_founding_reservation so we don't need a real DB or Stripe
    monkeypatch.setattr(appmod, "_ship_founding_reservation",
                        lambda cx, sub: {"charged": True, "sub_id": sub["id"], "amount_cents": 0})

    # Patch list_founding_pending to return empty so no real DB needed
    import dashboard.subscriptions as _s
    monkeypatch.setattr(_s, "list_founding_pending", lambda cx, slug: [])

    client = appmod.app.test_client()
    r = client.post("/api/founding/ship",
                    headers={"X-Console-Key": "secret-abc"},
                    json={"slug": "neuro-magnesium"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["results"] == []
