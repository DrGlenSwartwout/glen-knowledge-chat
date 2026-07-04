"""Task 8: patient-portal recommendation landing + accept-to-cart.

The patient portal surfaces the patient's ACTIVE practitioner recommendation
(practitioner_recommendations.active_for_patient) as a card, with a one-tap
accept (adds the items to a MEMBER-PRICED cart + marks 'accepted') and a dismiss
(marks 'dismissed').

CRITICAL (carry-forward from the per-patient view review): the suggested-step
items carry a price that is $0 on prod (their source table is local-only). The
accept route must therefore price the items ONLY through the existing
member-priced portal path (_portal_priced_lines → pricing.compute), never the
recommendation's own price_cents. These tests assert the accepted line price
equals the patient's real member price (NOT $0).

Fixture/helper patterns mirror tests/test_portal_item_reorder.py (portal seed,
active membership, repertoire SKU, _price_cart expected-price cross-check).
"""
import sqlite3
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "REPERTOIRE_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_portal(appmod, email="pat@example.com", name="Pat", content=None):
    from dashboard import client_portal as cp
    content = content or {"greeting": "hi", "video": {}, "layers": []}
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, content)
    cx.close()
    return token


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


def _seed_recommendation(appmod, email, items, *, note="keep going", pid="prac-1"):
    from dashboard import practitioner_recommendations as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pr.init_table(cx)
        rec_id = pr.create(cx, practitioner_id=pid, patient_email=email,
                           items=items, note=note)
    return rec_id


def _rec_status(appmod, email):
    from dashboard import practitioner_recommendations as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rec = pr.active_for_patient(cx, email)
        return rec["status"] if rec else None


# ── data surface: the portal payload carries the active recommendation ────────

def test_portal_payload_surfaces_active_recommendation(client):
    c, appmod = client
    email = "surface@example.com"
    tok = _seed_portal(appmod, email)
    _seed_recommendation(appmod, email,
                         [{"slug": "neuro-magnesium", "qty": 1, "price_cents": 0}],
                         note="one more month")

    r = c.get(f"/api/portal/{tok}")
    assert r.status_code == 200
    body = r.get_json()
    reco = body.get("recommendation")
    assert reco is not None
    assert reco["note"] == "one more month"
    assert len(reco["items"]) == 1
    item = reco["items"][0]
    assert item["slug"] == "neuro-magnesium"
    # the remedy NAME is surfaced (not the raw slug), and NO $0 price is shown
    assert item.get("name")
    assert "price_cents" not in item
    assert "price" not in item


def test_portal_payload_no_recommendation_when_none(client):
    c, appmod = client
    tok = _seed_portal(appmod, "norec@example.com")
    r = c.get(f"/api/portal/{tok}")
    assert r.status_code == 200
    body = r.get_json()
    assert not body.get("recommendation")


# ── accept: member-priced cart + status flip ─────────────────────────────────

def test_accept_prices_at_member_price_not_zero_and_marks_accepted(client, monkeypatch):
    c, appmod = client
    email = "accept-member@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["neuro-magnesium"])
    # The recommendation carries the $0-on-prod suggested-step price. It must NEVER
    # be used — the item is re-priced through the member-priced portal path.
    _seed_recommendation(appmod, email,
                         [{"slug": "neuro-magnesium", "qty": 1, "price_cents": 0}])

    # The authoritative member price via the real pricing engine.
    expected_unit = appmod._price_cart(
        [{"slug": "neuro-magnesium", "qty": 1}],
        ship={"country": "US", "state": "TX"}, email=email,
    )["priced"]["lines"][0]["line_total_cents"]
    assert 0 < expected_unit < 6997  # a real, discounted member price

    r = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body.get("accepted") is True
    assert body["lines"], "accept must return the member-priced cart lines"
    line = body["lines"][0]
    # the line price is the real member price — NOT the recommendation's $0
    assert round(line["amount"] * 100) == expected_unit
    assert line["amount"] > 0

    assert _rec_status(appmod, email) == "accepted"


def test_accept_builds_live_member_priced_checkout_when_stripe_active(client, monkeypatch):
    """Full loop: with card checkout active, accept routes the recommended items
    through the SAME member-priced portal checkout (QBO invoice → Stripe URL), and
    the invoiced line carries the member price — never the recommendation's $0."""
    c, appmod = client
    email = "accept-live@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["neuro-magnesium"])
    _seed_recommendation(appmod, email,
                         [{"slug": "neuro-magnesium", "qty": 1, "price_cents": 0}])

    captured = {}
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    def _fake_invoice(cust, lines, **kw):
        captured["lines"] = lines
        total = sum(l["amount"] * l["qty"] for l in lines)
        return {"Id": "INV1", "DocNumber": "1001", "TotalAmt": total}
    monkeypatch.setattr(qbo_billing, "create_invoice", _fake_invoice)
    monkeypatch.setattr(appmod, "_ingest_order", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder",
                        lambda out, email: "https://checkout.stripe/reco")

    expected_unit = appmod._price_cart(
        [{"slug": "neuro-magnesium", "qty": 1}],
        ship={"country": "US", "state": "TX"}, email=email,
    )["priced"]["lines"][0]["line_total_cents"]

    r = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r.status_code == 200
    body = r.get_json()
    assert body["stripe_url"] == "https://checkout.stripe/reco"
    assert round(captured["lines"][0]["amount"] * 100) == expected_unit
    assert captured["lines"][0]["amount"] > 0
    assert _rec_status(appmod, email) == "accepted"


def test_accept_no_active_recommendation_is_rejected(client):
    c, appmod = client
    tok = _seed_portal(appmod, "empty@example.com")
    r = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r.status_code == 400
    assert not r.get_json().get("ok")


# ── dismiss ──────────────────────────────────────────────────────────────────

def test_dismiss_marks_dismissed_and_card_disappears(client):
    c, appmod = client
    email = "dismiss@example.com"
    tok = _seed_portal(appmod, email)
    _seed_recommendation(appmod, email, [{"slug": "neuro-magnesium", "qty": 1}])

    # card is visible first
    body = c.get(f"/api/portal/{tok}").get_json()
    assert body.get("recommendation")

    r = c.post(f"/api/portal/{tok}/recommendation/dismiss")
    assert r.status_code == 200
    assert r.get_json().get("dismissed") is True

    # active_for_patient now returns None → status dismissed
    assert _rec_status(appmod, email) is None  # active_for_patient excludes dismissed

    # and the portal card is gone
    body2 = c.get(f"/api/portal/{tok}").get_json()
    assert not body2.get("recommendation")


def test_accepted_recommendation_no_longer_shows_as_actionable_card(client):
    c, appmod = client
    email = "accepted-hide@example.com"
    tok = _seed_portal(appmod, email)
    _seed_recommendation(appmod, email, [{"slug": "neuro-magnesium", "qty": 1}])
    # mark it accepted directly (simulating a prior accept in a stripe-active env)
    from dashboard import practitioner_recommendations as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rec = pr.active_for_patient(cx, email)
        pr.set_status(cx, rec["id"], "accepted")
    body = c.get(f"/api/portal/{tok}").get_json()
    # only a 'sent' recommendation is an actionable card
    assert not body.get("recommendation")


def test_accept_replay_is_rejected_no_duplicate_invoice(client, monkeypatch):
    """Regression: a direct POST replay against an already-accepted recommendation
    must be rejected (no duplicate QBO invoice/Stripe session). Only a 'sent'
    recommendation can be accepted."""
    c, appmod = client
    email = "replay-test@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["neuro-magnesium"])
    _seed_recommendation(appmod, email,
                         [{"slug": "neuro-magnesium", "qty": 1, "price_cents": 0}])

    # Mock invoice creation to track duplicate calls
    invoice_count = {"calls": 0}
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    def _count_invoice(cust, lines, **kw):
        invoice_count["calls"] += 1
        total = sum(l["amount"] * l["qty"] for l in lines)
        return {"Id": f"INV{invoice_count['calls']}", "DocNumber": str(1000 + invoice_count['calls']), "TotalAmt": total}
    monkeypatch.setattr(qbo_billing, "create_invoice", _count_invoice)
    monkeypatch.setattr(appmod, "_ingest_order", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder",
                        lambda out, email: "https://checkout.stripe/reco")

    # First accept should succeed
    r1 = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r1.status_code == 200
    assert r1.get_json().get("accepted") is True
    assert invoice_count["calls"] == 1
    assert _rec_status(appmod, email) == "accepted"

    # Second accept (replay) should be rejected
    r2 = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r2.status_code == 400
    assert not r2.get_json().get("ok")
    assert "No active recommendation to accept" in r2.get_json().get("error", "")
    # NO second invoice should be created
    assert invoice_count["calls"] == 1
