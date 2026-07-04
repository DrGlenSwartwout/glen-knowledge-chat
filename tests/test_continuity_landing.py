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

def test_accept_marks_accepted_without_pricing_even_with_active_membership(client, monkeypatch):
    """Retargeted (was test_accept_prices_at_member_price_not_zero_and_marks_accepted):
    that test asserted accept minted a member-priced cart. Under the new contract
    (fix/rec-accept-to-reorder), accept NEVER prices or mints anything — it only
    flips status — regardless of the patient having an active membership + a
    real member price available. The member-price-at-checkout behavior is
    re-asserted against the reorder checkout itself, not this route."""
    c, appmod = client
    email = "accept-member@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["neuro-magnesium"])
    # The recommendation carries the $0-on-prod suggested-step price. It is
    # irrelevant now — accept doesn't touch pricing at all.
    _seed_recommendation(appmod, email,
                         [{"slug": "neuro-magnesium", "qty": 1, "price_cents": 0}])

    r = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body.get("accepted") is True
    assert "lines" not in body
    assert "stripe_url" not in body or body.get("stripe_url") is None

    assert _rec_status(appmod, email) == "accepted"


def test_accept_does_not_call_qbo_or_stripe_even_when_stripe_active(client, monkeypatch):
    """Retargeted (was test_accept_builds_live_member_priced_checkout_when_stripe_active):
    that test asserted accept built a live QBO invoice → Stripe checkout. Under the
    new contract, even with a live-card-checkout environment (_STRIPE_ACTIVE=True),
    accept must not call QBO or Stripe at all — the invoice is minted later, by the
    normal reorder checkout, not by accept."""
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

    r = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r.status_code == 200
    body = r.get_json()
    assert "stripe_url" not in body or body.get("stripe_url") is None
    assert "lines" not in captured, "accept must never call create_invoice"
    assert _rec_status(appmod, email) == "accepted"


def test_accept_mints_no_invoice_and_marks_accepted(client, monkeypatch):
    """New contract (fast-follow off #572): accept mints NOTHING — no QBO invoice,
    no Stripe session, no order row. It only flips the recommendation to
    'accepted'. Purchase happens later through the normal reorder checkout."""
    c, appmod = client
    email = "accept-noinvoice@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["neuro-magnesium"])
    _seed_recommendation(appmod, email,
                         [{"slug": "neuro-magnesium", "qty": 1, "price_cents": 0}])

    # If accept somehow still called through to QBO/Stripe, fail loudly.
    from dashboard import qbo_billing
    def _boom(*a, **k):
        raise AssertionError("accept must not mint an invoice")
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", _boom)
    monkeypatch.setattr(qbo_billing, "create_invoice", _boom)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)

    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        appmod._bos_orders.init_orders_table(cx)
        before = len(appmod._bos_orders.list_orders_by_email(cx, email, limit=200))

    r = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body.get("accepted") is True
    assert "stripe_url" not in body or body.get("stripe_url") is None
    assert "lines" not in body

    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        after = len(appmod._bos_orders.list_orders_by_email(cx, email, limit=200))
    assert after == before  # NO order/invoice created

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


# ── Task 2 (fast-follow off #572): accepted-rec items surface in reorder ────

def test_accepted_recommendation_items_appear_in_reorder_deduped(client):
    """An ACCEPTED recommendation's items merge into the portal payload's
    reorder_items, deduped by slug — a recommended slug already present in the
    curated reorder set is not doubled."""
    c, appmod = client
    email = "reorder-merge@example.com"
    tok = _seed_portal(appmod, email, content={
        "greeting": "hi", "video": {}, "layers": [],
        "reorder_items": [{"slug": "neuro-magnesium", "qty": 1}],
    })
    rec_id = _seed_recommendation(
        appmod, email,
        [{"slug": "neuro-magnesium", "qty": 1, "price_cents": 0},
         {"slug": "nous-energy", "qty": 1, "price_cents": 0}])
    from dashboard import practitioner_recommendations as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pr.set_status(cx, rec_id, "accepted")

    body = c.get(f"/api/portal/{tok}").get_json()
    slugs = [it["slug"] for it in body["reorder_items"]]
    assert "nous-energy" in slugs             # the new recommended item is present
    assert slugs.count("neuro-magnesium") == 1  # not doubled (dedup by slug)


def test_accepted_recommendation_not_merged_while_still_sent(client):
    """A 'sent' (not-yet-accepted) recommendation's items must NOT leak into the
    reorder list — only 'accepted' recs merge in."""
    c, appmod = client
    email = "reorder-notyet@example.com"
    tok = _seed_portal(appmod, email, content={
        "greeting": "hi", "video": {}, "layers": [], "reorder_items": [],
    })
    _seed_recommendation(appmod, email, [{"slug": "nous-energy", "qty": 1, "price_cents": 0}])

    body = c.get(f"/api/portal/{tok}").get_json()
    slugs = [it["slug"] for it in body["reorder_items"]]
    assert "nous-energy" not in slugs


def test_reorder_checkout_of_accepted_items_prices_member_and_one_invoice(client, monkeypatch):
    """Once an accepted recommendation's item is in reorder_items, checking it
    out through the NORMAL per-item reorder checkout (unchanged from Task 6b)
    mints exactly ONE invoice at the real member price — never the $0
    suggested-step price. Reuses the member-price assertion pattern from the
    old accept test (test_portal_item_reorder.py), retargeted to /checkout."""
    c, appmod = client
    email = "reorder-checkout@example.com"
    tok = _seed_portal(appmod, email, content={
        "greeting": "hi", "video": {}, "layers": [], "reorder_items": [],
    })
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["neuro-magnesium"])
    # Checkout entitlement is the client's own portal-channel purchase history
    # (unchanged Task 6b gate) — seed a prior portal-reorder purchase so the
    # accepted-rec slug is entitled to be reordered.
    import json as _json
    from datetime import datetime, timezone, timedelta
    from dashboard.orders import init_orders_table
    created = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        init_orders_table(cx)
        cx.execute(
            "INSERT INTO orders (created_at, source, external_ref, channel, email, "
            "items_json, total_cents, status) VALUES (?,?,?,?,?,?,?,?)",
            (created, "portal-reorder", f"o-seed-{email}", "retail", email,
             _json.dumps([{"slug": "neuro-magnesium", "qty": 1,
                           "name": "neuro-magnesium", "unit_cents": 6997}]),
             6997, "done"))
        cx.commit()

    rec_id = _seed_recommendation(
        appmod, email, [{"slug": "neuro-magnesium", "qty": 1, "price_cents": 0}])
    from dashboard import practitioner_recommendations as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pr.set_status(cx, rec_id, "accepted")

    body = c.get(f"/api/portal/{tok}").get_json()
    assert any(it["slug"] == "neuro-magnesium" for it in body["reorder_items"])

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
    assert 0 < expected_unit < 6997  # real member price, never $0 nor full price

    r = c.post(f"/api/portal/{tok}/checkout",
               json={"items": [{"slug": "neuro-magnesium", "qty": 1}]})
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://checkout.stripe/reco"
    assert len(captured["lines"]) == 1  # exactly ONE invoice line, no duplication
    assert round(captured["lines"][0]["amount"] * 100) == expected_unit


def test_accept_replay_is_rejected_no_duplicate_invoice(client, monkeypatch):
    """Regression: a direct POST replay against an already-accepted recommendation
    must be rejected. Only a 'sent' recommendation can be accepted — now trivially
    true for invoices too, since accept never mints one in the first place (retargeted
    from the pre-fix/rec-accept-to-reorder contract, which counted QBO invoice calls
    across accept + replay)."""
    c, appmod = client
    email = "replay-test@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["neuro-magnesium"])
    _seed_recommendation(appmod, email,
                         [{"slug": "neuro-magnesium", "qty": 1, "price_cents": 0}])

    # Mock invoice creation to prove accept/replay never call it.
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

    # First accept should succeed — and mint NO invoice.
    r1 = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r1.status_code == 200
    assert r1.get_json().get("accepted") is True
    assert invoice_count["calls"] == 0
    assert _rec_status(appmod, email) == "accepted"

    # Second accept (replay) should be rejected
    r2 = c.post(f"/api/portal/{tok}/recommendation/accept")
    assert r2.status_code == 400
    assert not r2.get_json().get("ok")
    assert "No active recommendation to accept" in r2.get_json().get("error", "")
    # Still no invoice from either call
    assert invoice_count["calls"] == 0
