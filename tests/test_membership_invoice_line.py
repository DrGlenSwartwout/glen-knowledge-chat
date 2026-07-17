import os
from dashboard import membership_products as mp


def test_line_slug_and_for():
    assert mp.line_slug("month") == "membership:month"
    line = mp.line_for("month")
    assert line["slug"] == "membership:month"
    assert line["kind"] == "membership"
    assert line["tier"] == "month"
    assert line["qty"] == 1
    assert line["unit_cents"] == mp.get_tier("month")["price_cents"] == 9900
    assert line["line_cents"] == 9900
    assert line["name"] == mp.get_tier("month")["label"]


def test_line_for_unknown_tier_is_none():
    assert mp.line_for("nope") is None


def test_tier_of_line_detects_by_kind_and_slug():
    assert mp.tier_of_line({"kind": "membership", "tier": "month"}) == "month"
    assert mp.tier_of_line({"slug": "membership:year_prepay"}) == "year_prepay"
    assert mp.tier_of_line({"slug": "paracleanse", "qty": 1}) is None
    assert mp.tier_of_line({}) is None


def test_cart_has_membership_tier():
    cart = [{"slug": "paracleanse", "qty": 1}, {"slug": "membership:month", "kind": "membership"}]
    assert mp.cart_has_membership_tier(cart) == "month"
    assert mp.cart_has_membership_tier([{"slug": "paracleanse"}]) is None


def test_invoice_offer_tiers_default_and_env(monkeypatch):
    monkeypatch.delenv("MEMBERSHIP_INVOICE_TIERS", raising=False)
    assert mp.invoice_offer_tiers() == ["month"]
    monkeypatch.setenv("MEMBERSHIP_INVOICE_TIERS", "month, year_prepay , bogus")
    # keeps order, drops unknown tiers
    assert mp.invoice_offer_tiers() == ["month", "year_prepay"]
