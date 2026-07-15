import datetime
from dashboard import membership_products as mp

def test_three_tiers_with_exact_amounts():
    assert set(mp.TIERS) == {"month", "year_monthly", "year_prepay"}
    assert mp.get_tier("month")["price_cents"] == 9900
    assert mp.get_tier("year_monthly")["price_cents"] == 9900
    assert mp.get_tier("year_prepay")["price_cents"] == 99000

def test_billing_types():
    assert mp.get_tier("month")["billing"] == "one_time"
    assert mp.get_tier("year_prepay")["billing"] == "one_time"
    ym = mp.get_tier("year_monthly")
    assert ym["billing"] == "recurring_capped"
    assert ym["term_charges"] == 12
    assert ym["cadence_months"] == 1

def test_sources_are_membership_namespaced():
    for t in mp.all_tiers():
        assert t["source"].startswith("membership_")

def test_grant_days_covers_the_term_plus_grace():
    today = datetime.date(2026, 7, 15)
    # 1 month tier: ~31 days + 4 grace
    assert 33 <= mp.grant_days("month", today) <= 36
    # 1 year tiers: ~365 days + 4 grace
    assert 366 <= mp.grant_days("year_prepay", today) <= 370
    assert 366 <= mp.grant_days("year_monthly", today) <= 370

def test_unknown_tier_returns_none():
    assert mp.get_tier("nope") is None
