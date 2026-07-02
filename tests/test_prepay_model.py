# tests/test_prepay_model.py
"""Pure-module tests for the prepay ladder (dashboard/prepay.py).

Mirrors the pure-module pattern of test_subscriptions_model.py: no Flask, no
Stripe, no DB — just the tier table + term/price math.
"""
import sys
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from dashboard import prepay


# ---------------------------------------------------------------------------
# Tier table integrity
# ---------------------------------------------------------------------------

def test_anchor_is_99_dollars():
    assert prepay.MONTHLY_ANCHOR_CENTS == 9900


def test_four_tiers_in_ladder_order():
    keys = [t["key"] for t in prepay.TIERS]
    assert keys == ["1mo", "3mo", "6mo", "12mo"]


def test_tier_prices_and_months():
    expected = {"1mo": (1, 9900), "3mo": (3, 26700), "6mo": (6, 45000), "12mo": (12, 59900)}
    for key, (months, price) in expected.items():
        t = prepay.get_tier(key)
        assert t["months"] == months, key
        assert t["price_cents"] == price, key


def test_one_month_tier_equals_anchor():
    assert prepay.get_tier("1mo")["price_cents"] == prepay.MONTHLY_ANCHOR_CENTS


def test_badges():
    assert prepay.get_tier("6mo")["badge"] == "Most Popular"
    assert prepay.get_tier("12mo")["badge"] == "Founding Member"
    assert prepay.get_tier("1mo")["badge"] == ""


def test_get_tier_unknown_returns_none():
    assert prepay.get_tier("9mo") is None
    assert prepay.get_tier("") is None
    assert prepay.get_tier(None) is None


# ---------------------------------------------------------------------------
# Per-month + savings math (computed, not hardcoded marketing numbers)
# ---------------------------------------------------------------------------

def test_per_month_cents():
    assert prepay.per_month_cents("1mo") == 9900
    assert prepay.per_month_cents("3mo") == 8900
    assert prepay.per_month_cents("6mo") == 7500
    # 59900 / 12 = 4991.67 -> rounds to ~$50/mo
    assert prepay.per_month_cents("12mo") == 4992


def test_savings_pct_vs_anchor():
    assert prepay.savings_pct_vs_anchor("1mo") == 0
    assert prepay.savings_pct_vs_anchor("3mo") == 10
    assert prepay.savings_pct_vs_anchor("6mo") == 24
    assert prepay.savings_pct_vs_anchor("12mo") == 50


def test_tiers_public_shape():
    pub = prepay.tiers_public()
    assert len(pub) == 4
    row = next(r for r in pub if r["key"] == "6mo")
    assert row["price_cents"] == 45000
    assert row["per_month_cents"] == 7500
    assert row["savings_pct"] == 24
    assert row["badge"] == "Most Popular"
    assert row["label"] == "6 months"
    assert row["months"] == 6


# ---------------------------------------------------------------------------
# Term math — calendar-accurate (delegates to subscriptions.add_months)
# ---------------------------------------------------------------------------

def test_term_end_date_basic():
    assert prepay.term_end_date("2026-07-01", 6) == "2027-01-01"
    assert prepay.term_end_date("2026-07-01", 12) == "2027-07-01"


def test_term_end_date_month_end_clamp():
    # Jan 31 + 1 month clamps to Feb 28 (non-leap 2027)
    assert prepay.term_end_date("2027-01-31", 1) == "2027-02-28"


def test_term_days_leap_year():
    # 2028 is a leap year: 2028-02-01 + 1mo = 2028-03-01 -> 29 days
    assert prepay.term_days("2028-02-01", 1) == 29
    # non-leap: 2027-02-01 + 1mo = 2027-03-01 -> 28 days
    assert prepay.term_days("2027-02-01", 1) == 28


def test_term_days_full_year():
    # 2026-07-01 .. 2027-07-01 = 365 days (no leap day in that span)
    assert prepay.term_days("2026-07-01", 12) == 365


# ---------------------------------------------------------------------------
# Renewal price (default: 6-month/loyalty rate, tunable pending owner confirm)
# ---------------------------------------------------------------------------

def test_renewal_monthly_rate_is_six_month_rate():
    assert prepay.RENEWAL_MONTHLY_CENTS == 7500


def test_renewal_price_cents_scales_with_term():
    assert prepay.renewal_price_cents("1mo") == 7500
    assert prepay.renewal_price_cents("6mo") == 45000
    assert prepay.renewal_price_cents("12mo") == 90000


def test_renewal_price_unknown_tier_none():
    assert prepay.renewal_price_cents("nope") is None
