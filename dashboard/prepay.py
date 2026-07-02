"""Prepay-ladder pricing — the single source of truth for membership prices.

A prospective member can prepay 1/3/6/12 months up front at a progressive
per-month discount ("commit now, save"). All money is in CENTS. This is a PURE
module (no Flask, no Stripe, no DB) — term math delegates to
subscriptions.add_months so tests run in isolation, mirroring subscriptions.py.

The 1-month tier IS the $99/mo anchor: MONTHLY_ANCHOR_CENTS is the reroute target
for the membership-price hardcodes that used to live as bare 9900 literals in
group_bundle.py / portal_offers.py (and 99.00 dollars in app.py's QBO tiers).
"""

from dashboard import subscriptions as _subs

# The $99/mo regular rate. Single source of truth for the membership fee.
MONTHLY_ANCHOR_CENTS = 9900

# When a prepaid term ends, renewal is offered at a shallower (loyalty) rate than
# the first-term deal so we don't train discount-waiting — default = the 6-month
# ("Most Popular") per-month rate. Tunable; pending owner confirmation.
RENEWAL_MONTHLY_CENTS = 7500

# Ladder rungs, in ascending term order. price_cents is the one-time prepay total.
TIERS = [
    {"key": "1mo",  "months": 1,  "price_cents": 9900,  "badge": "",                "label": "1 month"},
    {"key": "3mo",  "months": 3,  "price_cents": 26700, "badge": "",                "label": "3 months"},
    {"key": "6mo",  "months": 6,  "price_cents": 45000, "badge": "Most Popular",    "label": "6 months"},
    {"key": "12mo", "months": 12, "price_cents": 59900, "badge": "Founding Member", "label": "12 months"},
]

_BY_KEY = {t["key"]: t for t in TIERS}


def get_tier(key):
    """Return the tier descriptor dict for *key*, or None if unknown."""
    return _BY_KEY.get(key) if key else None


def per_month_cents(key) -> int:
    """Effective per-month cost of a tier (rounded to the nearest cent)."""
    t = get_tier(key)
    if not t:
        return 0
    return round(t["price_cents"] / t["months"])


def savings_pct_vs_anchor(key) -> int:
    """Percent saved vs paying the monthly anchor for the same number of months
    (rounded). 0 for the 1-month anchor tier."""
    t = get_tier(key)
    if not t:
        return 0
    full = MONTHLY_ANCHOR_CENTS * t["months"]
    if full <= 0:
        return 0
    return round((full - t["price_cents"]) / full * 100)


def tiers_public() -> list:
    """The ladder as the picker UI needs it: descriptor + computed per-month +
    savings %. No hidden fields, safe to serialize to the page."""
    return [
        {
            "key": t["key"],
            "months": t["months"],
            "price_cents": t["price_cents"],
            "per_month_cents": per_month_cents(t["key"]),
            "savings_pct": savings_pct_vs_anchor(t["key"]),
            "badge": t["badge"],
            "label": t["label"],
        }
        for t in TIERS
    ]


def term_end_date(start_yyyy_mm_dd: str, months: int) -> str:
    """Calendar-accurate term-end date (delegates to subscriptions.add_months, so
    month-end clamping matches the rest of the subscription code)."""
    return _subs.add_months(start_yyyy_mm_dd, int(months))


def term_days(start_yyyy_mm_dd: str, months: int) -> int:
    """Number of days in an N-month term starting on *start* — the grant length
    to pass to _grant_membership. Calendar-accurate (leap years, month lengths)."""
    from datetime import datetime
    start = datetime.strptime(start_yyyy_mm_dd, "%Y-%m-%d")
    end = datetime.strptime(term_end_date(start_yyyy_mm_dd, months), "%Y-%m-%d")
    return (end - start).days


def renewal_price_cents(key):
    """One-time price to renew the SAME term length at the loyalty renewal rate,
    or None for an unknown tier. Used only to populate the renewal-prompt copy —
    no charge is ever made automatically."""
    t = get_tier(key)
    if not t:
        return None
    return RENEWAL_MONTHLY_CENTS * t["months"]
