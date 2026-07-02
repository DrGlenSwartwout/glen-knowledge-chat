"""Continuous Care pricing — the single source of truth for Continuous Care terms.

Continuous Care is term-based and never auto-renews: a client buys a month, or
prepays 3/6/12 months up front at a modest progressive discount graduated to
"2 months free" at the annual term. All money is in CENTS. This is a PURE module
(no Flask, no Stripe, no DB) — term math delegates to subscriptions.add_months so
tests run in isolation, mirroring subscriptions.py.

The monthly term IS the $99/mo care rate: MONTHLY_ANCHOR_CENTS is the single
source for that rate (also the reroute target for the $99/mo literals in
group_bundle.py / portal_offers.py and app.py's QBO tier).
"""

from dashboard import subscriptions as _subs

# The $99/mo Continuous Care rate. Single source of truth.
MONTHLY_ANCHOR_CENTS = 9900

# When a prepaid term ends, renewal is offered at a shallower (loyalty) rate than
# the annual "2 months free" deal so we don't train discount-waiting — default =
# the 6-month per-month rate ($91/mo). Tunable.
RENEWAL_MONTHLY_CENTS = 9100

# Continuous Care terms, in ascending order. price_cents is the one-time prepay
# total. Discounts are graduated (0% / ~2% / ~8% / ~17%) capping at "2 months free"
# at the annual term — a modest loyalty reward, not a fire-sale.
TIERS = [
    {"key": "1mo",  "months": 1,  "price_cents": 9900,  "badge": "",             "label": "Monthly"},
    {"key": "3mo",  "months": 3,  "price_cents": 29000, "badge": "",             "label": "3 months"},
    {"key": "6mo",  "months": 6,  "price_cents": 54600, "badge": "Most Popular", "label": "6 months"},
    {"key": "12mo", "months": 12, "price_cents": 99000, "badge": "2 months free", "label": "12 months"},
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
