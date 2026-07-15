"""Membership product catalog: the three buyable tiers, all granting the same
entitlement (live group coaching + member pricing) and differing only in billing.
Reuses existing fulfillment primitives — one-time tiers are grant-only (like the
prepay ladder, so the charge cron never bills them and they cannot auto-renew);
the recurring-capped tier uses a subscriptions row with term_charges_total that
self-cancels at the cap (dashboard.subscriptions + app charge cron)."""
import calendar
import datetime

GRACE_DAYS = 4

TIERS = {
    "month": {
        "key": "month", "label": "Monthly Membership", "price_cents": 9900,
        "billing": "one_time", "source": "membership_month",
        "term_charges": 1, "cadence_months": 1, "grant_months": 1,
    },
    "year_monthly": {
        "key": "year_monthly", "label": "Annual Membership (monthly)",
        "price_cents": 9900, "billing": "recurring_capped",
        "source": "membership_year_monthly",
        "term_charges": 12, "cadence_months": 1, "grant_months": 12,
    },
    "year_prepay": {
        "key": "year_prepay", "label": "Annual Membership (full pay)",
        "price_cents": 99000, "billing": "one_time",
        "source": "membership_year_prepay",
        "term_charges": 1, "cadence_months": 1, "grant_months": 12,
    },
}

_ORDER = ["month", "year_monthly", "year_prepay"]

def get_tier(key):
    return TIERS.get(key)

def all_tiers():
    return [TIERS[k] for k in _ORDER]

def _add_months(d, months):
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return datetime.date(y, m, day)

def grant_days(key, today):
    t = TIERS[key]
    end = _add_months(today, t["grant_months"])
    return (end - today).days + GRACE_DAYS
