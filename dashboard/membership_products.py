"""Membership product catalog: the three buyable tiers, all granting the same
entitlement (live group coaching + member pricing) and differing only in billing.
Reuses existing fulfillment primitives — one-time tiers are grant-only (like the
prepay ladder, so the charge cron never bills them and they cannot auto-renew);
the recurring-capped tier uses a subscriptions row with term_charges_total that
self-cancels at the cap (dashboard.subscriptions + app charge cron)."""
import calendar
import datetime
import os

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

def _tier_sources():
    return tuple(t["source"] for t in TIERS.values())

def owns_group(cx, email):
    """True iff the email holds an active membership-tier grant. Namespaced to the
    tier sources so prepay/continuous-care/founding grants are unaffected."""
    if not email:
        return False
    now = datetime.datetime.utcnow().isoformat()
    srcs = _tier_sources()
    ph = ",".join("?" * len(srcs))
    row = cx.execute(
        f"SELECT 1 FROM memberships WHERE lower(email)=lower(?) "
        f"AND expires_at > ? AND source IN ({ph}) LIMIT 1",
        (email, now, *srcs)).fetchone()
    return row is not None


_MEMBERSHIP_LINE_PREFIX = "membership:"


def line_slug(tier_key):
    return f"{_MEMBERSHIP_LINE_PREFIX}{tier_key}"


def line_for(tier_key):
    """The stored order-line dict for a membership tier, or None if the tier is unknown.
    Carries kind='membership' + tier so pricing/rendering can recognize it without a
    product-catalog lookup (the slug is intentionally NOT a catalog product)."""
    t = TIERS.get(tier_key)
    if not t:
        return None
    return {"slug": line_slug(tier_key), "name": t["label"], "qty": 1,
            "unit_cents": t["price_cents"], "line_cents": t["price_cents"],
            "kind": "membership", "tier": tier_key}


def tier_of_line(line):
    """Tier key if `line` is a membership line (by kind marker or slug prefix), else None."""
    if not isinstance(line, dict):
        return None
    if line.get("kind") == "membership":
        tk = line.get("tier") or (line.get("slug") or "")[len(_MEMBERSHIP_LINE_PREFIX):]
        return tk if tk in TIERS else None
    slug = (line.get("slug") or "")
    if slug.startswith(_MEMBERSHIP_LINE_PREFIX):
        tk = slug[len(_MEMBERSHIP_LINE_PREFIX):]
        return tk if tk in TIERS else None
    return None


def cart_has_membership_tier(lines):
    """First membership tier key present in `lines`, else None."""
    for ln in (lines or []):
        tk = tier_of_line(ln)
        if tk:
            return tk
    return None


def invoice_offer_tiers():
    """Tier keys offered by the on-invoice membership control. Configurable via the
    MEMBERSHIP_INVOICE_TIERS env var (comma-separated); unknown tiers are dropped;
    default ['month']."""
    raw = (os.environ.get("MEMBERSHIP_INVOICE_TIERS") or "").strip()
    if not raw:
        return ["month"]
    out = [k.strip() for k in raw.split(",") if k.strip() in TIERS]
    return out or ["month"]
