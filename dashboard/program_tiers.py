"""Sellable membership tiers for the client-portal program page.

Pure + parameter-based; never imports app, so it unit-tests in isolation.
Prices are referenced from the canonical constants (family_plan.PLAN,
portal_offers.MEMBERSHIP_PRICE_CENTS) and never hardcoded here.
"""
from dashboard import family_plan as _fp
from dashboard import portal_offers as _po


def _state(owned, enabled):
    if owned:
        return "owned"
    return "available" if enabled else "coming_soon"


def program_blocks(*, paid_owned, family_owned, paid_live, family_enabled):
    """The three sellable tiers, in ladder order, with per-viewer state."""
    free = {
        "key": "free",
        "name": "Free membership",
        "benefits": [
            "Your private portal with your Biofield Analysis and matched remedies",
            "Order your remedies whenever you want",
            "Referral tracking so you can share and be credited",
        ],
        "price_cents": 0,
        "value_cents": None,
        "period": "",
        "cta_label": None,
        "checkout_path": None,
        "cta_kind": "none",
        "state": "owned",
    }
    paid = {
        "key": "paid",
        "name": "Guided membership",
        "benefits": [
            "Live group coaching with Dr. Glen",
            "Your protocol re-matched as you progress",
            "Your AI ally and Terrain Restore support",
            "Billed $99 per month for 12 months; your first month is charged today",
        ],
        "price_cents": _po.MEMBERSHIP_PRICE_CENTS,
        "value_cents": None,
        "period": "/mo",
        "cta_label": "Join",
        "checkout_path": "/portal/offer/continuous-care/checkout",
        "cta_kind": "checkout_post",
        "state": _state(paid_owned, paid_live),
    }
    family = {
        "key": "family",
        "name": _fp.PLAN["label"],
        "benefits": [
            "Everything in guided membership for your whole household",
            "Cover the people you care for under one plan",
            "One simple monthly price for the family",
        ],
        "price_cents": _fp.PLAN["amount_cents"],
        "value_cents": _fp.PLAN["value_cents"],
        "period": "/mo",
        "cta_label": "Reply to arrange",
        "checkout_path": None,
        "cta_kind": "arrange",
        "state": _state(family_owned, family_enabled),
    }
    return [free, paid, family]


def current_tier_key(tiers):
    """Highest owned tier: family > paid > free."""
    owned = {t["key"] for t in tiers if t.get("state") == "owned"}
    for key in ("family", "paid", "free"):
        if key in owned:
            return key
    return "free"


GROW_PATHS = [
    {"key": "practitioner", "name": "Become a Practitioner",
     "blurb": "Offer Biofield Analysis to your own clients at wholesale.",
     "url": "/practitioner/register"},
    {"key": "coach", "name": "Coach Training",
     "blurb": "Train as a coach and grow into the certification path.",
     "url": "/practitioner/register"},
    {"key": "cert", "name": "Certification",
     "blurb": "Earn your certification with Dr. Glen.",
     "url": "/cert"},
]
