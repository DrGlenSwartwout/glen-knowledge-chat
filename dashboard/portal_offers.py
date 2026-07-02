"""Upgrade-ladder offer catalog + eligibility resolver.

next_offers() returns the ladder rungs a person is eligible for — flag-on AND
not already owned — in ladder order. The portal surfaces the FIRST (single next
rung). Pure + cx-based; never imports app, so it unit-tests in isolation.
"""
import sqlite3

from dashboard import subscriptions as _subs
from dashboard import biofield_store as _bf
from dashboard import prepay as _prepay

MEMBERSHIP_PRICE_CENTS = _prepay.MONTHLY_ANCHOR_CENTS
BIOFIELD_PRICE_CENTS = 30000


def _owns_group(cx, email):
    try:
        _subs.init_subscriptions_table(cx)
        _subs.migrate_add_membership_columns(cx)
        cx.row_factory = sqlite3.Row  # active_memberships_by_email does dict(row)
        return bool(_subs.active_memberships_by_email(cx, email))
    except Exception:
        return False


def _owns_biofield(cx, email):
    try:
        _bf.init_table(cx)
        row = cx.execute(
            "SELECT paid_at FROM biofield_readiness WHERE lower(email)=lower(?)",
            (str(email or "").strip(),)).fetchone()
        return bool(row and row[0])
    except Exception:
        return False


# Ladder order. Each rung: a static descriptor + an owned(cx,email) predicate.
_LADDER = [
    {"key": "live_group", "title": "Join the Live Group", "price_cents": MEMBERSHIP_PRICE_CENTS,
     "period": "/mo", "blurb": "Live group coaching with Dr. Glen — your next step on the path.",
     "cta_label": "Join", "checkout_path": "/portal/offer/live-group/checkout",
     "owned": _owns_group},
    {"key": "biofield", "title": "Causal Biofield Analysis", "price_cents": BIOFIELD_PRICE_CENTS,
     "period": "", "blurb": "A personalized Biofield-designed program reading your causal chain.",
     "cta_label": "Book", "checkout_path": "/biofield/checkout",
     "owned": _owns_biofield},
]


def _public(rung):
    """The shape exposed to the page (drops the owned predicate)."""
    return {k: rung[k] for k in ("key", "title", "price_cents", "period", "blurb",
                                 "cta_label", "checkout_path")}


def next_offers(cx, email, roles, *, enabled_keys):
    """Eligible rungs (key in enabled_keys AND not owned), in ladder order."""
    email = (email or "").strip().lower()
    out = []
    for rung in _LADDER:
        if rung["key"] not in enabled_keys:
            continue
        if rung["owned"](cx, email):
            continue
        out.append(_public(rung))
    return out
