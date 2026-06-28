"""dashboard/pay_it_forward.py — Pay It Forward: Gift surface helpers.

Pure module: accepts a sqlite connection; no Flask imports, no import-time side
effects. Builds on the existing points ledger (dashboard.points) and gift
attribution (dashboard.referrals / referral_redemptions). Money is integer cents.
"""

from dashboard import points as _points
from dashboard import referrals as _referrals

# Gift-power granted per confirmed healing milestone (redemption-value cents).
MILESTONE_REWARD_CENTS = 500


def _norm(email):
    return (email or "").strip().lower()


def award_milestone(cx, email, *, milestone_key, value_cents=MILESTONE_REWARD_CENTS):
    """Credit gift-power points for a confirmed healing milestone.
    Idempotent per (email, milestone_key) via points.credit's order_ref+reason guard."""
    e = _norm(email)
    key = (milestone_key or "").strip()
    if not e or not key:
        return
    _points.init_points_table(cx)
    _points.credit(cx, e, value_cents=int(value_cents),
                   reason="healing_milestone",
                   order_ref=f"milestone:{e}:{key}")
