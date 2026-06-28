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


def chain_summary(cx, email, *, max_depth=2):
    """Reconstruct the healing chain seeded by `email` from referral_redemptions
    (owner_email gifted -> referee_email redeemed), walked to max_depth levels.
    Returns per-level counts and total distinct people reached. Read-only.
    Excludes the seed and de-dupes people already counted at a shallower level."""
    _referrals.init_tables(cx)
    seed = _norm(email)
    levels = []
    reached = set()
    frontier = {seed}
    for _ in range(max_depth):
        if not frontier:
            break
        placeholders = ",".join("?" for _ in frontier)
        rows = cx.execute(
            "SELECT DISTINCT lower(referee_email) FROM referral_redemptions "
            f"WHERE lower(owner_email) IN ({placeholders})",
            tuple(frontier)).fetchall()
        nxt = {r[0] for r in rows if r[0] and r[0] != seed and r[0] not in reached}
        if not nxt:
            break
        levels.append(len(nxt))
        reached |= nxt
        frontier = nxt
    return {
        "reached": len(reached),
        "l1": levels[0] if len(levels) > 0 else 0,
        "l2": levels[1] if len(levels) > 1 else 0,
        "levels": levels,
    }
