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
        if nxt or levels:
            levels.append(len(nxt))
        reached |= nxt
        frontier = nxt
    return {
        "reached": len(reached),
        "l1": levels[0] if len(levels) > 0 else 0,
        "l2": levels[1] if len(levels) > 1 else 0,
        "levels": levels,
    }


def healer_level(reached):
    """Derived Healer status from distinct people reached. Recognition only;
    gates nothing in this slice. 1 = Giver, 2 = Healer, 3 = Lightkeeper."""
    r = int(reached or 0)
    if r >= 10:
        return 3
    if r >= 3:
        return 2
    return 1


def _masked_name(cx, email):
    """Display name for a recipient email: 'First L.' (first name + last initial),
    or just the first name, else 'A friend'. Never returns the email or a prefix.
    Best-effort: a missing people table or row yields 'A friend'."""
    e = _norm(email)
    try:
        row = cx.execute(
            "SELECT first_name, last_name, name FROM people WHERE lower(email)=?",
            (e,)).fetchone()
    except Exception:
        return "A friend"
    if not row:
        return "A friend"
    first = (row[0] or "").strip()
    last = (row[1] or "").strip()
    full = (row[2] or "").strip()
    if not first and full:
        parts = full.split()
        first = parts[0]
        last = parts[1] if len(parts) > 1 else last
    if not first:
        return "A friend"
    if "@" in first:
        return "A friend"
    return f"{first} {last[0]}." if last else first


def _product_for_code(cx, code):
    """Best-effort gifted-product slug for a redemption's coupon code; '' if unknown
    or the coupons table is absent."""
    if not code:
        return ""
    try:
        row = cx.execute("SELECT product_slug FROM coupons WHERE code=?", (code,)).fetchone()
    except Exception:
        return ""
    return (row[0] or "") if row else ""


def chain_recipients(cx, email, *, limit=10):
    """The requester's most-recent direct (L1) gift recipients, newest first,
    capped at `limit`. Returns [{"name", "product", "redeemed_at"}] with masked
    names. Read-only; L2+ are intentionally excluded (counts only via chain_summary).
    Only rows the requester owns (owner_email == email) are returned."""
    _referrals.init_tables(cx)
    owner = _norm(email)
    rows = cx.execute(
        "SELECT referee_email, code, created_at FROM referral_redemptions "
        "WHERE lower(owner_email)=? ORDER BY created_at DESC LIMIT ?",
        (owner, int(limit))).fetchall()
    out = []
    for referee_email, code, created_at in rows:
        out.append({
            "name": _masked_name(cx, referee_email),
            "product": _product_for_code(cx, code),
            "redeemed_at": created_at,
        })
    return out
