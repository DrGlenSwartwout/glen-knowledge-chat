"""Pay It Forward: free-month membership comps.

A free month = advancing a membership's billing cycle without charging and
without bumping order_count. Two triggers feed it: a banked counter
(subscriptions.free_months_remaining, for bounties/testimonials) and a live
threshold (has_active_paying_referral, Mechanic A). Every comp/grant is
idempotent via membership_comps.idem_key. Pure module: takes a sqlite
connection, no Flask, no import-time side effects."""

from dashboard import subscriptions as _subs
from dashboard import referrals as _rf


def _norm(email):
    return (email or "").strip().lower()


def init_comps_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS membership_comps ("
        "idem_key TEXT PRIMARY KEY, email TEXT, sub_id INTEGER, "
        "months INTEGER, reason TEXT, created_at TEXT)")
    cx.commit()


def _already(cx, idem_key):
    return cx.execute("SELECT 1 FROM membership_comps WHERE idem_key=? LIMIT 1",
                      (idem_key,)).fetchone() is not None


def has_active_paying_referral(cx, email):
    """True iff a person this email referred is currently an active, full
    (order_count>=1), non-paused membership. One SQL join; excludes trials and
    paused members."""
    _rf.init_tables(cx)
    row = cx.execute(
        "SELECT 1 FROM referral_redemptions rr "
        "JOIN subscriptions s ON lower(s.email)=lower(rr.referee_email) "
        "WHERE lower(rr.owner_email)=? "
        "AND s.status='active' AND s.kind='membership' "
        "AND COALESCE(s.skip_next,0)=0 AND s.order_count>=1 LIMIT 1",
        (_norm(email),)).fetchone()
    return row is not None


def grant_free_month(cx, email, *, months=1, reason, idem_key):
    """Bank `months` free months on the member's active membership. Idempotent
    per idem_key. Returns {sub_id, free_months_remaining} or None if the member
    has no active membership."""
    _subs.migrate_add_free_months(cx)
    init_comps_table(cx)
    e = _norm(email)
    rows = _subs.active_memberships_by_email(cx, e)
    if not rows:
        return None
    sub_id = rows[0]["id"]
    if not _already(cx, idem_key):
        cx.execute(
            "UPDATE subscriptions SET free_months_remaining=COALESCE(free_months_remaining,0)+?, "
            "updated_at=? WHERE id=?", (int(months), _subs._now_iso(), sub_id))
        cx.execute(
            "INSERT INTO membership_comps(idem_key,email,sub_id,months,reason,created_at) "
            "VALUES (?,?,?,?,?,?)", (idem_key, e, sub_id, int(months), reason, _subs._now_iso()))
        cx.commit()
    cur = int(_subs.get(cx, sub_id).get("free_months_remaining") or 0)
    return {"sub_id": sub_id, "free_months_remaining": cur}


def comp_membership_cycle(cx, sub_id, *, reason, idem_key, from_bank=False):
    """Comp ONE cycle for a membership: advance the date without charging or
    bumping order_count, write an audit row. If from_bank, decrement the banked
    counter (and refuse if none remain). Returns True if it comped, else False."""
    _subs.migrate_add_free_months(cx)
    init_comps_table(cx)
    if _already(cx, idem_key):
        return False
    if from_bank:
        row = cx.execute(
            "SELECT COALESCE(free_months_remaining,0), email FROM subscriptions WHERE id=?",
            (sub_id,)).fetchone()
        if not row or int(row[0]) <= 0:
            return False
        email = row[1]
        cx.execute(
            "UPDATE subscriptions SET free_months_remaining=free_months_remaining-1, "
            "updated_at=? WHERE id=?", (_subs._now_iso(), sub_id))
    else:
        row = cx.execute("SELECT email FROM subscriptions WHERE id=?", (sub_id,)).fetchone()
        if not row:
            return False
        email = row[0]
    _subs.comp_cycle(cx, sub_id)
    cx.execute(
        "INSERT INTO membership_comps(idem_key,email,sub_id,months,reason,created_at) "
        "VALUES (?,?,?,?,?,?)", (idem_key, _norm(email), sub_id, 1, reason, _subs._now_iso()))
    cx.commit()
    return True
