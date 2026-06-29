"""dashboard/pif_gift_notes.py — Pay It Forward Tier 2: recipient note invites.

Pure module: takes a sqlite connection, no Flask imports, no import-time side
effects. Selects gift redemptions due for a 'how did it help?' invite and tracks
that the invite was sent (note_invited_at). Read/select + a single stamp write.
"""

from dashboard import referrals as _referrals


def _norm(email):
    return (email or "").strip().lower()


def ensure_columns(cx):
    """Additively add the note_invited_at column to referral_redemptions."""
    _referrals.init_tables(cx)
    try:
        cx.execute("ALTER TABLE referral_redemptions ADD COLUMN note_invited_at TEXT")
        cx.commit()
    except Exception:
        pass  # already present


def pending_invites(cx, *, days, max_age_days=60, limit=200):
    """Redemptions between `days` and `max_age_days` old, never invited, with a non-empty
    recipient email.  The max_age_days upper bound prevents blasting the historical backlog
    when the feature flag is first flipped on.
    Returns [{referee_email, owner_email, code, order_ref, created_at}]."""
    ensure_columns(cx)
    cutoff = f"-{int(days)} days"
    max_age = f"-{int(max_age_days)} days"
    rows = cx.execute(
        "SELECT referee_email, owner_email, code, order_ref, created_at "
        "FROM referral_redemptions "
        "WHERE note_invited_at IS NULL "
        "AND TRIM(COALESCE(referee_email,'')) <> '' "
        "AND datetime(created_at) <= datetime('now', ?) "
        "AND datetime(created_at) >= datetime('now', ?) "
        "ORDER BY created_at ASC LIMIT ?",
        (cutoff, max_age, int(limit))).fetchall()
    return [{"referee_email": r[0], "owner_email": r[1], "code": r[2],
             "order_ref": r[3], "created_at": r[4]} for r in rows]


def mark_invited(cx, referee_email, order_ref):
    """Stamp note_invited_at=now for the redemption (idempotency guard)."""
    ensure_columns(cx)
    cx.execute(
        "UPDATE referral_redemptions SET note_invited_at = datetime('now') "
        "WHERE referee_email=? AND order_ref=?",
        (_norm(referee_email), order_ref or ""))
    cx.commit()
