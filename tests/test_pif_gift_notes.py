import sqlite3
from datetime import datetime, timedelta

from dashboard import pif_gift_notes as gn
from dashboard import referrals


def _cx():
    cx = sqlite3.connect(":memory:")
    referrals.init_tables(cx)
    gn.ensure_columns(cx)
    return cx


def _ago(days):
    """Return an ISO-format datetime string `days` ago."""
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


def _redeem(cx, referee, owner, code, order_ref, created_at):
    cx.execute(
        "INSERT INTO referral_redemptions (referee_email, code, owner_email, order_ref, created_at) "
        "VALUES (?,?,?,?,?)", (referee, code, owner, order_ref, created_at))
    cx.commit()


def test_pending_selects_old_uninvited():
    cx = _cx()
    _redeem(cx, "b@x.com", "a@x.com", "C1", "o1", _ago(20))  # 20 days old: past delay, inside max_age
    rows = gn.pending_invites(cx, days=14)
    assert len(rows) == 1
    assert rows[0]["referee_email"] == "b@x.com"
    assert rows[0]["owner_email"] == "a@x.com"
    assert rows[0]["order_ref"] == "o1"


def test_pending_excludes_too_recent():
    cx = _cx()
    # created_at = now: not yet `days` old
    cx.execute("INSERT INTO referral_redemptions (referee_email, code, owner_email, order_ref, created_at) "
               "VALUES ('b@x.com','C1','a@x.com','o1', datetime('now'))")
    cx.commit()
    assert gn.pending_invites(cx, days=14) == []


def test_pending_excludes_blank_email():
    cx = _cx()
    _redeem(cx, "", "a@x.com", "C1", "o1", _ago(20))
    assert gn.pending_invites(cx, days=14) == []


def test_mark_invited_makes_idempotent():
    cx = _cx()
    _redeem(cx, "b@x.com", "a@x.com", "C1", "o1", _ago(20))
    gn.mark_invited(cx, "b@x.com", "o1")
    assert gn.pending_invites(cx, days=14) == []  # no longer pending


def test_limit_respected():
    cx = _cx()
    for i in range(3):
        _redeem(cx, f"r{i}@x.com", "a@x.com", f"C{i}", f"o{i}", _ago(20))
    assert len(gn.pending_invites(cx, days=14, limit=2)) == 2


def test_max_age_excludes_too_old():
    """A redemption older than max_age_days must NOT be selected (no backfill blast)."""
    cx = _cx()
    _redeem(cx, "old@x.com", "a@x.com", "C_old", "o_old", _ago(200))  # 200 days >> 60-day window
    rows = gn.pending_invites(cx, days=14, max_age_days=60)
    assert rows == []


def test_max_age_includes_in_window():
    """A redemption inside the window (>days old, <max_age_days old) IS selected."""
    cx = _cx()
    _redeem(cx, "b@x.com", "a@x.com", "C1", "o1", _ago(20))  # 20 days: >14 delay, <60 max_age
    rows = gn.pending_invites(cx, days=14, max_age_days=60)
    assert len(rows) == 1
    assert rows[0]["referee_email"] == "b@x.com"
