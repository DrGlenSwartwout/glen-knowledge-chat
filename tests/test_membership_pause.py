"""Two-step soft-pause route: GET previews (no state change), POST confirms and
sets skip_next. Tokened via the same membership_cancel token. Seeds the real
LOG_DB (mirrors test_membership_reminder_cancel)."""
import sqlite3
from datetime import datetime, timedelta, timezone

import app as appmod
from dashboard import subscriptions as subs

EMAIL = "pause-test@example.com"


def _seed_membership_and_token():
    cx = sqlite3.connect(appmod.LOG_DB)
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (EMAIL,))
    cx.execute("DELETE FROM auth_tokens WHERE email=?", (EMAIL,))
    sid = subs.create_membership(cx, email=EMAIL, stripe_customer_id="c",
                                 stripe_payment_method_id="p", amount_cents=9900,
                                 next_charge_date="2026-07-20", cadence_months=1)
    tok = "pausetok123"
    exp = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
               "VALUES (?,?,?,?,?)",
               (appmod._hash_token(tok), EMAIL, "membership_cancel",
                datetime.now(timezone.utc).isoformat(), exp))
    cx.commit()
    cx.close()
    return sid, tok


def _cleanup():
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (EMAIL,))
    cx.execute("DELETE FROM auth_tokens WHERE email=?", (EMAIL,))
    cx.commit()
    cx.close()


def _skip(sid):
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    v = subs.get(cx, sid)["skip_next"]
    cx.close()
    return v


def test_get_previews_without_pausing():
    sid, tok = _seed_membership_and_token()
    try:
        r = appmod.app.test_client().get(f"/membership/pause/{tok}")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert '"confirmed": false' in body
        assert "2026-08-20" in body          # resume date previewed (one cycle later)
        assert _skip(sid) == 0               # GET did NOT change state
    finally:
        _cleanup()


def test_post_confirms_and_sets_skip():
    sid, tok = _seed_membership_and_token()
    try:
        r = appmod.app.test_client().post(f"/membership/pause/{tok}")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert '"confirmed": true' in body and '"ok": true' in body
        assert _skip(sid) == 1               # next charge now skipped
    finally:
        _cleanup()


def test_post_cadence_sets_months():
    sid, tok = _seed_membership_and_token()
    try:
        r = appmod.app.test_client().post(f"/membership/pause/{tok}",
                                          data={"mode": "cadence", "months": "3"})
        assert r.status_code == 200
        cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
        assert subs.get(cx, sid)["cadence_months"] == 3
        cx.close()
    finally:
        _cleanup()


def test_invalid_token_is_not_valid():
    r = appmod.app.test_client().get("/membership/pause/garbage-token")
    assert r.status_code == 200
    assert '"valid": false' in r.get_data(as_text=True)
