import json, sqlite3
from dashboard import rewards

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, tags TEXT DEFAULT '[]')")
    cx.execute("CREATE TABLE affiliate_signups (slug TEXT UNIQUE, email TEXT, status TEXT)")
    rewards.init_affiliate_earnings_table(cx)
    return cx

def _person(cx, email, tags):
    cx.execute("INSERT INTO people (email, tags) VALUES (?,?)", (email, json.dumps(tags)))
    cx.commit()

def test_reward_mode_pro_influencer_is_cash():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups VALUES ('jane','jane@x.com','approved')")
    _person(cx, "jane@x.com", ["type:client", "tier:pro-influencer"])
    assert rewards.reward_mode_for_slug(cx, "jane") == "cash"

def test_reward_mode_doctor_is_points():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups VALUES ('doc','doc@x.com','approved')")
    _person(cx, "doc@x.com", ["type:practitioner"])
    assert rewards.reward_mode_for_slug(cx, "doc") == "points"

def test_reward_mode_default_client_is_points():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups VALUES ('cl','cl@x.com','approved')")
    _person(cx, "cl@x.com", ["type:client"])
    assert rewards.reward_mode_for_slug(cx, "cl") == "points"

def test_referrer_email_for_slug():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups VALUES ('jane','jane@x.com','approved')")
    assert rewards.referrer_email_for_slug(cx, "jane") == "jane@x.com"
    assert rewards.referrer_email_for_slug(cx, "nope") is None

def test_affiliate_earnings_accrue_and_pending_total():
    cx = _cx()
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV1", amount_cents=500)
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV2", amount_cents=700)
    assert rewards.pending_cash_total(cx, "jane") == 1200
    # idempotent per order_ref
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV1", amount_cents=500)
    assert rewards.pending_cash_total(cx, "jane") == 1200

def test_settings_defaults():
    s = rewards.load_settings({})
    assert s["referral_reward_pct"] == 0.05
    assert s["cash_out_threshold_cents"] == 10000
    assert s["cash_out_face_pct"] == 0.70
