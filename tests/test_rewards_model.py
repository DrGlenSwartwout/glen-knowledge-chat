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


def test_referral_pct_for_modules_interpolates():
    from dashboard import rewards
    s = rewards.load_settings({})            # defaults include referral_cert_anchors
    assert rewards.referral_pct_for_modules(0, s) == 0.05     # 5%
    assert rewards.referral_pct_for_modules(6, s) == 0.10     # 10%
    assert rewards.referral_pct_for_modules(12, s) == 0.15    # 15%
    assert abs(rewards.referral_pct_for_modules(3, s) - 0.075) < 1e-9   # midpoint
    assert rewards.referral_pct_for_modules(99, s) == 0.15    # flat beyond last
    assert rewards.referral_pct_for_modules(-4, s) == 0.05    # clamp at 0


def test_referral_pct_for_modules_falls_back_to_flat_when_no_anchors():
    from dashboard import rewards
    s = rewards.load_settings({"referral_cert_anchors": None})
    s2 = dict(s); s2.pop("referral_cert_anchors", None)
    assert rewards.referral_pct_for_modules(12, s2) == s2["referral_reward_pct"]


def test_referral_pct_for_modules_bad_anchors_falls_back():
    from dashboard import rewards
    s = dict(rewards.load_settings({}))
    s["referral_cert_anchors"] = "garbage"
    assert rewards.referral_pct_for_modules(12, s) == s["referral_reward_pct"]


def test_process_payout_points_mode_refuses():
    import sqlite3, pytest
    from dashboard import points
    from dashboard.actions_rewards import process_payout
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE affiliate_signups (slug TEXT UNIQUE, email TEXT, status TEXT)")
    cx.execute("INSERT INTO affiliate_signups VALUES ('doc','doc@x.com','approved')")
    points.init_points_table(cx)
    points.credit(cx, "doc@x.com", value_cents=15000, reason="referral", order_ref="r1")
    with pytest.raises(ValueError):
        process_payout({"slug": "doc", "mode": "points"}, {"cx": cx})
    # balance untouched: no redeem row written
    assert points.balance(cx, "doc@x.com") == 15000


def test_process_payout_cash_mode_still_pays():
    from dashboard.actions_rewards import process_payout
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups VALUES ('jane','jane@x.com','approved')")
    _person(cx, "jane@x.com", ["tier:pro-influencer"])
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV1", amount_cents=12000)
    out = process_payout({"slug": "jane", "mode": "cash"}, {"cx": cx})
    assert out["mode"] == "cash"
    assert out["amount_cents"] == 12000
    assert out["status"] == "paid"
    assert rewards.pending_cash_total(cx, "jane") == 0
