import sqlite3
from dashboard import course_tokens, courses_identity as ci
from dashboard import courses_identity as cid
from dashboard import course_entitlements as ce


def test_no_token_is_level_zero(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    assert ci.member_level_for(cx, None) == 0
    assert ci.member_level_for(cx, "") == 0


def test_valid_course_token_is_level_one(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    course_tokens.init_course_tokens_table(cx)
    raw = course_tokens.mint_course_token(cx, "learner@example.com", "L")
    assert ci.member_level_for(cx, raw) == 1


def test_garbage_token_is_level_zero(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    course_tokens.init_course_tokens_table(cx)
    assert ci.member_level_for(cx, "not-a-real-token") == 0


def _cx():
    c = sqlite3.connect(":memory:")
    course_tokens.init_course_tokens_table(c)
    ce.init_course_entitlements_table(c)
    return c


def test_anon_is_zero():
    assert cid.member_level_for(_cx(), None) == 0
    assert cid.member_level_for(_cx(), "not-a-real-token") == 0


def test_registered_no_entitlement_is_one():
    c = _cx()
    tok = course_tokens.mint_course_token(c, "reg@x.com", "R")
    assert cid.member_level_for(c, tok) == 1


def test_registered_with_cert_is_two():
    c = _cx()
    tok = course_tokens.mint_course_token(c, "paid@x.com", "P")
    ce.grant_cert(c, "paid@x.com", source="stripe", stripe_ref="cs_1")
    assert cid.member_level_for(c, tok) == 2


def test_registered_with_active_membership_is_one():
    # Membership is now the drip-active flag only, not full-cert access — an active
    # membership with no cert/plan entitlement stays at registered level 1.
    c = _cx()
    tok = course_tokens.mint_course_token(c, "sub@x.com", "S")
    ce.grant_membership(c, "sub@x.com", until_epoch=9_999_999_999.0, source="stripe", stripe_ref="sub_1")
    assert cid.member_level_for(c, tok) == 1
    assert ce.drip_active(c, "sub@x.com") is True


def test_registered_with_active_plan_is_two():
    c = _cx()
    tok = course_tokens.mint_course_token(c, "plan@x.com", "PL")
    ce.grant_plan(c, "plan@x.com", until_epoch=9_999_999_999.0, source="stripe", stripe_ref="sub_plan1")
    assert cid.member_level_for(c, tok) == 2


def test_expired_membership_falls_back_to_one():
    c = _cx()
    tok = course_tokens.mint_course_token(c, "old@x.com", "O")
    ce.grant_membership(c, "old@x.com", until_epoch=1000.0, source="stripe", stripe_ref="sub_2")
    # token still resolves to an email (level 1); membership never grants level 2 now anyway
    assert cid.member_level_for(c, tok) == 1
