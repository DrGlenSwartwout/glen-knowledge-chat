import sqlite3
from dashboard import referrals as rf


def _cx():
    return sqlite3.connect(":memory:")


def test_get_or_create_code_stable_and_unique():
    cx = _cx()
    c1 = rf.get_or_create_code(cx, "Owner@X.com")
    c2 = rf.get_or_create_code(cx, "owner@x.com")          # same person (lowercased) -> same code
    assert c1 == c2 and c1
    c3 = rf.get_or_create_code(cx, "other@x.com")
    assert c3 != c1
    assert rf.owner_of(cx, c1) == "owner@x.com"
    assert rf.owner_of(cx, "NOPE") is None


def test_resolve_valid_and_guards():
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    # valid referee
    assert rf.resolve(cx, code, "friend@x.com", pct=10) == {"owner_email": "owner@x.com", "coupon_pct": 10}
    # self-referral blocked (case-insensitive)
    assert rf.resolve(cx, code, "OWNER@x.com", pct=10) is None
    # unknown code
    assert rf.resolve(cx, "NOPE", "friend@x.com", pct=10) is None


def test_one_redemption_per_referee_ever():
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    assert rf.has_redeemed(cx, "friend@x.com") is False
    assert rf.record_redemption(cx, code, "owner@x.com", "Friend@x.com", "INV-1") is True
    assert rf.has_redeemed(cx, "friend@x.com") is True       # lowercased
    # a second redemption by the same referee is a no-op insert, and resolve now blocks
    assert rf.record_redemption(cx, code, "owner@x.com", "friend@x.com", "INV-2") is False
    assert rf.resolve(cx, code, "friend@x.com", pct=10) is None


import importlib


def _reload_ref_app(monkeypatch, tmp_path, referrals="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", referrals)
    monkeypatch.setenv("REFERRAL_PCT", "10")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_resolve_checkout_coupon_pct(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "_active_coupon_pct", lambda: None)
    import sqlite3
    from dashboard import referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        code = rf.get_or_create_code(cx, "owner@x.com")
    # valid referee -> referral pct + ctx
    pct, ctx = appmod._resolve_checkout_coupon_pct(code, "friend@x.com")
    assert pct == 10 and ctx == {"code": code, "owner_email": "owner@x.com"}
    # self-referral -> falls back to daily (None here), no ctx
    pct, ctx = appmod._resolve_checkout_coupon_pct(code, "owner@x.com")
    assert pct is None and ctx is None
    # daily coupon beats a smaller referral -> max wins
    monkeypatch.setattr(appmod, "_active_coupon_pct", lambda: 15)
    pct, ctx = appmod._resolve_checkout_coupon_pct(code, "friend@x.com")
    assert pct == 15 and ctx == {"code": code, "owner_email": "owner@x.com"}


def test_resolve_flag_off(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path, referrals="false")
    monkeypatch.setattr(appmod, "_active_coupon_pct", lambda: 5)
    pct, ctx = appmod._resolve_checkout_coupon_pct("ANYCODE", "friend@x.com")
    assert pct == 5 and ctx is None          # referral ignored when flag off


def test_my_code_endpoint(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    c.set_cookie("rm_reorder_email", "owner@x.com")     # _reorder_email_from_cookie source
    r1 = c.get("/api/referral/my-code").get_json()
    r2 = c.get("/api/referral/my-code").get_json()
    assert r1["code"] and r1["code"] == r2["code"]       # stable


def test_my_code_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path, referrals="false")
    c = appmod.app.test_client(); c.set_cookie("rm_reorder_email", "owner@x.com")
    assert c.get("/api/referral/my-code").status_code == 404


def test_record_referral_if_any(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path)
    import sqlite3
    from dashboard import referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        code = rf.get_or_create_code(cx, "owner@x.com")
    ctx = {"code": code, "owner_email": "owner@x.com"}
    assert appmod._record_referral_if_any(ctx, "friend@x.com", "INV-1") is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rf.has_redeemed(cx, "friend@x.com") is True
    # None ctx -> no-op; second time for same referee -> False (already redeemed)
    assert appmod._record_referral_if_any(None, "friend@x.com", "INV-2") is False
    assert appmod._record_referral_if_any(ctx, "friend@x.com", "INV-2") is False


def test_record_referral_flag_off(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path, referrals="false")
    ctx = {"code": "X", "owner_email": "owner@x.com"}
    assert appmod._record_referral_if_any(ctx, "friend@x.com", "INV-1") is False


def test_referral_enabled_endpoint(monkeypatch, tmp_path):
    import importlib
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("REFERRALS", "true")
    import app as appmod; importlib.reload(appmod)
    assert appmod.app.test_client().get("/api/referral/enabled").get_json() == {"enabled": True}
    monkeypatch.setenv("REFERRALS", "false"); importlib.reload(appmod)
    assert appmod.app.test_client().get("/api/referral/enabled").get_json() == {"enabled": False}
