import importlib
import sqlite3
from dashboard import referrals as rf, points


def _reload(monkeypatch, tmp_path, pct="20", tier2=True):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    monkeypatch.setenv("REFERRER_REWARD_PCT", pct)
    import app as appmod
    importlib.reload(appmod)
    monkeypatch.setattr(appmod, "REFERRAL_TIER2_ENABLED", tier2)
    return appmod


def _order(referee="patient@x.com", total=7000, shipping=1300, get=0):
    return {"email": referee, "total_cents": total, "shipping_cents": shipping, "get_cents": get}


def test_dispensary_portal_pays_no_l1_but_stamps(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, tier2=False)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "DISP", "doc@x.com", "patient@x.com", "INV-1",
                             kind="dispensary_portal")
        credited = appmod._settle_referrer_reward(cx, _order(), "INV-1")
    assert credited == 0
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "doc@x.com") == 0        # no L1 to the practitioner
        assert rf.redemption_by_order_ref(cx, "INV-1")["rewarded_at"]  # stamped: no replay


def test_dispensary_portal_pays_no_l2_via_old_path(monkeypatch, tmp_path):
    # The old _settle_referrer_reward must NOT credit L2 for a dispensary_portal row;
    # dispensary L2 is owned solely by settle_dispensary_l2 now.
    appmod = _reload(monkeypatch, tmp_path, tier2=True)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
        rf.record_redemption(cx, "DISP", "doc@x.com", "patient@x.com", "INV-1",
                             kind="dispensary_portal")
        appmod._settle_referrer_reward(cx, _order(), "INV-1")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "upline@x.com") == 0   # old path no longer pays dispensary L2
        assert points.balance(cx, "doc@x.com") == 0      # L1 already suppressed


def test_ambassador_referral_still_pays_l1(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, tier2=False)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "AMB", "amb@x.com", "friend@x.com", "INV-2")  # kind='referral'
        credited = appmod._settle_referrer_reward(cx, _order(referee="friend@x.com"), "INV-2")
    assert credited == 1140   # 5700 * 20 // 100 — Ambassador flow unchanged
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "amb@x.com") == 1140
