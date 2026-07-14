import sqlite3
from dashboard import reward_actions as ra, data_sharing_rewards as dr, review_gifts as rg


def _cx():
    cx = sqlite3.connect(":memory:")
    dr.init_reward_tables(cx); ra.init_fulfilled_column(cx)
    rg.init_table(cx); rg.migrate_reward_columns(cx)
    return cx


def _pending(cx, email="a@ex.com", tier=3):
    cx.execute("INSERT INTO member_reward_grants (email, reward_type, tier, status, granted_at) "
               "VALUES (?, 'store_credit', ?, 'pending', '2026-07-14T00:00:00Z')", (email, tier))
    cx.commit()
    return cx.execute("SELECT id FROM member_reward_grants WHERE email=?", (email,)).fetchone()[0]


def test_select_gift_creates_gift_and_fulfills():
    cx = _cx(); gid = _pending(cx)
    out = ra.select_gift(cx, gid, "GIFT-SAMPLE-3", "Rae")
    assert out["ok"] is True
    assert rg.pending_reward_for(cx, "a@ex.com")[0]["gift_sku"] == "GIFT-SAMPLE-3"
    assert cx.execute("SELECT status FROM member_reward_grants WHERE id=?", (gid,)).fetchone()[0] == "fulfilled"


def test_select_gift_rejects_wrong_level_sku():
    cx = _cx(); gid = _pending(cx, tier=3)
    out = ra.select_gift(cx, gid, "GIFT-SAMPLE-4", "Rae")   # tier-4 sku for a tier-3 grant
    assert out["ok"] is False and rg.pending_reward_for(cx, "a@ex.com") == []


def test_action_registered():
    from dashboard import actions
    ra.register()  # idempotent (guarded); ensures this test is order-independent
    assert actions.get_action("reward.select_gift") is not None


def test_select_gift_action_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv("REWARD_GIFTS_ENABLED", raising=False)
    cx = _cx(); gid = _pending(cx)
    from dashboard import actions
    out = actions.get_action("reward.select_gift").executor({"grant_id": gid, "sku": "GIFT-SAMPLE-3"}, {"cx": cx})
    assert out["ok"] is False
    assert rg.pending_reward_for(cx, "a@ex.com") == []                                  # no gift created
    assert cx.execute("SELECT status FROM member_reward_grants WHERE id=?", (gid,)).fetchone()[0] == "pending"  # grant untouched


def test_select_gift_no_duplicate_for_same_grant():
    cx = _cx(); gid = _pending(cx)
    rg.add_reward_gift(cx, "a@ex.com", "GIFT-SAMPLE-3", "P", gid)   # simulate an orphaned prior gift for this grant
    out = ra.select_gift(cx, gid, "GIFT-SAMPLE-3", "Rae")
    assert out["ok"] is True
    assert len(rg.pending_reward_for(cx, "a@ex.com")) == 1          # still ONE gift, not two
