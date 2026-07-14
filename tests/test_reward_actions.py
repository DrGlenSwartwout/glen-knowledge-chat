import sqlite3
from dashboard import reward_actions as ra
from dashboard import data_sharing_rewards as dr


def _cx():
    cx = sqlite3.connect(":memory:")
    dr.init_reward_tables(cx)
    ra.init_fulfilled_column(cx)
    return cx


def _pending(cx, email="a@ex.com", rt="store_credit"):
    cx.execute("INSERT INTO member_reward_grants (email, reward_type, tier, status, granted_at) "
               "VALUES (?,?,?, 'pending', '2026-07-14T00:00:00Z')", (email, rt, 3))
    cx.commit()
    return cx.execute("SELECT id FROM member_reward_grants WHERE email=? AND reward_type=?",
                      (email, rt)).fetchone()[0]


def test_fulfill_flips_pending_and_stamps_actor():
    cx = _cx(); gid = _pending(cx)
    assert ra.set_reward_status(cx, gid, "fulfilled", "Rae") is True
    row = cx.execute("SELECT status, granted_by, fulfilled_at FROM member_reward_grants WHERE id=?", (gid,)).fetchone()
    assert row[0] == "fulfilled" and row[1] == "Rae" and row[2] is not None


def test_fulfill_is_idempotent_and_never_downgrades():
    cx = _cx(); gid = _pending(cx)
    ra.set_reward_status(cx, gid, "fulfilled", "Rae")
    # second call: no pending row -> no-op, returns False, does not revert or restamp
    assert ra.set_reward_status(cx, gid, "dismissed", "Shaira") is False
    row = cx.execute("SELECT status, granted_by FROM member_reward_grants WHERE id=?", (gid,)).fetchone()
    assert row[0] == "fulfilled" and row[1] == "Rae"


def test_dismiss_flips_pending():
    cx = _cx(); gid = _pending(cx)
    assert ra.set_reward_status(cx, gid, "dismissed", "owner") is True
    assert cx.execute("SELECT status FROM member_reward_grants WHERE id=?", (gid,)).fetchone()[0] == "dismissed"


def test_actions_registered():
    from dashboard import actions
    ra.register()  # idempotent (guarded); ensures this test is order-independent
    assert actions.get_action("reward.fulfill") is not None
    assert actions.get_action("reward.dismiss") is not None
