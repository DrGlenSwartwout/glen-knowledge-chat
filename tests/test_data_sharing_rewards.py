import sqlite3
from dashboard import data_sharing_rewards as dr

def _cx():
    cx = sqlite3.connect(":memory:")
    dr.init_reward_tables(cx)
    return cx

def test_tier2_grants_auto_and_is_idempotent():
    cx = _cx()
    calls = []
    fn = lambda c, e: calls.append(e)
    first = dr.grant_rewards_for_tier(cx, "a@ex.com", 2, free_unlock_fn=fn)
    assert "founding_badge" in first and "free_reveal_unlock" in first
    # Second run grants nothing new, no duplicate ledger rows, no second unlock
    second = dr.grant_rewards_for_tier(cx, "a@ex.com", 2, free_unlock_fn=fn)
    assert second == []
    rows = cx.execute("SELECT COUNT(*) FROM member_reward_grants WHERE email='a@ex.com'").fetchone()[0]
    assert rows == 3  # founding_badge, free_reveal_unlock, early_access
    assert calls == ["a@ex.com"]  # unlock fired exactly once

def test_tier3_writes_pending_not_auto():
    cx = _cx()
    dr.grant_rewards_for_tier(cx, "b@ex.com", 3)
    status = cx.execute("SELECT status FROM member_reward_grants "
                        "WHERE email='b@ex.com' AND reward_type='store_credit'").fetchone()[0]
    assert status == "pending"

def test_never_downgrade():
    cx = _cx()
    dr.grant_rewards_for_tier(cx, "c@ex.com", 3)
    cx.execute("UPDATE member_reward_grants SET status='fulfilled' "
               "WHERE email='c@ex.com' AND reward_type='store_credit'")
    cx.commit()
    # Re-running must NOT move fulfilled back to pending
    dr.grant_rewards_for_tier(cx, "c@ex.com", 3)
    status = cx.execute("SELECT status FROM member_reward_grants "
                        "WHERE email='c@ex.com' AND reward_type='store_credit'").fetchone()[0]
    assert status == "fulfilled"
