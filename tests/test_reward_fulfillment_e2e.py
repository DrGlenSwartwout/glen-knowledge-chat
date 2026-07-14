"""End-to-end trace for data-sharing reward Phase 2: a Tier-3 opt-in creates a
pending reward, it surfaces on the console next-action queue, and an operator
fulfill flips it to terminal and removes it from the queue. Also confirms the
Tier-2 biofield free-reveal unlock fired along the way."""
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    return app


def _seed(tmp_db, email):
    from dashboard import client_portal as cp, biofield_reveals as br
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        br.init_table(cx)
        token, _ = cp.upsert_portal(cx, email, "M", {})
        cx.execute(
            "INSERT INTO biofield_reveals (email, scan_date, created_at, updated_at) "
            "VALUES (?,?,?,?)",
            (email, "2026-07-01", "2026-07-01T00:00:00Z", "2026-07-01T00:00:00Z"))
        cx.commit()
        rid = cx.execute("SELECT id FROM biofield_reveals WHERE email=? ORDER BY id DESC LIMIT 1",
                         (email,)).fetchone()[0]
    return token, rid


def _rowcx(tmp_db):
    cx = sqlite3.connect(str(tmp_db))
    cx.row_factory = sqlite3.Row
    return cx


def test_tier3_optin_surfaces_then_fulfills(monkeypatch, tmp_db):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    from dashboard import console_next_action as na, reward_actions as ra, biofield_reveals as br
    email = "member@ex.com"
    token, rid = _seed(tmp_db, email)

    # Tier-3 opt-in (attributed "share my story")
    r = app.app.test_client().post(f"/api/portal/{token}/sharing",
                                   json={"toggles": {"share_story": True}})
    assert r.status_code == 200
    assert r.get_json()["consent"]["tier"] == 3

    # A store_credit reward is pending
    cx = _rowcx(tmp_db)
    grow = cx.execute("SELECT id, status FROM member_reward_grants "
                      "WHERE email=? AND reward_type='store_credit'", (email,)).fetchone()
    assert grow is not None and grow["status"] == "pending"
    gid = grow["id"]

    # Tier-2 auto biofield unlock also fired on the way (cumulative rewards)
    assert br.free_unlock_reveal_id(cx, email) == rid

    # It surfaces as a console next-action reward record (the full list_actionable
    # aggregate + route table-init is covered by test_next_action_reward_route.py;
    # here we exercise the reward-specific lister/resolver chain).
    recs = na._reward_records(cx)
    rg = [r for r in recs if r["id"] == gid]
    assert len(rg) == 1
    assert na.resolve_reward_grant(rg[0])["action"]["keys"] == ["reward.fulfill"]

    # Operator fulfills it
    assert ra.set_reward_status(cx, gid, "fulfilled", "Rae") is True

    # Gone from the queue; row is terminal and stamped
    assert not [r for r in na._reward_records(cx) if r["id"] == gid]
    row = cx.execute("SELECT status, granted_by, fulfilled_at FROM member_reward_grants WHERE id=?",
                     (gid,)).fetchone()
    assert row["status"] == "fulfilled" and row["granted_by"] == "Rae" and row["fulfilled_at"] is not None
    cx.close()
