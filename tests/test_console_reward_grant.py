import sqlite3

from dashboard import console_next_action as na
from dashboard import data_sharing_rewards as dr
from dashboard import reward_actions as ra
from dashboard import household_holds as hh


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row  # REQUIRED — matches prod; without it r["id"] raises
    dr.init_reward_tables(cx)
    ra.init_fulfilled_column(cx)
    # list_actionable() runs every record-type lister; #884's household lister is
    # the one that raises on a missing table (others tolerate it), so create it.
    hh.init_hold_tables(cx)
    return cx


def _pending(cx, email="a@ex.com", rt="store_credit"):
    cx.execute(
        "INSERT INTO member_reward_grants (email, reward_type, tier, status, granted_at) "
        "VALUES (?,?,?, 'pending', '2026-07-14T00:00:00Z')", (email, rt, 3))
    cx.commit()
    return cx.execute(
        "SELECT id FROM member_reward_grants WHERE email=? AND reward_type=?",
        (email, rt)).fetchone()[0]


def test_resolver_descriptor_shape(monkeypatch):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    cx = _cx()
    gid = _pending(cx)
    recs = na._reward_records(cx)
    assert len(recs) == 1 and recs[0]["id"] == gid
    d = na.resolve_reward_grant(recs[0])
    assert d["type"] == "reward_grant" and d["actionable"] is True
    assert d["action"]["keys"] == ["reward.fulfill"]
    assert d["action"]["body"]["grant_id"] == gid
    assert d["secondary"]["action"]["keys"] == ["reward.dismiss"]
    assert d["secondary"]["action"]["body"]["grant_id"] == gid


def test_lister_empty_when_flag_off(monkeypatch):
    monkeypatch.delenv("DATA_SHARING_REWARD_ENABLED", raising=False)
    cx = _cx()
    _pending(cx)
    assert na._reward_records(cx) == []


def test_fulfilled_row_not_surfaced(monkeypatch):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    cx = _cx()
    gid = _pending(cx)
    ra.set_reward_status(cx, gid, "fulfilled", "Rae")
    assert na._reward_records(cx) == []


def test_appears_in_list_actionable(monkeypatch):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    from dashboard import biofield_reveals, ff_match_drafts
    cx = _cx()
    biofield_reveals.init_table(cx)
    ff_match_drafts.init_table(cx)
    try:
        from dashboard import orders as _ord
        _ord.init_orders_table(cx)
    except Exception:
        cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "email TEXT, name TEXT, items_json TEXT, total_cents INTEGER, "
                   "status TEXT, created_at TEXT)")
    monkeypatch.setattr(na, "_handoff_records", lambda cx: [])
    _pending(cx)
    items = na.list_actionable(cx)
    reward_items = [d for d in items if d["type"] == "reward_grant"]
    assert len(reward_items) == 1


def test_reward_card_links_to_picker_when_gifts_enabled(monkeypatch):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    cx = _cx()
    gid = _pending(cx)
    d = na.resolve_reward_grant(na._reward_records(cx)[0])
    assert d["action"]["kind"] == "link" and d["action"]["url"] == "/console/rewards"
    assert d["secondary"]["action"]["keys"] == ["reward.dismiss"]


def test_reward_card_dispatches_fulfill_when_gifts_disabled(monkeypatch):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    monkeypatch.delenv("REWARD_GIFTS_ENABLED", raising=False)
    cx = _cx()
    gid = _pending(cx)
    d = na.resolve_reward_grant(na._reward_records(cx)[0])
    assert d["action"]["keys"] == ["reward.fulfill"]
