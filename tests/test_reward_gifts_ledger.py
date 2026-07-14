import sqlite3
from dashboard import review_gifts as rg


def _cx():
    cx = sqlite3.connect(":memory:")
    rg.init_table(cx); rg.migrate_reward_columns(cx)
    return cx


def test_add_reward_gift_and_pending():
    cx = _cx()
    gid = rg.add_reward_gift(cx, "A@Ex.com", "GIFT-SAMPLE-3", "Placeholder", 42)
    p = rg.pending_reward_for(cx, "a@ex.com")
    assert len(p) == 1 and p[0]["id"] == gid
    assert p[0]["source"] == "reward" and p[0]["status"] == "approved"
    assert p[0]["reward_grant_id"] == 42


def test_pending_reward_excludes_review_source():
    cx = _cx()
    rg.add_suggestion(cx, 7, "a@ex.com", "SKU1", "Review gift", "nice")
    # approve it so it'd be pending under pending_for, but it's source review
    gid = cx.execute("SELECT id FROM review_gifts WHERE review_id=7").fetchone()[0]
    rg.set_status(cx, gid, "approved", "op")
    assert rg.pending_reward_for(cx, "a@ex.com") == []          # reward-only
    assert len(rg.pending_for(cx, "a@ex.com")) == 1             # review path intact


def test_mark_fulfilled_consumes_pending():
    cx = _cx()
    gid = rg.add_reward_gift(cx, "a@ex.com", "GIFT-SAMPLE-3", "P", 1)
    rg.mark_fulfilled(cx, gid, 555)
    assert rg.pending_reward_for(cx, "a@ex.com") == []


def test_reward_catalog_by_level():
    opts3 = rg.reward_options_for_level(3)
    assert all(o["level"] == 3 and o.get("active") for o in opts3)
    assert any(o["sku"] for o in opts3)   # seeded placeholders exist
