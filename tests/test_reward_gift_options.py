import sqlite3
from dashboard import review_gifts as rg


def _cx():
    cx = sqlite3.connect(":memory:")
    rg.init_reward_gift_options(cx)
    return cx


def test_seed_places_placeholders():
    cx = _cx()
    opts = rg.list_gift_options(cx)
    assert any(o["sku"] == "GIFT-SAMPLE-3" and o["level"] == 3 for o in opts)
    assert rg.reward_options_for_level(cx, 3)   # non-empty from DB


def test_seed_once_not_resurrected_after_delete():
    cx = _cx()
    for o in rg.list_gift_options(cx):
        rg.delete_gift_option(cx, o["id"])
    assert rg.list_gift_options(cx) == []
    rg.init_reward_gift_options(cx)              # re-init (simulates redeploy)
    assert rg.list_gift_options(cx) == []        # marker present -> NOT re-seeded


def test_delete_on_virgin_db_does_not_raise():
    rg.delete_gift_option(sqlite3.connect(":memory:"), 999)   # no init_table call first; must self-init


def test_add_delete_toggle():
    cx = _cx()
    oid = rg.add_gift_option(cx, 3, "GIFT-NEW", "New gift")
    assert any(o["sku"] == "GIFT-NEW" for o in rg.reward_options_for_level(cx, 3))
    rg.set_gift_option_active(cx, oid, 0)
    assert not any(o["sku"] == "GIFT-NEW" for o in rg.reward_options_for_level(cx, 3))  # inactive hidden
    rg.delete_gift_option(cx, oid)
    assert not any(o["sku"] == "GIFT-NEW" for o in rg.list_gift_options(cx))
