import sqlite3
from dashboard import review_gifts as rg


def _cx():
    return sqlite3.connect(":memory:")


def test_catalog_loads_and_validates():
    cat = rg.load_catalog()
    skus = {g["sku"] for g in cat}
    assert {"gift-toothbrush", "gift-nightlight", "gift-tuningfork"} <= skus
    assert rg.valid_sku("gift-tuningfork") and not rg.valid_sku("gift-nope")


def test_add_and_get_for_review():
    cx = _cx()
    gid = rg.add_suggestion(cx, 7, "a@x.com", "gift-tuningfork", "Tuning fork", "into sound work")
    g = rg.get_for_review(cx, 7)
    assert g["id"] == gid and g["email"] == "a@x.com" and g["status"] == "suggested"
    assert g["gift_sku"] == "gift-tuningfork" and g["gift_label"] == "Tuning fork"


def test_recent_active_gift_cap():
    cx = _cx()
    rg.add_suggestion(cx, 1, "a@x.com", "gift-nightlight", "Night light", "r")
    assert rg.recent_active_gift(cx, "a@x.com", 30) is True          # within window, non-rejected
    assert rg.recent_active_gift(cx, "other@x.com", 30) is False
    # a rejected gift frees the slot
    cx2 = _cx()
    g = rg.add_suggestion(cx2, 1, "b@x.com", "gift-nightlight", "Night light", "r")
    rg.set_status(cx2, g, "rejected")
    assert rg.recent_active_gift(cx2, "b@x.com", 30) is False


def test_approve_swap_pending_and_fulfill():
    cx = _cx()
    gid = rg.add_suggestion(cx, 1, "a@x.com", "gift-nightlight", "Night light", "r")
    rg.swap_sku(cx, gid, "gift-toothbrush", "Bamboo toothbrush")
    rg.set_status(cx, gid, "approved", by="Glen")
    pend = rg.pending_for(cx, "a@x.com")
    assert len(pend) == 1 and pend[0]["gift_sku"] == "gift-toothbrush" and pend[0]["status"] == "approved"
    rg.mark_fulfilled(cx, gid, 555)
    assert rg.pending_for(cx, "a@x.com") == []                       # fulfilled -> not pending
    assert rg.get_for_review(cx, 1)["fulfilled_order_id"] == 555


def test_suggested_queue():
    cx = _cx()
    a = rg.add_suggestion(cx, 1, "a@x.com", "gift-nightlight", "Night light", "r")
    b = rg.add_suggestion(cx, 2, "b@x.com", "gift-toothbrush", "Toothbrush", "r")
    rg.set_status(cx, a, "approved")
    q = rg.suggested_queue(cx)
    assert [g["review_id"] for g in q] == [2]                        # only still-suggested
