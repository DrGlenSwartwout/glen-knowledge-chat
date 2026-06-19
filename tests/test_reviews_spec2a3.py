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


from dashboard import review_scoring as rs


class _Blk:
    def __init__(self, t): self.type = "text"; self.text = t


class _Msg:
    def __init__(self, t): self.content = [_Blk(t)]


class _FakeClient:
    def __init__(self, payload): self._p = payload
    @property
    def messages(self):
        outer = self
        class _M:
            def create(self, **kw): return _Msg(outer._p)
        return _M()


_CAT = [{"sku": "gift-tuningfork", "label": "Tuning fork", "description": "sound/energy"},
        {"sku": "gift-nightlight", "label": "Red nightlight", "description": "sleep"}]


def test_build_gift_prompt_includes_catalog_and_history():
    system, user = rs.build_gift_prompt("loved it, sleep better", {"name": "Longevity"},
                                        ["Magnesium", "Neuro"], _CAT)
    assert "gift-tuningfork" in user and "Longevity" in user and "Magnesium" in user


def test_suggest_gift_returns_valid_sku():
    c = _FakeClient('{"sku": "gift-nightlight", "reason": "they mentioned sleep"}')
    out = rs.suggest_gift(c, "sleep better", {"name": "X"}, [], _CAT)
    assert out == {"sku": "gift-nightlight", "reason": "they mentioned sleep"}


def test_suggest_gift_none_on_invalid_sku():
    c = _FakeClient('{"sku": "gift-nope", "reason": "x"}')
    assert rs.suggest_gift(c, "t", {"name": "X"}, [], _CAT) is None


def test_suggest_gift_none_on_empty_catalog():
    c = _FakeClient('{"sku": "gift-nightlight", "reason": "x"}')
    assert rs.suggest_gift(c, "t", {"name": "X"}, [], []) is None


def test_suggest_gift_none_on_bad_json():
    assert rs.suggest_gift(_FakeClient("not json"), "t", {"name": "X"}, [], _CAT) is None


def test_suggest_gift_strips_dashes():
    c = _FakeClient('{"sku": "gift-tuningfork", "reason": "good — fit"}')
    out = rs.suggest_gift(c, "t", {"name": "X"}, [], _CAT, strip=lambda s: s.replace("—", ","))
    assert "—" not in out["reason"]
