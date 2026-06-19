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


import importlib


def _reload_gift_app(monkeypatch, tmp_path, gifts="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REVIEWS_ENABLED", "true")
    monkeypatch.setenv("REVIEWS_VIDEO", "true")
    monkeypatch.setenv("REVIEWS_GIFTS", gifts)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed_scored_video(appmod, slug, email="b@x.com", video_points=5, written=0):
    import sqlite3
    from dashboard import product_reviews as pr, review_video_jobs as vj
    d = appmod._REVIEW_MEDIA_DIR / slug; d.mkdir(parents=True, exist_ok=True)
    (d / "v.webm").write_bytes(b"X")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, slug, email, "B", 5, video_kind="upload", video_ref="v.webm")
        pr.set_ai_result(cx, rid, written, "", 0); pr.set_points(cx, rid, written)
        vj.enqueue(cx, rid)
    return rid


def _patch_pipeline(monkeypatch, video_points=5, gift_sku="gift-nightlight"):
    import journal_blueprint
    monkeypatch.setattr(journal_blueprint, "_whisper_transcribe",
                        lambda p: {"text": "great review", "duration": 12.0, "words": []})
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_video", lambda *a, **k: {
        "video_points": video_points, "publish_risk": False, "risk_reasons": "", "recommend_publish": True})
    monkeypatch.setattr(rs, "suggest_gift", lambda *a, **k: ({"sku": gift_sku, "reason": "fits"} if gift_sku else None))


def test_worker_suggests_gift_at_5_points(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_scored_video(appmod, slug)
    _patch_pipeline(monkeypatch, video_points=5)
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        g = rg.get_for_review(cx, rid)
    assert g is not None and g["status"] == "suggested" and g["gift_sku"] == "gift-nightlight"


def test_worker_no_gift_under_5(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_scored_video(appmod, slug)
    _patch_pipeline(monkeypatch, video_points=3)        # total 3
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rg.get_for_review(cx, rid) is None


def test_worker_monthly_cap(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rg.add_suggestion(cx, 999, "b@x.com", "gift-toothbrush", "Toothbrush", "earlier")  # within 30d
    rid = _seed_scored_video(appmod, slug, email="b@x.com")
    _patch_pipeline(monkeypatch, video_points=5)
    appmod._drain_review_videos()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rg.get_for_review(cx, rid) is None        # capped


def test_worker_gift_flag_off(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path, gifts="false")
    assert appmod._REVIEWS_GIFTS is False
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_scored_video(appmod, slug)
    _patch_pipeline(monkeypatch, video_points=5)
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rg.get_for_review(cx, rid) is None


from dashboard.rbac import Actor, OWNER


def test_gift_actions_approve_swap_reject(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    import sqlite3
    from dashboard import product_reviews as pr, review_gifts as rg
    from dashboard import dispatch as d
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, video_kind="upload", video_ref="v.webm")
        gid = rg.add_suggestion(cx, rid, "a@x.com", "gift-nightlight", "Night light", "r")
        # approve with a swap sku
        res = d.dispatch_action(cx, "reviews.gift_approve", {"review_id": rid, "sku": "gift-tuningfork"},
                                Actor(role=OWNER, name="Glen"), source="panel")
        assert res["status"] == "done"
        g = rg.get_for_review(cx, rid)
        assert g["status"] == "approved" and g["gift_sku"] == "gift-tuningfork" and g["approved_by"] == "Glen"
        # reject path on a fresh review
        rid2 = pr.upsert_review(cx, "x", "b@x.com", "Bob", 5, video_kind="upload", video_ref="v.webm")
        g2 = rg.add_suggestion(cx, rid2, "b@x.com", "gift-nightlight", "Night light", "r")
        d.dispatch_action(cx, "reviews.gift_reject", {"review_id": rid2},
                          Actor(role=OWNER, name="Glen"), source="panel")
        assert rg.get_for_review(cx, rid2)["status"] == "rejected"


def test_console_reviews_includes_gift_and_catalog(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""
    import sqlite3
    from dashboard import product_reviews as pr, review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, video_kind="upload", video_ref="v.webm")
        rg.add_suggestion(cx, rid, "a@x.com", "gift-nightlight", "Night light", "fits sleep")
    c = appmod.app.test_client()
    rows = c.get("/api/console/reviews").get_json()["pending"]
    row = next(r for r in rows if r["email"] == "a@x.com")
    assert row["gift"]["gift_sku"] == "gift-nightlight" and row["gift"]["status"] == "suggested"
    cat = c.get("/api/console/gift-catalog").get_json()["catalog"]
    assert any(g["sku"] == "gift-tuningfork" for g in cat)


def test_order_entry_adds_gift_line_and_fulfills(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    # the in-house order route is OWNER-gated; resolve_actor returns OWNER for the console key
    import dashboard as _d
    _d.CONSOLE_SECRET = "k"; appmod.CONSOLE_SECRET = "k"
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    p = appmod._get_product(slug)
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        gid = rg.add_suggestion(cx, 1, "buyer@x.com", "gift-nightlight", "Red nightlight", "r")
        rg.set_status(cx, gid, "approved", by="Glen")
    c = appmod.app.test_client()
    body = {"customer": {"email": "buyer@x.com", "name": "Buyer",
                         "address": {"street": "1 A St", "city": "Hilo", "state": "HI", "zip": "96720"}},
            "lines": [{"slug": slug, "qty": 1}]}
    r = c.post("/api/orders/manual", json=body, headers={"X-Console-Key": "k"}).get_json()
    assert r["ok"]
    oid = r["order_id"]
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import orders as o
        order = o.list_orders_by_email(cx, "buyer@x.com")[0]
        names = [it.get("name", "") for it in order["items"]]
        assert any("Red nightlight" in n for n in names)            # $0 gift line present
        gift_lines = [it for it in order["items"] if it.get("gift")]
        assert gift_lines and gift_lines[0]["unit_cents"] == 0
        assert rg.get_for_review(cx, 1)["fulfilled_order_id"] == oid  # marked fulfilled
        assert rg.pending_for(cx, "buyer@x.com") == []
    # a SECOND order does not re-add the gift
    r2 = c.post("/api/orders/manual", json=body, headers={"X-Console-Key": "k"}).get_json()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import orders as o
        order2 = [x for x in o.list_orders_by_email(cx, "buyer@x.com") if x["id"] == r2["order_id"]][0]
        assert not any(it.get("gift") for it in order2["items"])
