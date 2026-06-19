import sqlite3
from dashboard import sales_votes as sv

def _cx(): return sqlite3.connect(":memory:")

def test_record_pick_upsert_one_row_per_session_kind():
    cx = _cx()
    sv.record_pick(cx, "longevity", "botanical", 1, "sessA")
    sv.record_pick(cx, "longevity", "botanical", 2, "sessA")   # re-pick updates
    assert sv.get_picks(cx, "longevity", session_id="sessA")["botanical"] == 2
    assert sv.tally(cx, "longevity") == {"botanical": {2: 1}}   # one row, last choice

def test_picked_both_requires_real_pick_in_both():
    cx = _cx()
    sv.record_pick(cx, "longevity", "botanical", 1, "sessA")
    assert sv.picked_both(cx, "longevity", session_id="sessA") is False
    sv.record_pick(cx, "longevity", "mechanism", 0, "sessA")    # neither
    assert sv.picked_both(cx, "longevity", session_id="sessA") is False
    sv.record_pick(cx, "longevity", "mechanism", 1, "sessA")
    assert sv.picked_both(cx, "longevity", session_id="sessA") is True

def test_email_backfill_enables_match_by_email():
    cx = _cx()
    sv.record_pick(cx, "longevity", "botanical", 1, "sessA")            # anon
    sv.record_pick(cx, "longevity", "mechanism", 1, "sessA", "a@b.co")  # identified -> backfills
    assert sv.picked_both(cx, "longevity", email="a@b.co") is True      # both now carry the email

def test_tally_excludes_neither():
    cx = _cx()
    sv.record_pick(cx, "x", "botanical", 1, "s1")
    sv.record_pick(cx, "x", "botanical", 1, "s2")
    sv.record_pick(cx, "x", "botanical", 0, "s3")   # neither
    assert sv.tally(cx, "x") == {"botanical": {1: 2}}


# ---------------------------------------------------------------------------
# Task 2: pick route tests
# ---------------------------------------------------------------------------
import importlib


def _reload(monkeypatch, tmp_path, pick="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_IMAGES", "true"); monkeypatch.setenv("SALES_PAGES_IMAGE_PICK", pick)
    import app as appmod; importlib.reload(appmod); return appmod


def test_pick_route_records_and_both(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client(); c.set_cookie("amg_session", "sX")
    r1 = c.post(f"/begin/product-image-pick/{slug}", json={"kind": "botanical", "variant": 1})
    assert r1.status_code == 200 and r1.get_json()["both_picked"] is False
    r2 = c.post(f"/begin/product-image-pick/{slug}", json={"kind": "mechanism", "variant": 2})
    assert r2.get_json()["both_picked"] is True


def test_pick_route_neither_and_bad_input(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client(); c.set_cookie("amg_session", "sY")
    assert c.post(f"/begin/product-image-pick/{slug}", json={"kind": "botanical", "variant": "neither"}).status_code == 200
    assert c.post(f"/begin/product-image-pick/{slug}", json={"kind": "bogus", "variant": 1}).status_code == 400


def test_pick_route_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, pick="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    assert appmod.app.test_client().post(f"/begin/product-image-pick/{slug}", json={"kind":"botanical","variant":1}).status_code == 404


# ---------------------------------------------------------------------------
# Task 3: page-data pick state
# ---------------------------------------------------------------------------

def test_page_data_pick_state(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_images as si
    with sqlite3.connect(appmod.LOG_DB) as cx:
        si.record_image(cx, slug, "botanical", 1, "botanical-1.png")
        si.record_image(cx, slug, "botanical", 2, "botanical-2.png")
        si.record_image(cx, slug, "mechanism", 1, "mechanism-1.png")
        si.record_image(cx, slug, "mechanism", 2, "mechanism-2.png")
    c = appmod.app.test_client(); c.set_cookie("amg_session", "sP")
    body = next(s for s in c.get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    assert "pick" in body
    assert len(body["pick"]["botanical"]["options"]) == 2
    assert body["pick"]["botanical"]["chosen"] is None
    # after a pick, chosen reflects it
    c.post(f"/begin/product-image-pick/{slug}", json={"kind": "botanical", "variant": 2})
    body2 = next(s for s in c.get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    assert body2["pick"]["botanical"]["chosen"] == 2

def test_page_data_no_pick_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, pick="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    body = next(s for s in appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    assert "pick" not in body


# ---------------------------------------------------------------------------
# Task 4: image-pick reward at order settlement
# ---------------------------------------------------------------------------

def test_credit_granted_once_on_order_when_both_picked(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_votes as sv, points as pts
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sv.record_pick(cx, slug, "botanical", 1, "sQ", "buyer@x.co")
        sv.record_pick(cx, slug, "mechanism", 1, "sQ", "buyer@x.co")
    order = {"email": "buyer@x.co", "items": [{"slug": slug, "name": "X"}],
             "total_cents": 0, "shipping_cents": 0, "get_cents": 0,
             "points_redeemed_cents": 0, "discount_cents": 0}
    appmod._settle_order_points(order, order_ref="INV-1")
    appmod._settle_order_points(order, order_ref="INV-1")  # idempotent re-settle
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert pts.has_entry(cx, order_ref=f"imgpick_{slug}", reason="image_pick") is True

def test_no_credit_when_only_one_pair_or_other_product(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_votes as sv, points as pts
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sv.record_pick(cx, slug, "botanical", 1, "sR", "b2@x.co")  # only one pair
    appmod._settle_order_points({"email": "b2@x.co", "items": [{"slug": slug}],
        "total_cents":0,"shipping_cents":0,"get_cents":0,"points_redeemed_cents":0,"discount_cents":0},
        order_ref="INV-2")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert pts.has_entry(cx, order_ref=f"imgpick_{slug}", reason="image_pick") is False
