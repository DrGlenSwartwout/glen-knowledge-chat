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
