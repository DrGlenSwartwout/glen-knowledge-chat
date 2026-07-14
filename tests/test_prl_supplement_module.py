import sqlite3
from dashboard import prl_supplement as prl

SEED = {
  "products": [
    {"name": "NeuroVen", "external_id": "1", "url": "u1", "focus_tags": [],
     "product_type": "supplement", "best_ff": "Neuroprotect", "relation": "substitute", "ff_alts": []},
    {"name": "Tranquinol", "external_id": "2", "url": "u2", "focus_tags": [],
     "product_type": "supplement", "best_ff": "Sleep Syntropy", "relation": "substitute", "ff_alts": []},
  ],
  "focus_area_products": [
    {"focus_area_id": 9, "focus_area_name": "Nervous System", "prl_product_name": "NeuroVen", "rank": 0},
    {"focus_area_id": 9, "focus_area_name": "Nervous System", "prl_product_name": "Tranquinol", "rank": 1},
  ],
  "focus_area_items": [
    {"focus_area_id": 9, "item_code": "ED4"},
    {"focus_area_id": 9, "item_code": "EI1"},
    {"focus_area_id": 14, "item_code": "ED8"},
  ],
}

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    prl.init_tables(cx); prl.sync_from_seed(cx, SEED); return cx

def test_sync_counts_and_idempotent():
    cx = _cx()
    c = prl.sync_from_seed(cx, SEED)  # second run
    assert c["products"] == 2 and c["focus_area_products"] == 2
    assert cx.execute("SELECT COUNT(*) FROM prl_products").fetchone()[0] == 2  # no dupes

def test_focus_areas_for_items_ranked():
    cx = _cx()
    fas = prl.focus_areas_for_items(cx, ["ED4", "EI1", "ED8"])
    assert fas[0]["focus_area_id"] == 9 and fas[0]["hits"] == 2  # FA 9 ranks first
    assert any(f["focus_area_id"] == 14 for f in fas)

def test_products_for_focus_area_joined_and_ordered():
    cx = _cx()
    ps = prl.products_for_focus_area(cx, 9)
    assert [p["name"] for p in ps] == ["NeuroVen", "Tranquinol"]
    assert ps[0]["best_ff"] == "Neuroprotect" and ps[0]["relation"] == "substitute"

def test_mirror_roundtrip():
    cx = _cx()
    assert prl.mirror_for_scan(cx, "s1") is None
    cx.execute("INSERT INTO prl_scan_mirror(scan_id, payload, captured_at) VALUES(?,?,?)",
               ("s1", '{"patterns":[]}', "2026-07-13"))
    assert prl.mirror_for_scan(cx, "s1") == {"patterns": []}
