import sqlite3
from dashboard import fullscript as fs

SEED = {
  "products": [
    {"name": "Magnesium Taurate", "external_id": "U3ByZWU6OlByb2R1Y3QtMTA3Njc2",
     "product_slug": "magnesium-taurate", "brand": "Jarrow Formulas", "url": None,
     "focus_tags": ["Nervous System"], "product_type": "supplement",
     "best_ff": "Neuro Magnesium", "relation": "substitute", "ff_alts": [],
     "source": "seed", "active": 1},
    {"name": "Pure Taurine 500mg", "external_id": "U3ByZWU6OlByb2R1Y3QtNjc1NjE",
     "product_slug": "pure-taurine-500-mg-100-caps", "brand": "Montiff", "url": None,
     "focus_tags": [], "product_type": "supplement",
     "best_ff": None, "relation": None, "ff_alts": [], "source": "seed", "active": 1},
  ],
  "focus_area_products": [
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Magnesium Taurate", "rank": 0},
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Pure Taurine 500mg", "rank": 1},
  ],
  "focus_area_items": [
    {"focus_area_id": 9, "item_code": "ED4"},
    {"focus_area_id": 9, "item_code": "EI1"},
    {"focus_area_id": 14, "item_code": "ED8"},
  ],
}


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    return cx


def test_sync_counts_and_idempotent():
    cx = _cx()
    c = fs.sync_from_seed(cx, SEED)  # second run
    assert c["products"] == 2 and c["focus_area_products"] == 2
    assert cx.execute("SELECT COUNT(*) FROM fullscript_products").fetchone()[0] == 2


def test_all_seven_tables_exist():
    cx = _cx()
    names = {r[0] for r in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"fullscript_products", "fullscript_focus_area_products",
            "fullscript_focus_area_items", "fullscript_condition_products",
            "fullscript_client_pins", "fullscript_review_links",
            "fullscript_clicks"} <= names


def test_product_columns_roundtrip():
    cx = _cx()
    r = cx.execute("SELECT * FROM fullscript_products WHERE name=?",
                   ("Magnesium Taurate",)).fetchone()
    assert r["external_id"] == "U3ByZWU6OlByb2R1Y3QtMTA3Njc2"
    assert r["brand"] == "Jarrow Formulas"
    assert r["product_slug"] == "magnesium-taurate"
    assert r["best_ff"] == "Neuro Magnesium" and r["relation"] == "substitute"
    assert r["source"] == "seed" and r["active"] == 1
