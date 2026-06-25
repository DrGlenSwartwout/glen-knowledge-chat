# tests/test_import_production.py
import sqlite3
from dashboard import production as prod
from dashboard import inventory as inv
from scripts.import_production_from_fmp import import_production_runs, import_production_items


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT, extras TEXT)")
        cx.execute("CREATE TABLE materials (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
        cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
        inv.init_inventory_schema(cx)
        prod.init_production_schema(cx)
        cx.execute("INSERT INTO ingredients VALUES (1,'r1','Mag',NULL)")
        cx.execute("INSERT INTO materials VALUES (5,'m1','Capsule')")
        cx.execute("INSERT INTO formulations VALUES (1,'p1','Brain Blend')")
        cx.commit()
    return db


def test_import_runs_items_and_consume(tmp_path):
    db = _db(tmp_path)
    runs = [{"id_pk": "900", "id_fk_product": "p1", "production_date": "2026-03-01", "qty": "100", "label": "B7"}]
    items = [
        {"id_pk": "9000", "id_fk_production": "900", "id_fk_raw": "r1", "qty": "2.5", "unit_measurement": "kg"},
        {"id_pk": "9001", "id_fk_production": "900", "id_fk_material": "m1", "qty": "100", "unit_measurement": "ea"},
    ]
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        nr = import_production_runs(cx, runs)
        ni = import_production_items(cx, items)
        n_consumed = prod.post_consumption(cx, mode="all")
        cx.commit()
    assert nr == 1 and ni == {"items": 2}
    assert n_consumed == 1                            # only the ingredient line consumes; material does not
    assert inv.on_hand(1, db) == -2.5                 # no baseline here, just the consumption
    # run resolved to formulation, item to ingredient/material
    run = prod.search_production_runs(db_path=db)[0]
    assert run["formulation_name"] == "Brain Blend" and run["batch_number"] == "B7"
    its = prod.list_run_items(run["id"], db)
    kinds = sorted(i["item_type"] for i in its)
    assert kinds == ["ingredient", "material"]


def test_reimport_preserves_curated_and_idempotent(tmp_path):
    db = _db(tmp_path)
    runs = [{"id_pk": "900", "id_fk_product": "p1", "production_date": "2026-03-01", "qty": "100", "label": "B7"}]
    items = [{"id_pk": "9000", "id_fk_production": "900", "id_fk_raw": "r1", "qty": "2.5", "unit_measurement": "kg"}]
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        import_production_runs(cx, runs); import_production_items(cx, items)
        prod.post_consumption(cx, mode="all"); cx.commit()
    rid = prod.search_production_runs(db_path=db)[0]["id"]
    prod.update_run_curated(rid, {"notes": "keep me"}, db_path=db)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        import_production_runs(cx, runs); import_production_items(cx, items)
        n2 = prod.post_consumption(cx, mode="all"); cx.commit()
    assert n2 == 0                                    # consumption idempotent
    assert prod.get_production_run(rid, db)["notes"] == "keep me"   # curated preserved
    assert inv.on_hand(1, db) == -2.5


def test_iso_date_normalization_and_from_date_filter(tmp_path):
    """Imported run_date is normalized M/D/YYYY -> ISO so the from_date consumption
    cutoff (lexical >=) selects the right runs. Without this, '9/18/2025' >= '2026-01-01'
    is wrongly True."""
    from scripts.import_production_from_fmp import _iso_date
    assert _iso_date("9/18/2025") == "2025-09-18"
    assert _iso_date("1/5/2026") == "2026-01-05"
    assert _iso_date("") is None
    assert _iso_date("2026-01-05") == "2026-01-05"   # already ISO, unchanged

    db = _db(tmp_path)
    runs = [
        {"id_pk": "900", "id_fk_product": "p1", "production_date": "9/18/2025", "qty": "10", "label": "OLD"},
        {"id_pk": "901", "id_fk_product": "p1", "production_date": "1/15/2026", "qty": "10", "label": "NEW"},
    ]
    items = [
        {"id_pk": "9000", "id_fk_production": "900", "id_fk_raw": "r1", "qty": "2.0", "unit_measurement": "kg"},
        {"id_pk": "9001", "id_fk_production": "901", "id_fk_raw": "r1", "qty": "3.0", "unit_measurement": "kg"},
    ]
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        import_production_runs(cx, runs)
        import_production_items(cx, items)
        cx.commit()
    # stored as ISO
    dates = sorted(r["run_date"] for r in prod.search_production_runs(db_path=db))
    assert dates == ["2025-09-18", "2026-01-15"]
    # from_date 2026-01-01 posts ONLY the 2026 run's consumption (3.0), not the 2025 one
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        n = prod.post_consumption(cx, mode="from_date", cutoff_date="2026-01-01")
        cx.commit()
    assert n == 1
    assert inv.on_hand(1, db) == -3.0     # only the Jan-2026 run consumed; Sep-2025 excluded
