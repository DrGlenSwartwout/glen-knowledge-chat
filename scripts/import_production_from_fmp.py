#!/usr/bin/env python3
"""Import FMP production + production_items → production_runs/items, then post consumption."""
import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.import_ingredients_from_fmp import _num, _clean, _extras, _upsert  # noqa: E402
from dashboard import production as prod  # noqa: E402
from dashboard import inventory as inv  # noqa: E402

csv.field_size_limit(sys.maxsize)


def import_production_runs(cx, rows) -> int:
    cx.row_factory = sqlite3.Row
    fmap = {r["fmp_id"]: r["id"] for r in cx.execute(
        "SELECT id, fmp_id FROM formulations WHERE fmp_id IS NOT NULL")}
    fmp_cols = ["formulation_id", "batch_number", "run_date", "quantity_units", "status", "source_kind", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_product", "production_date", "qty", "label", "notes"}
    n = 0
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        form_id = fmap.get((r.get("id_fk_product") or "").strip())
        vals = [fid, form_id, _clean(r.get("label")) or None, _clean(r.get("production_date")) or None,
                _num(r.get("qty")), "completed", "fmp", _extras(r, mapped)]
        _upsert(cx, "production_runs", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def import_production_items(cx, rows) -> dict:
    cx.row_factory = sqlite3.Row
    runmap = {r["fmp_id"]: r["id"] for r in cx.execute(
        "SELECT id, fmp_id FROM production_runs WHERE fmp_id IS NOT NULL")}
    ingmap = {r["fmp_id"]: (r["id"], r["name"]) for r in cx.execute(
        "SELECT id, fmp_id, name FROM ingredients WHERE fmp_id IS NOT NULL")}
    matmap = {r["fmp_id"]: (r["id"], r["name"]) for r in cx.execute(
        "SELECT id, fmp_id, name FROM materials WHERE fmp_id IS NOT NULL")}
    fmp_cols = ["production_run_id", "item_type", "ingredient_id", "material_id", "item_label",
                "qty_used", "unit", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_production", "id_fk_raw", "id_fk_material",
                              "qty", "unit_measurement", "notes"}
    n = 0
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        run_id = runmap.get((r.get("id_fk_production") or "").strip())
        raw = (r.get("id_fk_raw") or "").strip()
        mat = (r.get("id_fk_material") or "").strip()
        if raw and raw in ingmap:
            kind, iid, mid, label = "ingredient", ingmap[raw][0], None, ingmap[raw][1]
        elif mat and mat in matmap:
            kind, iid, mid, label = "material", None, matmap[mat][0], matmap[mat][1]
        else:
            kind, iid, mid, label = ("ingredient" if raw else "material" if mat else None), None, None, None
        vals = [fid, run_id, kind, iid, mid, label, _num(r.get("qty")),
                _clean(r.get("unit_measurement")) or None, _extras(r, mapped)]
        _upsert(cx, "production_run_items", fmp_cols, vals, fmp_cols)
        n += 1
    return {"items": n}


def _read(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _db_path():
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=_db_path())
    ap.add_argument("--dir", default="/tmp/fmp-export/newapp")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--consumption", choices=["all", "record_only"], default="all")
    ap.add_argument("--consumption-from", default=None, help="YYYY-MM-DD; only runs on/after post consumption")
    args = ap.parse_args()
    runs = _read(str(Path(args.dir) / "production.csv"))
    items = _read(str(Path(args.dir) / "production_items.csv"))
    cx = sqlite3.connect(args.db)
    cx.row_factory = sqlite3.Row
    try:
        inv.init_inventory_schema(cx)
        prod.init_production_schema(cx)
        nr = import_production_runs(cx, runs)
        ni = import_production_items(cx, items)
        mode = "from_date" if args.consumption_from else args.consumption
        nc = prod.post_consumption(cx, mode=mode, cutoff_date=args.consumption_from)
        if args.write:
            cx.commit()
            print(f"WROTE runs={nr} items={ni['items']} consumption={nc}")
        else:
            cx.rollback()
            print(f"DRY-RUN runs={nr} items={ni['items']} consumption={nc} (rolled back)")
    finally:
        cx.close()


if __name__ == "__main__":
    main()
