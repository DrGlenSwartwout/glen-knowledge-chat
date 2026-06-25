"""Import FileMaker formulation recipes into chat_log.db: formulations (FF products)
+ formulation_items (recipe lines referencing Phase-1 ingredients). Idempotent by
fmp_id; curated-preserving. Dry-run default; --write."""
from __future__ import annotations
import argparse, csv, os, re, sqlite3, sys
csv.field_size_limit(sys.maxsize)

from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert

EXPORT_DIR = os.environ.get("FMP_EXPORT_DIR", "/tmp/fmp-export/newapp")


def _name_after_dash(raw):
    # zc_raw_display like "100mg - R-Lipoic Acid" -> "R-Lipoic Acid"; "" if no name
    s = raw or ""
    return _clean(s.split(" - ", 1)[1]) if " - " in s else ""


def import_formulations(cx, product_rows):
    n = 0
    fmp_cols = ["name", "status", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "product_name", "type", "active",
                              "product_slug", "notes"}
    for r in product_rows:
        if (r.get("type") or "").strip() != "Functional Formulation":
            continue
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        name = _clean(r.get("product_name")) or f"(unnamed FMP formulation {fid})"
        status = "active" if _active(r.get("active")) == 1 else "inactive"
        _upsert(cx, "formulations", fmp_cols, [fid, name, status, _extras(r, mapped)], fmp_cols)
        n += 1
    return n


def import_formulation_items(cx, item_rows, ff_product_ids):
    ing = {r[1]: r[0] for r in cx.execute("SELECT id, fmp_id FROM ingredients WHERE fmp_id IS NOT NULL")}
    form = {r[1]: r[0] for r in cx.execute("SELECT id, fmp_id FROM formulations WHERE fmp_id IS NOT NULL")}
    n, unresolved = 0, 0
    fmp_cols = ["formulation_id", "ingredient_id", "ingredient_name", "dose", "dose_unit", "raw_text", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_product", "id_fk_raw", "id_fk_material",
                              "qty", "unit_measurement", "zc_mg", "zc_raw_display", "notes"}
    ov = {row[0]: set(json.loads(row[1] or "[]"))
          for row in cx.execute("SELECT fmp_id, overrides FROM formulation_items WHERE fmp_id IS NOT NULL")}
    for r in item_rows:
        pid = (r.get("id_fk_product") or "").strip()
        if pid not in ff_product_ids:
            continue
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        form_id = form.get(pid)
        raw_fk = (r.get("id_fk_raw") or "").strip()
        ing_id = ing.get(raw_fk)
        if raw_fk and ing_id is None:
            unresolved += 1
        mg = _num(r.get("zc_mg"))
        if mg and mg > 0:
            dose, unit = mg, "mg"
        else:
            dose, unit = _num(r.get("qty")), (_clean(r.get("unit_measurement")) or None)
        name = _name_after_dash(r.get("zc_raw_display")) or None
        vals = [fid, form_id, ing_id, name, dose, unit, _clean(r.get("zc_raw_display")) or None, _extras(r, mapped)]
        upd = [c for c in fmp_cols if c not in ov.get(fid, ())]
        _upsert(cx, "formulation_items", fmp_cols, vals, upd)
        n += 1
    return {"items": n, "unresolved": unresolved}


def _read(name):
    with open(os.path.join(EXPORT_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    products = _read("products.csv")
    items = _read("products_items.csv")
    ff_ids = {(p.get("id_pk") or "").strip() for p in products if (p.get("type") or "").strip() == "Functional Formulation"}
    print(f"FF formulations={len(ff_ids)} products_items={len(items)}")
    if not args.write:
        print("(dry run — pass --write to import)")
        return 0
    from dashboard.ingredient_catalog import _default_db_path, init_ingredients_schema
    from dashboard.formulations import init_formulations_schema
    cx = sqlite3.connect(args.db or _default_db_path()); cx.row_factory = sqlite3.Row
    init_ingredients_schema(cx); init_formulations_schema(cx)
    nf = import_formulations(cx, products)
    res = import_formulation_items(cx, items, ff_ids)
    cx.commit(); cx.close()
    print(f"wrote formulations={nf} items={res['items']} unresolved_ingredient_links={res['unresolved']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
