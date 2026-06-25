"""Import FileMaker materials + material/product supplier links into chat_log.db.
Idempotent by fmp_id; curated-preserving. Dry-run default; --write."""
from __future__ import annotations
import argparse, csv, os, sqlite3, sys
csv.field_size_limit(sys.maxsize)
from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert

EXPORT_DIR = os.environ.get("FMP_EXPORT_DIR", "/tmp/fmp-export/newapp")


def import_materials(cx, rows):
    n = 0
    fmp_cols = ["name", "type", "status", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "material_name", "active", "notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        name = _clean(r.get("material_name")) or f"(unnamed FMP material {fid})"
        status = "active" if _active(r.get("active")) == 1 else "inactive"
        _upsert(cx, "materials", fmp_cols, [fid, name, _clean(r.get("type")) or None, status, _extras(r, mapped)], fmp_cols)
        n += 1
    return n


def _import_supplier_links(cx, rows, table, link_col, link_resolver):
    sup = {r["fmp_id"]: (r["id"], r["company"]) for r in cx.execute("SELECT id, fmp_id, company FROM suppliers WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = [link_col, "supplier_id", "supplier_name", "sku", "price", "purchase_size", "purchase_size_unit", "mfg", "contact", "product_link", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_material", "id_fk_product", "id_fk_supplier", "product_id", "active", "preferred", "notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        link_val = link_resolver(r)
        s = sup.get((r.get("id_fk_supplier") or "").strip())
        sid, sname = (s[0], s[1]) if s else (None, None)
        vals = [fid, link_val, sid, sname, _clean(r.get("product_id")) or None, _num(r.get("price")),
                _num(r.get("purchase_size")), _clean(r.get("purchase_size_unit")) or None, _clean(r.get("mfg")) or None,
                _clean(r.get("contact")) or None, _clean(r.get("product_link")) or None, _extras(r, mapped)]
        _upsert(cx, table, fmp_cols, vals, fmp_cols)
        n += 1
    return n


def import_material_suppliers(cx, rows):
    mat = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM materials WHERE fmp_id IS NOT NULL")}
    return _import_supplier_links(cx, rows, "material_suppliers", "material_id",
                                  lambda r: mat.get((r.get("id_fk_material") or "").strip()))


def import_product_suppliers(cx, rows):
    return _import_supplier_links(cx, rows, "product_suppliers", "fmp_product_id",
                                  lambda r: (r.get("id_fk_product") or "").strip() or None)


def _read(name):
    with open(os.path.join(EXPORT_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    materials = _read("materials.csv"); msup = _read("materials_supplier.csv"); psup = _read("products_supplier.csv")
    print(f"materials={len(materials)} material_suppliers={len(msup)} product_suppliers={len(psup)}")
    if not args.write:
        print("(dry run — pass --write)")
        return 0
    from dashboard.ingredient_catalog import _default_db_path, init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    cx = sqlite3.connect(args.db or _default_db_path()); cx.row_factory = sqlite3.Row
    init_ingredients_schema(cx); init_materials_schema(cx)
    nm = import_materials(cx, materials); nms = import_material_suppliers(cx, msup); nps = import_product_suppliers(cx, psup)
    cx.commit(); cx.close()
    print(f"wrote materials={nm} material_suppliers={nms} product_suppliers={nps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
