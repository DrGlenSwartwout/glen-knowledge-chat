"""Import FileMaker purchase-order history (po + po_items + po_receiving) into
chat_log.db. Line items reference ingredients/materials/products. Idempotent by
fmp_id; curated-preserving. Dry-run default; --write."""
from __future__ import annotations
import argparse, csv, os, sqlite3, sys
csv.field_size_limit(sys.maxsize)
from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert

EXPORT_DIR = os.environ.get("FMP_EXPORT_DIR", "/tmp/fmp-export/newapp")


def import_purchase_orders(cx, rows):
    cx.row_factory = sqlite3.Row
    sup = {r["fmp_id"]: (r["id"], r["company"]) for r in cx.execute("SELECT id, fmp_id, company FROM suppliers WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = ["supplier_id", "supplier_name", "vendor_po_no", "po_date", "status", "tax",
                "shipping_amount", "shipper", "tracking_number", "due_date", "posted_date", "qb_id", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_supplier", "closed", "notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        s = sup.get((r.get("id_fk_supplier") or "").strip())
        sid, sname = (s[0], s[1]) if s else (None, None)
        status = "closed" if _active(r.get("closed")) == 1 else "open"
        vals = [fid, sid, sname, _clean(r.get("vendor_po_no")) or None, _clean(r.get("po_date")) or None, status,
                _num(r.get("tax")), _num(r.get("shipping_amount")), _clean(r.get("shipper")) or None,
                _clean(r.get("tracking_number")) or None, _clean(r.get("due_date")) or None,
                _clean(r.get("posted_date")) or None, _clean(r.get("qb_id")) or None, _extras(r, mapped)]
        _upsert(cx, "purchase_orders", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def import_po_items(cx, rows):
    cx.row_factory = sqlite3.Row
    po = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM purchase_orders WHERE fmp_id IS NOT NULL")}
    ing = {r["fmp_id"]: (r["id"], r["name"]) for r in cx.execute("SELECT id, fmp_id, name FROM ingredients WHERE fmp_id IS NOT NULL")}
    mat = {r["fmp_id"]: (r["id"], r["name"]) for r in cx.execute("SELECT id, fmp_id, name FROM materials WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = ["po_id", "item_kind", "item_label", "ingredient_id", "material_id", "fmp_product_id",
                "sku", "qty", "qty_unit", "qty_left", "cost", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_po", "id_fk_raw", "id_fk_material", "id_fk_product",
                              "product_id", "fee_name", "notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        po_id = po.get((r.get("id_fk_po") or "").strip())
        raw = (r.get("id_fk_raw") or "").strip()
        matf = (r.get("id_fk_material") or "").strip()
        prod = (r.get("id_fk_product") or "").strip()
        ing_id = ing.get(raw)
        mat_id = mat.get(matf)
        if ing_id:
            kind, label, iid, mid, pid = "ingredient", ing_id[1], ing_id[0], None, None
        elif mat_id:
            kind, label, iid, mid, pid = "material", mat_id[1], None, mat_id[0], None
        elif prod:
            kind, label, iid, mid, pid = "product", _clean(r.get("fee_name")) or None, None, None, prod
        else:
            kind, label, iid, mid, pid = ("fee" if _clean(r.get("fee_name")) else "other"), _clean(r.get("fee_name")) or None, None, None, None
        vals = [fid, po_id, kind, label, iid, mid, pid, _clean(r.get("product_id")) or None,
                _num(r.get("qty")), _clean(r.get("qty_unit")) or None, _num(r.get("qty_left")), _num(r.get("cost")), _extras(r, mapped)]
        _upsert(cx, "po_items", fmp_cols, vals, fmp_cols)
        n += 1
    return {"items": n}


def import_po_receiving(cx, rows):
    cx.row_factory = sqlite3.Row
    po = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM purchase_orders WHERE fmp_id IS NOT NULL")}
    item = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM po_items WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = ["po_id", "po_item_id", "qty_received", "received_size", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_po", "id_fk_po_item", "id_fk_material", "id_fk_product", "id_fk_raw"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        vals = [fid, po.get((r.get("id_fk_po") or "").strip()), item.get((r.get("id_fk_po_item") or "").strip()),
                _num(r.get("qty_received")), _clean(r.get("received_size")) or None, _extras(r, mapped)]
        _upsert(cx, "po_receiving", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def _read(name):
    with open(os.path.join(EXPORT_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    po = _read("po.csv"); items = _read("po_items.csv"); rec = _read("po_receiving.csv")
    print(f"po={len(po)} po_items={len(items)} po_receiving={len(rec)}")
    if not args.write:
        print("(dry run — pass --write)")
        return 0
    from dashboard.ingredient_catalog import _default_db_path, init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    cx = sqlite3.connect(args.db or _default_db_path()); cx.row_factory = sqlite3.Row
    init_ingredients_schema(cx); init_materials_schema(cx); init_purchase_orders_schema(cx)
    npo = import_purchase_orders(cx, po); ri = import_po_items(cx, items); nr = import_po_receiving(cx, rec)
    cx.commit(); cx.close()
    print(f"wrote purchase_orders={npo} po_items={ri['items']} po_receiving={nr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
