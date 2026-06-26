#!/usr/bin/env python3
"""Import FMP invoice_items.csv into the product_sales table (idempotent).
Usage: python3 scripts/import_invoices_from_fmp.py --items /tmp/fmp-export/newapp/invoice_items.csv \
         --products data/products.json --db chat_log.db [--write]"""
import argparse, csv, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
csv.field_size_limit(sys.maxsize)
from dashboard import product_sales as ps


def run_import(items_csv, products_json, db_path, write=False):
    rows = list(csv.DictReader(open(items_csv)))
    slug_for = ps.slug_map_from_products_json(products_json)
    agg = ps.aggregate_rows(rows, slug_for)
    out = {"line_items": len(rows), "product_rows": len(agg), "written": 0}
    if write:
        with sqlite3.connect(db_path) as cx:
            ps.init_product_sales_table(cx)
            out["written"] = ps.write_fmp_sales(cx, agg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="/tmp/fmp-export/newapp/invoice_items.csv")
    ap.add_argument("--products", default="data/products.json")
    ap.add_argument("--db", default="chat_log.db")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    res = run_import(a.items, a.products, a.db, write=a.write)
    print(res)


if __name__ == "__main__":
    main()
