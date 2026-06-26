import csv, json, sqlite3
from pathlib import Path
import importlib.util


def _load():
    p = Path(__file__).resolve().parent.parent / "scripts" / "import_invoices_from_fmp.py"
    spec = importlib.util.spec_from_file_location("imp_inv", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_run_import_dry_then_write(tmp_path):
    items = tmp_path / "invoice_items.csv"
    with open(items, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id_fk_product", "qty", "zc_ext_price", "zc_year", "zc_month", "invoice_date", "description", "fee_name"])
        w.writeheader()
        w.writerow({"id_fk_product": "425", "qty": "2", "zc_ext_price": "138", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/3/2026", "description": "Microbiome", "fee_name": ""})
        w.writerow({"id_fk_product": "", "qty": "1", "zc_ext_price": "10", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/3/2026", "description": "Shipping", "fee_name": "Shipping"})
    pj = tmp_path / "products.json"
    pj.write_text(json.dumps({"products": {"microbiome": {"fmp_id": "425"}}}))
    db = tmp_path / "chat_log.db"
    mod = _load()
    dry = mod.run_import(str(items), str(pj), str(db), write=False)
    assert dry["product_rows"] == 1 and dry["written"] == 0
    res = mod.run_import(str(items), str(pj), str(db), write=True)
    assert res["written"] == 1
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT product_slug, units, revenue_cents FROM product_sales").fetchone()
    assert row == ("microbiome", 2.0, 13800)
