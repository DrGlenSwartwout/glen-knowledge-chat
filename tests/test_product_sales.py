import json, sqlite3
import pytest
from dashboard import product_sales as ps


def _rows():
    return [
        {"id_fk_product": "425", "qty": "2", "zc_ext_price": "$138.00", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/3/2026", "description": "Microbiome", "fee_name": ""},
        {"id_fk_product": "425", "qty": "1", "zc_ext_price": "69", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/9/2026", "description": "Microbiome", "fee_name": ""},
        {"id_fk_product": "425", "qty": "3", "zc_ext_price": "207", "zc_year": "2025", "zc_month": "12", "invoice_date": "12/1/2025", "description": "Microbiome", "fee_name": ""},
        {"id_fk_product": "", "qty": "1", "zc_ext_price": "10", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/3/2026", "description": "Shipping", "fee_name": "Shipping"},  # fee → skip
        {"id_fk_product": "73", "qty": "1", "zc_ext_price": "90", "zc_year": "", "zc_month": "", "invoice_date": "2026-06-15", "description": "Nous Energy", "fee_name": ""},  # period from invoice_date
    ]


def test_aggregate_groups_skips_fees_and_converts():
    agg = ps.aggregate_rows(_rows(), {"425": "microbiome"})
    by = {(r["product_fmp_id"], r["period"]): r for r in agg}
    assert ("", "2026-06") not in by  # fee line skipped
    m = by[("425", "2026-06")]
    assert m["units"] == 3 and m["revenue_cents"] == 20700 and m["product_slug"] == "microbiome" and m["product_name"] == "Microbiome"
    assert by[("425", "2025-12")]["revenue_cents"] == 20700
    assert by[("73", "2026-06")]["units"] == 1 and by[("73", "2026-06")]["product_slug"] is None  # date fallback + unmatched slug


def test_write_idempotent_and_top_products():
    cx = sqlite3.connect(":memory:")
    ps.init_product_sales_table(cx)
    agg = ps.aggregate_rows(_rows(), {"425": "microbiome"})
    n1 = ps.write_fmp_sales(cx, agg)
    n2 = ps.write_fmp_sales(cx, agg)  # re-import
    assert n1 == n2  # idempotent: same row count, no duplicates
    assert cx.execute("SELECT COUNT(*) FROM product_sales").fetchone()[0] == n1
    top = ps.top_products(cx, year=2026, by="revenue", limit=10)
    assert top[0]["product_fmp_id"] == "425" and top[0]["revenue_cents"] == 20700  # 2026 only
    tu = ps.top_products(cx, year=None, by="units", limit=10)
    assert tu[0]["product_fmp_id"] == "425" and tu[0]["units"] == 6  # all-time units
