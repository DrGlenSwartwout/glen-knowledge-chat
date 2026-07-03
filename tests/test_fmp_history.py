import sqlite3
from dashboard import fmp_orders as fo
from dashboard import purchase_history as ph
from dashboard import fmp_history as fh


def _cx():
    cx = sqlite3.connect(":memory:")
    fo.ensure_tables(cx)
    ph.init_purchase_history_table(cx)
    return cx


def _seed(cx):
    cx.executemany(
        "INSERT INTO fmp_clients (id_pk,name_first,name_last,company,email,phone_res,phone_cell,phone_business) "
        "VALUES (?,?,?,?,?,?,?,?)", [
            ("c1", "Jo", "Cud", "SSO", "jo@x.com", "", "", ""),
            ("c2", "No", "Email", "", "", "", "", ""),  # blank email
        ])
    cx.executemany(
        "INSERT INTO fmp_invoices (id_pk,id_fk_client,invoice_date,status,subtotal,total,shipping,outstanding) "
        "VALUES (?,?,?,?,?,?,?,?)", [
            ("i1", "c1", "2026-04-01", "Closed", "100.00", "113.00", "13.00", "0.00"),
            ("i2", "c2", "2026-01-15", "Closed", "50.00", "50.00", "0.00", "0.00"),
        ])
    cx.executemany(
        "INSERT INTO fmp_invoice_items (id_pk,id_fk_invoice,id_fk_product,description,qty,price,ext_price) "
        "VALUES (?,?,?,?,?,?,?)", [
            ("it1", "i1", "10", "Neuro-Magnesium", "1", "40.00", "40.00"),   # resolved
            ("it2", "i1", "952", "Some Excluded Product", "1", "10.00", "10.00"),  # excluded
            ("it3", "i1", "999", "Unmapped Product", "1", "5.00", "5.00"),  # unmapped
            ("it4", "i2", "10", "Neuro-Magnesium", "1", "40.00", "40.00"),  # resolved but no email
        ])
    cx.commit()


_SLUG_MAP = {
    "resolved": {"10": "neuro-magnesium"},
    "review": {},
    "exclude": [952],
    "_generated_note": "fixture",
}


def test_rebuild_from_fmp_maps_resolved_skips_excluded_unmapped_and_noemail():
    cx = _cx()
    _seed(cx)
    result = fh.rebuild_from_fmp(cx, _SLUG_MAP)

    assert result == {"rows": 1, "skipped_excluded": 1, "skipped_unmapped": 1, "skipped_noemail": 1}

    rows = cx.execute(
        "SELECT email, slug, purchased_at, source, source_ref FROM purchase_history").fetchall()
    assert rows == [("jo@x.com", "neuro-magnesium", "2026-04-01", "fmp", "it1")]


def test_rebuild_from_fmp_is_idempotent_replace():
    cx = _cx()
    _seed(cx)
    fh.rebuild_from_fmp(cx, _SLUG_MAP)
    result2 = fh.rebuild_from_fmp(cx, _SLUG_MAP)
    assert result2["rows"] == 1
    rows = cx.execute("SELECT COUNT(*) FROM purchase_history WHERE source='fmp'").fetchall()
    assert rows[0][0] == 1
