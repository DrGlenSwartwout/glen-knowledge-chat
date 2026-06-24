"""Load FileMaker CSV exports into chat_log.db snapshot tables (fmp_snap_*).
A local, PII-bearing reference snapshot so the report tool needs no live Supabase."""
import sqlite3
from dashboard.biofield_fmp_snapshot import snapshot_csv_dir


def _write(p, name, text):
    f = p / f"{name}.csv"
    f.write_text(text, encoding="utf-8")
    return f


def test_loads_csv_into_prefixed_table(tmp_path):
    src = tmp_path / "export"
    src.mkdir()
    _write(src, "client_remedy", "id,remedy,timing\n1,Sterol Max,with food\n2,TMG,at night\n")
    db = str(tmp_path / "chat_log.db")

    counts = snapshot_csv_dir(src, db)

    assert counts["client_remedy"] == 2
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute("SELECT * FROM fmp_snap_client_remedy ORDER BY id").fetchall()
    assert [dict(r) for r in rows] == [
        {"id": "1", "remedy": "Sterol Max", "timing": "with food"},
        {"id": "2", "remedy": "TMG", "timing": "at night"},
    ]


def test_reload_is_idempotent_replace(tmp_path):
    src = tmp_path / "export"
    src.mkdir()
    _write(src, "products", "id,product_name\n1,A\n")
    db = str(tmp_path / "chat_log.db")
    snapshot_csv_dir(src, db)
    # second load with different content fully replaces (no stale rows)
    _write(src, "products", "id,product_name\n9,Z\n")
    counts = snapshot_csv_dir(src, db)
    assert counts["products"] == 1
    with sqlite3.connect(db) as cx:
        rows = cx.execute("SELECT id FROM fmp_snap_products").fetchall()
    assert rows == [("9",)]
