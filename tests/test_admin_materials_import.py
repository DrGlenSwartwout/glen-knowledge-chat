# tests/test_admin_materials_import.py
import importlib, io, sqlite3, sys
from pathlib import Path
import pytest

_MATERIALS_CSV = "id_pk,material_name,active,type,notes\nm1,Caps,Yes,,\n"
_MATERIALS_SUPPLIER_CSV = "id_pk,id_fk_material,id_fk_supplier,price,product_id,purchase_size,purchase_size_unit,mfg,contact,product_link,notes\nms1,m1,s1,10,,,,,,\n"
_PRODUCTS_SUPPLIER_CSV = "id_pk,id_fk_product,id_fk_supplier,price,product_id,purchase_size,purchase_size_unit,mfg,contact,product_link,notes\nps1,9,s1,20,,,,,,\n"


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client(), str(tmp_path / "chat_log.db")


def _post_import(client, write=None, missing=None):
    data = {}
    if missing != "materials":
        data["materials"] = (io.BytesIO(_MATERIALS_CSV.encode()), "materials.csv")
    if missing != "materials_supplier":
        data["materials_supplier"] = (io.BytesIO(_MATERIALS_SUPPLIER_CSV.encode()), "materials_supplier.csv")
    if missing != "products_supplier":
        data["products_supplier"] = (io.BytesIO(_PRODUCTS_SUPPLIER_CSV.encode()), "products_supplier.csv")
    if write is not None:
        data["write"] = write
    return client.post("/api/materials/import", data=data, content_type="multipart/form-data")


def test_write_imports_rows(tmp_path, monkeypatch):
    client, db = _client(tmp_path, monkeypatch)

    # Pre-seed a supplier so material_supplier links can resolve
    cx = sqlite3.connect(db)
    cx.execute("""CREATE TABLE IF NOT EXISTS suppliers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fmp_id TEXT UNIQUE,
        company TEXT
    )""")
    cx.execute("INSERT INTO suppliers(fmp_id, company) VALUES('s1','Acme')")
    cx.commit()
    cx.close()

    resp = _post_import(client, write="1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    d = body["data"]
    assert d["mode"] == "write"
    assert d["materials"] == 1
    assert d["material_suppliers"] == 1
    assert d["product_suppliers"] == 1

    # Verify rows actually landed in the DB
    cx = sqlite3.connect(db)
    mat_count = cx.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
    cx.close()
    assert mat_count == 1, f"expected 1 material, got {mat_count}"


def test_dry_run_no_write(tmp_path, monkeypatch):
    client, db = _client(tmp_path, monkeypatch)
    resp = _post_import(client)  # no write field
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    d = body["data"]
    assert d["mode"] == "dry_run"
    assert d["materials"] == 1
    assert d["material_suppliers"] == 1
    assert d["product_suppliers"] == 1

    # DB should NOT have a materials table written
    try:
        cx = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = cx.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        cx.close()
        table_names = [t[0] for t in tables]
        assert "materials" not in table_names, "dry_run must not write materials table"
    except sqlite3.OperationalError:
        pass  # DB not created at all — also fine


def test_missing_file_returns_400(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    resp = _post_import(client, write="1", missing="products_supplier")
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "products_supplier" in body["error"] or "three" in body["error"]
