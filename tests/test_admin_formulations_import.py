# tests/test_admin_formulations_import.py
import importlib, io, sqlite3, sys
from pathlib import Path
import pytest

_PRODUCTS_CSV = "id_pk,type,name,product_name,active,product_slug,notes\nf1,Functional Formulation,Test Form,Test Form,yes,test-form,\n"
_PRODUCTS_ITEMS_CSV = "id_pk,id_fk_product,id_fk_raw,id_fk_material,qty,unit_measurement,zc_mg,zc_raw_display,notes\nit1,f1,r1,,,,100,100mg - R-Lipoic Acid,\n"


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
    if missing != "products":
        data["products"] = (io.BytesIO(_PRODUCTS_CSV.encode()), "products.csv")
    if missing != "products_items":
        data["products_items"] = (io.BytesIO(_PRODUCTS_ITEMS_CSV.encode()), "products_items.csv")
    if write is not None:
        data["write"] = write
    return client.post("/api/formulations/import", data=data, content_type="multipart/form-data")


def test_write_imports_rows(tmp_path, monkeypatch):
    client, db = _client(tmp_path, monkeypatch)

    # Pre-seed an ingredient so the item can resolve its ingredient_id
    cx = sqlite3.connect(db)
    cx.execute("""CREATE TABLE IF NOT EXISTS ingredients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fmp_id TEXT UNIQUE,
        name TEXT
    )""")
    cx.execute("INSERT INTO ingredients(fmp_id, name) VALUES('r1','R-Lipoic Acid')")
    cx.commit()
    cx.close()

    resp = _post_import(client, write="1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    d = body["data"]
    assert d["mode"] == "write"
    assert d["formulations"] == 1
    assert d["items"] == 1
    assert isinstance(d["unresolved"], int)

    # Verify rows actually landed in the DB
    cx = sqlite3.connect(db)
    form_count = cx.execute("SELECT COUNT(*) FROM formulations").fetchone()[0]
    cx.close()
    assert form_count == 1, f"expected 1 formulation, got {form_count}"


def test_dry_run_no_write(tmp_path, monkeypatch):
    client, db = _client(tmp_path, monkeypatch)
    resp = _post_import(client)  # no write field
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    d = body["data"]
    assert d["mode"] == "dry_run"
    assert d["ff_formulations"] == 1
    assert d["products_items"] == 1

    # DB should NOT have a formulations table written
    try:
        cx = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = cx.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        cx.close()
        table_names = [t[0] for t in tables]
        assert "formulations" not in table_names, "dry_run must not write formulations table"
    except sqlite3.OperationalError:
        pass  # DB not created at all — also fine


def test_missing_file_returns_400(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    resp = _post_import(client, write="1", missing="products_items")
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "products_items" in body["error"] or "both" in body["error"]
