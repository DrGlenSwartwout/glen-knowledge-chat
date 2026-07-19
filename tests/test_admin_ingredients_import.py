# tests/test_admin_ingredients_import.py
import importlib, io, sqlite3, sys
from pathlib import Path
import pytest

_SUPPLIERS_CSV = "id_pk,company,active\ns1,Acme Botanicals,yes\n"
_INGREDIENTS_CSV = "id_pk,name_common,active\ni1,R-Lipoic Acid,yes\ni2,Alpha Lipoic Acid,yes\n"
_SOURCES_CSV = "id_pk,id_fk_raw,id_fk_supplier,product_id,price,purchase_size,purchase_size_unit,shipping\nsrc1,i1,s1,SKU-001,10.00,1,kg,2.00\n"


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
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
    if missing != "suppliers":
        data["suppliers"] = (io.BytesIO(_SUPPLIERS_CSV.encode()), "suppliers.csv")
    if missing != "ingredients":
        data["ingredients"] = (io.BytesIO(_INGREDIENTS_CSV.encode()), "ingredients.csv")
    if missing != "sources":
        data["sources"] = (io.BytesIO(_SOURCES_CSV.encode()), "sources.csv")
    if write is not None:
        data["write"] = write
    return client.post("/api/ingredients/import", data=data, content_type="multipart/form-data")


def test_write_imports_rows(tmp_path, monkeypatch):
    client, db = _client(tmp_path, monkeypatch)
    resp = _post_import(client, write="1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    d = body["data"]
    assert d["mode"] == "write"
    assert d["suppliers"] == 1
    assert d["ingredients"] == 2
    assert d["sources"] == 1
    assert isinstance(d["canonical_applied"], int)
    assert isinstance(d["canonical_skipped"], int)

    # Verify rows actually landed in the DB
    cx = sqlite3.connect(db)
    sup_count = cx.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
    ing_count = cx.execute("SELECT COUNT(*) FROM ingredients").fetchone()[0]
    src_count = cx.execute("SELECT COUNT(*) FROM ingredient_sources").fetchone()[0]
    cx.close()
    assert sup_count == 1, f"expected 1 supplier, got {sup_count}"
    assert ing_count == 2, f"expected 2 ingredients, got {ing_count}"
    assert src_count == 1, f"expected 1 source, got {src_count}"


def test_dry_run_no_write(tmp_path, monkeypatch):
    client, db = _client(tmp_path, monkeypatch)
    resp = _post_import(client)  # no write field
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    d = body["data"]
    assert d["mode"] == "dry_run"
    assert d["suppliers"] == 1
    assert d["ingredients"] == 2
    assert d["sources"] == 1
    assert isinstance(d["clusters"], int)

    # DB should NOT have any tables written (file may not even exist)
    try:
        cx = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = cx.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        cx.close()
        table_names = [t[0] for t in tables]
        assert "ingredients" not in table_names, "dry_run must not write ingredients table"
    except sqlite3.OperationalError:
        pass  # DB not created at all — also fine


def test_missing_file_returns_400(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    resp = _post_import(client, write="1", missing="sources")
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "sources" in body["error"] or "three" in body["error"]
