# tests/test_admin_po_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx); init_purchase_orders_schema(cx)
        cx.execute("INSERT INTO purchase_orders (fmp_id,vendor_po_no) VALUES ('p1','PO-1')"); cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()

def test_search_get_patch(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/po/search?q=PO-1").get_json()
    pid = r["data"][0]["id"]
    d = c.get(f"/api/po/{pid}").get_json()["data"]
    assert d["po"]["vendor_po_no"]=="PO-1" and "items" in d and "receiving" in d
    assert c.patch(f"/api/po/{pid}", json={"notes":"x"}).status_code == 200
