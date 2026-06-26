import sqlite3, pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    import dashboard as _dashboard
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(_dashboard, "CONSOLE_SECRET", "test-secret")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("CREATE TABLE product_sales (product_fmp_id TEXT, period TEXT, units REAL, revenue_cents INTEGER, source TEXT)")
        cx.executemany("INSERT INTO product_sales VALUES (?,?,?,?, 'fmp')",
                       [("425", "2026-06", 12, 0), ("425", "2026-05", 6, 0)])
        cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
        cx.execute("INSERT INTO formulations(id,fmp_id,name) VALUES (1,'425','Microbiome')")
        cx.execute("CREATE TABLE formulation_items (formulation_id INTEGER, ingredient_id INTEGER, dose REAL, dose_unit TEXT)")
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, name TEXT, extras TEXT, par_level REAL, par_level_unit TEXT)")
        # tables required by reorder_report / on_order_by_ingredient / on_hand
        cx.execute("CREATE TABLE purchase_orders (id INTEGER PRIMARY KEY, status TEXT)")
        cx.execute("CREATE TABLE po_items (id INTEGER PRIMARY KEY, po_id INTEGER, ingredient_id INTEGER, qty REAL)")
        cx.execute("CREATE TABLE po_receiving (id INTEGER PRIMARY KEY, po_item_id INTEGER, qty_received REAL)")
        cx.execute("CREATE TABLE inventory_txns (id INTEGER PRIMARY KEY, ingredient_id INTEGER, qty REAL, txn_type TEXT)")
        cx.execute("CREATE TABLE ingredient_sources (id INTEGER PRIMARY KEY, ingredient_id INTEGER, supplier_id INTEGER, supplier_name TEXT, price_per_unit REAL, unit_size REAL, unit_type TEXT, minimum_order REAL, minimum_order_unit TEXT, preferred INTEGER DEFAULT 0)")
        cx.execute("CREATE TABLE suppliers (id INTEGER PRIMARY KEY, company TEXT)")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_velocity_source_returns_table(client):
    r = client.get("/api/reorder/report?source=velocity&basis=3mo&horizon=3", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200
    body = r.get_json()
    vt = body.get("velocity_table") or (body.get("data") or {}).get("velocity_table")
    assert vt and vt[0]["fmp_id"] == "425" and vt[0]["projected_qty"] == 6.0 * 3


def test_reorder_report_requires_auth(client):
    assert client.get("/api/reorder/report?source=velocity").status_code in (401, 403)
