import json, sqlite3
import app as app_module


def test_endpoint_lazy_ingests_and_returns_recommendations(monkeypatch, tmp_path):
    # Point LOG_DB at a temp db seeded with a paid order, verify the endpoint ingests it.
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT, "
               "external_ref TEXT, email TEXT, name TEXT, items_json TEXT, address_json TEXT DEFAULT '{}', "
               "total_cents INTEGER, status TEXT, pay_status TEXT, paid_at TEXT)")
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_at, status) VALUES (?,?,?,?,?)",
               ("a@b.com", json.dumps([{"slug": "neuro-magnesium"}]), "paid", "2026-07-10", "done"))
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_bos_actor", lambda: object())
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.get("/api/console/client-360?email=a@b.com")
    assert r.status_code == 200
    data = r.get_json()
    recs = data["recommendations"]
    assert any(p["product_key"] == "neuro-magnesium"
               and any(s["source"] == "purchased" for s in p["sources"]) for p in recs)
