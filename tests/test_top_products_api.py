import sqlite3, pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    from dashboard import product_sales as ps
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ps.init_product_sales_table(cx)
        ps.write_fmp_sales(cx, [
            {"product_fmp_id": "425", "product_slug": "microbiome", "product_name": "Microbiome",
             "period": "2026-06", "units": 63, "revenue_cents": 434000, "source": "fmp"},
            {"product_fmp_id": "73", "product_slug": None, "product_name": "Nous Energy",
             "period": "2025-06", "units": 10, "revenue_cents": 52000, "source": "fmp"},
        ])
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_top_products_authed_year_filter(client):
    r = client.get("/api/console/top-products?year=2026&limit=5", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200
    items = r.get_json()["products"]
    assert items and items[0]["product_fmp_id"] == "425" and len(items) == 1  # 2026 only


def test_top_products_requires_auth(client):
    assert client.get("/api/console/top-products").status_code in (401, 403)
