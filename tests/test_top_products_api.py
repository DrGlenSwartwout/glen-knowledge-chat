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


def test_sales_import_form_key_dry_then_write(client):
    import io
    csv_bytes = (b"id_fk_product,qty,zc_ext_price,zc_year,zc_month,invoice_date,description,fee_name\n"
                 b"425,2,138,2026,6,6/3/2026,Microbiome,\n"
                 b",1,10,2026,6,6/3/2026,Shipping,Shipping\n")
    def post(extra):
        data = {"invoice_items": (io.BytesIO(csv_bytes), "invoice_items.csv")}
        data.update(extra)
        return client.post("/api/console/sales/import", data=data, content_type="multipart/form-data")
    assert post({}).status_code == 401                      # no key
    j1 = post({"key": "test-secret"}).get_json()            # form key, dry-run
    assert j1["product_rows"] == 1 and j1["written"] == 0
    j2 = post({"key": "test-secret", "write": "1"}).get_json()
    assert j2["written"] == 1                                # form key accepted + write
