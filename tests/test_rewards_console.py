import sqlite3, pytest

@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        coupons.init_coupons_table(cx)
        coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

def test_console_lists_coupons_authed(client):
    r = client.get("/api/console/coupons", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200 and len(r.get_json()["coupons"]) == 1

def test_console_coupons_requires_auth(client):
    assert client.get("/api/console/coupons").status_code in (401, 403)
