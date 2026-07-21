import app as app_module


def test_console_client_page_served():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.get("/console/client")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "client-360" in body            # the page fetches the bundle endpoint
    assert "op-nav.js" in body
    assert "no-store" in r.headers.get("Cache-Control", "")
