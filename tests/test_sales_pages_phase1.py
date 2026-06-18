import importlib, os
import pytest

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config["TESTING"] = True
    return appmod.app, appmod

def test_product_url_built_when_flag_on(client):
    _, appmod = client
    # pick any real slug from the loaded catalog
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._PRODUCTS["products"][slug]["name"]
    assert appmod._sales_page_url(name) == f"/begin/product/{slug}"

def test_product_url_empty_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SALES_PAGES_ENABLED", "false")
    import importlib, app as appmod
    importlib.reload(appmod)
    name = next(iter(appmod._PRODUCTS["products"].values()))["name"]
    assert appmod._sales_page_url(name) == ""

def test_product_page_200_known_slug(client):
    appmod = client[1]
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client()
    r = c.get(f"/begin/product/{slug}")
    assert r.status_code == 200
    assert b"begin-product" in r.data or b"<!DOCTYPE html" in r.data

def test_product_page_404_unknown_slug(client):
    appmod = client[1]
    c = appmod.app.test_client()
    assert c.get("/begin/product/nope-not-real").status_code == 404
