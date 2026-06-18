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

def test_product_page_data_shape(client):
    appmod = client[1]
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client()
    data = c.get(f"/begin/product-page-data/{slug}").get_json()
    ids = [s["id"] for s in data["sections"]]
    assert ids == ["intro", "description", "video", "ingredients",
                   "comparison", "research", "images", "cta"]
    assert next(s for s in data["sections"] if s["id"] == "intro")["default_open"] is True
    assert all(s["default_open"] is False for s in data["sections"] if s["id"] != "intro")
    assert data["cta_url"] == f"/begin/buy/{slug}"
    # comparison carries packaging + microplastics rows and the category excipient callout
    comp = next(s for s in data["sections"] if s["id"] == "comparison")["body"]
    rows = {r["label"] for r in comp["rows"]}
    assert "Packaging" in rows and "Microplastic exposure" in rows
    assert "stearates" in comp["excipient_callout"].lower()
    assert len(comp["columns"]) == 3  # ours + 2 anonymized archetypes

def test_match_to_product_page_roundtrip(client):
    appmod = client[1]
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._PRODUCTS["products"][slug]["name"]
    assert appmod._sales_page_url(name) == f"/begin/product/{slug}"
    c = appmod.app.test_client()
    assert c.get(f"/begin/product/{slug}").status_code == 200
    data = c.get(f"/begin/product-page-data/{slug}").get_json()
    assert data["cta_url"] == f"/begin/buy/{slug}"

def test_section_pref_accepts_new_section_ids(client):
    """POST /begin/section-pref must accept the new sales-page section ids."""
    appmod = client[1]
    c = appmod.app.test_client()
    c.set_cookie("amg_session", "testsession123")
    for sec in ("comparison", "description", "video", "images"):
        r = c.post("/begin/section-pref",
                   json={"section": sec},
                   headers={"Content-Type": "application/json"})
        assert r.status_code == 200, f"section '{sec}' should return 200, got {r.status_code}"
        data = r.get_json()
        assert data and data.get("ok"), f"section '{sec}' should return ok=true"
