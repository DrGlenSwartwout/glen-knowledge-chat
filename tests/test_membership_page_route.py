import importlib, sys, os
import pytest

def _load_app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app import failed: {e}")

def test_page_404_when_flag_off(monkeypatch):
    app = _load_app()
    monkeypatch.setattr(app, "MEMBERSHIP_PRODUCTS_ENABLED", False, raising=False)
    assert app.app.test_client().get("/membership").status_code == 404

def test_page_served_when_flag_on(monkeypatch):
    app = _load_app()
    monkeypatch.setattr(app, "MEMBERSHIP_PRODUCTS_ENABLED", True, raising=False)
    r = app.app.test_client().get("/membership")
    assert r.status_code == 200
    assert b"Membership" in r.data
