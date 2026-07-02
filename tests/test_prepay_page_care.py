# tests/test_prepay_page_care.py
"""GET /prepay carries window.__CARE__ = {monthly_enabled: ...} so the picker can
offer the monthly-vs-upfront choice on the 6/12mo commitment tiers.

Mirrors the app-load pattern in test_prepay_checkout.py.
"""
import importlib, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_prepay_page_care_flag_true(monkeypatch):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "CONTINUOUS_CARE_MONTHLY_ENABLED", True, raising=False)
    r = app_module.app.test_client().get("/prepay")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "window.__CARE__" in body
    assert '"monthly_enabled": true' in body


def test_prepay_page_care_flag_false(monkeypatch):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "CONTINUOUS_CARE_MONTHLY_ENABLED", False, raising=False)
    r = app_module.app.test_client().get("/prepay")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "window.__CARE__" in body
    assert '"monthly_enabled": false' in body


def test_prepay_page_redirects_when_ladder_off(monkeypatch):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", False, raising=False)
    monkeypatch.setattr(app_module, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    r = app_module.app.test_client().get("/prepay", follow_redirects=False)
    assert r.status_code in (301, 302)


def test_prepay_page_no_store_headers(monkeypatch):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "CONTINUOUS_CARE_MONTHLY_ENABLED", True, raising=False)
    r = app_module.app.test_client().get("/prepay")
    cc = r.headers.get("Cache-Control", "")
    assert "no-store" in cc
