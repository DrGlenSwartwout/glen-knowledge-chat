"""Endpoint-level tests for the /api/shipping/* console API extensions.

Tests the dims-aware PATCH, packing-settings GET/POST, and product-bottles
GET/POST/DELETE endpoints added in Task 6.

Uses DATA_DIR monkeypatching so the shipping module uses a temp db.
"""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest


def _client(tmp_path, monkeypatch):
    """Build a Flask test client backed by a fresh temp db."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.shipping import init_shipping_schema
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)  # no auth in test
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    # Reload so _shipping picks up DATA_DIR and app re-reads CONSOLE_SECRET
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable in this env: {e}")
    return appmod.app.test_client()


def test_set_bottle_dims_via_patch(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    bottles = c.get("/api/shipping/bottles").get_json()["data"]
    bid = next(b["id"] for b in bottles if b["name"] == "30ml")
    r = c.patch(f"/api/shipping/bottles/{bid}",
                json={"name": "30ml", "diameter_mm": 41, "height_mm": 112})
    assert r.status_code == 200
    from dashboard.shipping import get_bottle_dims
    assert get_bottle_dims(db_path=str(tmp_path / "chat_log.db"))["30ml"] == (41, 112)


def test_list_bottles_includes_dims(tmp_path, monkeypatch):
    """list_bottle_types now returns diameter_mm / height_mm fields."""
    c = _client(tmp_path, monkeypatch)
    bottles = c.get("/api/shipping/bottles").get_json()["data"]
    b30 = next(b for b in bottles if b["name"] == "30ml")
    assert "diameter_mm" in b30
    assert "height_mm" in b30
    # Seeded value from _STANDARD_BOTTLES: 30ml → (40, 110)
    assert b30["diameter_mm"] == 40
    assert b30["height_mm"] == 110


def test_add_bottle_with_dims(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/shipping/bottles",
               json={"name": "test-bottle", "diameter_mm": 35, "height_mm": 95})
    assert r.status_code == 200
    from dashboard.shipping import get_bottle_dims
    dims = get_bottle_dims(db_path=str(tmp_path / "chat_log.db"))
    assert dims["test-bottle"] == (35, 95)


def test_packing_settings_get_and_post(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/shipping/packing-settings").get_json()["data"]["wrap_mm"] == 6
    assert c.post("/api/shipping/packing-settings", json={"wrap_mm": 8}).status_code == 200
    assert c.get("/api/shipping/packing-settings").get_json()["data"]["wrap_mm"] == 8


def test_packing_settings_preserves_other_key(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/api/shipping/packing-settings", json={"box_margin_mm": 12})
    data = c.get("/api/shipping/packing-settings").get_json()["data"]
    assert data["box_margin_mm"] == 12
    assert data["wrap_mm"] == 6  # default unchanged


def test_packing_settings_bad_value(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/shipping/packing-settings", json={"wrap_mm": "not-a-number"})
    assert r.status_code == 400


def test_product_bottle_override_endpoints(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/api/shipping/product-bottles",
                  json={"slug": "foo", "bottle_type": "30ml"}).status_code == 200
    listing = c.get("/api/shipping/product-bottles").get_json()["data"]
    assert listing["overrides"]["foo"] == "30ml"
    assert c.delete("/api/shipping/product-bottles/foo").status_code == 200


def test_product_bottle_override_cleared(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/api/shipping/product-bottles",
           json={"slug": "bar", "bottle_type": "100ml"})
    c.delete("/api/shipping/product-bottles/bar")
    listing = c.get("/api/shipping/product-bottles").get_json()["data"]
    assert "bar" not in listing["overrides"]


def test_product_bottle_missing_fields(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/shipping/product-bottles", json={"slug": "foo"})
    assert r.status_code == 400


def test_product_bottles_get_items_shape(tmp_path, monkeypatch):
    """GET /api/shipping/product-bottles returns items list + overrides dict."""
    c = _client(tmp_path, monkeypatch)
    data = c.get("/api/shipping/product-bottles").get_json()["data"]
    assert "items" in data
    assert "overrides" in data
    assert isinstance(data["items"], list)
    assert isinstance(data["overrides"], dict)
