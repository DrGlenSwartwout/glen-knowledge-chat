"""Task 5: GET /api/practitioner/terrain-restore-catalog -- all Terrain Restore
products (essences + tinctures + gemmotherapies), for the practitioner's Life
Stress substitution search. Signed-in practitioner only; not per-patient (it's
a catalog, not patient data).

Patterns borrowed from tests/test_life_stress_practitioner_read.py: app-import
+ _practitioner_session_pid override + LIFE_STRESS_ENABLED env, and gate-order
assertions (flag off -> 404, no pid -> 401).
"""
import importlib
import sys
from pathlib import Path

import pytest

PID = "doc1"


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture
def wired(monkeypatch):
    """Signed-in session for PID, flag ON."""
    app_module = _app()
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: PID)
    return app_module


def test_flag_off_404s(monkeypatch, wired):
    app_module = wired
    monkeypatch.delenv("LIFE_STRESS_ENABLED", raising=False)
    client = app_module.app.test_client()
    r = client.get("/api/practitioner/terrain-restore-catalog")
    assert r.status_code == 404


def test_requires_auth(monkeypatch, wired):
    app_module = wired
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: None)
    client = app_module.app.test_client()
    r = client.get("/api/practitioner/terrain-restore-catalog")
    assert r.status_code == 401
    assert r.get_json()["ok"] is False


def test_200_includes_known_terrain_restore_product(wired):
    app_module = wired
    client = app_module.app.test_client()
    r = client.get("/api/practitioner/terrain-restore-catalog")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    products = body["products"]
    assert len(products) > 0
    for p in products:
        assert "slug" in p
        assert "name" in p
    known = {"slug": "trauma-relief-in-terrain-restore",
             "name": "Trauma Relief in Terrain Restore"}
    assert known in products
