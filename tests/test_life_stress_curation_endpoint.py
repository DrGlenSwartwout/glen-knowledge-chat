"""Task 4: POST /api/practitioner/life-stress-curation/<patient_email> — the
practitioner's curated essence list (the prescription that replaces the
auto-pool everywhere the client sees it), plus the Phase 1 GET
(/api/practitioner/life-stress-selection/<patient_email>) extended to return
the current curation so the composer/editor can pre-fill.

Patterns borrowed from tests/test_life_stress_practitioner_read.py: app-import
+ LOG_DB monkeypatch + _practitioner_session_pid override + monkeypatched
continuity_view.authorized_patient, and gate-order assertions
(flag off -> 404, no pid -> 401, not authorized -> 403), plus a fixed
`app._PRODUCTS` so slug resolution is deterministic without a real
products.json lookup.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import life_stress_curation

PID = "doc1"
EMAIL = "pat@x.com"

PRODUCTS = {
    "products": {
        "grief-release": {"name": "Grief Release"},
        "calm-clarity": {"name": "Calm & Clarity"},
    }
}

POOL_BLOCK = {
    "items": [
        {"slug": "grief-release", "name": "Grief Release", "note": "grief pattern"},
        {"slug": "calm-clarity", "name": "Calm & Clarity", "note": "anxiety pattern"},
    ]
}


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Wires up a real sqlite file at app.LOG_DB, a signed-in session for PID,
    an always-authorized continuity_view.authorized_patient, a fixed products
    map (`_PRODUCTS`) + Life Stress pool (`_life_stress_for`), and the flag ON."""
    app_module = _app()
    db_path = tmp_path / "chat_log.db"
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        life_stress_curation.init_table(cx)

    monkeypatch.setattr(app_module, "LOG_DB", db_path)
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: PID)
    from dashboard import continuity_view as cv
    monkeypatch.setattr(cv, "authorized_patient", lambda cx, pid, email: True)
    monkeypatch.setattr(app_module, "_PRODUCTS", PRODUCTS)
    monkeypatch.setattr(app_module, "_life_stress_for", lambda email: dict(POOL_BLOCK))
    return app_module, db_path


def _post(client, email, payload):
    return client.post(f"/api/practitioner/life-stress-curation/{email}", json=payload)


def test_flag_off_404s(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.delenv("LIFE_STRESS_ENABLED", raising=False)
    client = app_module.app.test_client()
    r = _post(client, EMAIL, {"slugs": ["grief-release"]})
    assert r.status_code == 404


def test_requires_auth(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: None)
    client = app_module.app.test_client()
    r = _post(client, EMAIL, {"slugs": ["grief-release"]})
    assert r.status_code == 401
    assert r.get_json()["ok"] is False


def test_403_when_not_authorized(monkeypatch, wired):
    app_module, _db = wired
    from dashboard import continuity_view as cv
    monkeypatch.setattr(cv, "authorized_patient", lambda cx, pid, email: False)
    client = app_module.app.test_client()
    r = _post(client, EMAIL, {"slugs": ["grief-release"]})
    assert r.status_code == 403
    body = r.get_json()
    assert body["ok"] is False
    with sqlite3.connect(_db) as cx:
        cx.row_factory = sqlite3.Row
        assert life_stress_curation.get(cx, EMAIL) is None


def test_saves_resolving_slugs_drops_junk(wired):
    app_module, db_path = wired
    client = app_module.app.test_client()
    r = _post(client, EMAIL, {"slugs": ["grief-release", "junk-slug-not-real"],
                              "note": "for the grief pattern"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["saved"] == ["grief-release"]

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        stored = life_stress_curation.get(cx, EMAIL)
    assert stored["slugs"] == ["grief-release"]
    assert stored["note"] == "for the grief pattern"


def test_empty_slugs_clears(wired):
    app_module, db_path = wired
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        life_stress_curation.set(cx, EMAIL, PID, ["grief-release"], "prior note")

    client = app_module.app.test_client()
    r = _post(client, EMAIL, {"slugs": []})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["saved"] == []

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        assert life_stress_curation.get(cx, EMAIL) is None


def test_get_selection_returns_saved_curation(wired):
    app_module, db_path = wired
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        life_stress_curation.set(cx, EMAIL, PID, ["calm-clarity"], "steadier now")

    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/life-stress-selection/{EMAIL}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["curation"]["slugs"] == ["calm-clarity"]
    assert body["curation"]["note"] == "steadier now"
    assert body["curation"]["updated_at"] is not None


def test_get_selection_no_curation_yet(wired):
    app_module, _db = wired
    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/life-stress-selection/{EMAIL}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["curation"] is None
