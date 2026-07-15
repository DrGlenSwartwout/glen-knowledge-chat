"""Task 6: GET /api/practitioner/life-stress-selection/<patient_email> — a
read-only view of a client's saved Life Stress essence selection, for the
practitioner (Glen/Rae).

Patterns borrowed from tests/test_practitioner_program_endpoints.py: app-import
+ LOG_DB monkeypatch + _practitioner_session_pid override, and gate-order
assertions (flag off -> 404, no pid -> 401, not authorized -> 403), plus a
monkeypatched `_life_stress_for` (mirrors that file's `_client_condition_for`
override) so the 200 happy path is exercised without needing a real E4L scan.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import life_stress, life_stress_curation, life_stress_selection

PID = "doc1"
EMAIL = "pat@x.com"

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
    an always-authorized continuity_view.authorized_patient, a fixed Life
    Stress pool (`_life_stress_for`), and the flag ON."""
    app_module = _app()
    db_path = tmp_path / "chat_log.db"
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        life_stress_selection.init_table(cx)

    monkeypatch.setattr(app_module, "LOG_DB", db_path)
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: PID)
    from dashboard import continuity_view as cv
    monkeypatch.setattr(cv, "authorized_patient", lambda cx, pid, email: True)
    monkeypatch.setattr(app_module, "_life_stress_for", lambda email: dict(POOL_BLOCK))
    monkeypatch.setattr(life_stress, "recommend", lambda email, today: dict(POOL_BLOCK))
    return app_module, db_path


def test_flag_off_404s(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.delenv("LIFE_STRESS_ENABLED", raising=False)
    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/life-stress-selection/{EMAIL}")
    assert r.status_code == 404


def test_requires_auth(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: None)
    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/life-stress-selection/{EMAIL}")
    assert r.status_code == 401
    assert r.get_json()["ok"] is False


def test_403_when_not_authorized(monkeypatch, wired):
    app_module, _db = wired
    from dashboard import continuity_view as cv
    monkeypatch.setattr(cv, "authorized_patient", lambda cx, pid, email: False)
    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/life-stress-selection/{EMAIL}")
    assert r.status_code == 403
    body = r.get_json()
    assert body["ok"] is False
    assert "pool" not in body


def test_200_returns_pool_selected_and_updated_at(wired):
    app_module, db_path = wired
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        life_stress_selection.set(cx, EMAIL, ["grief-release"])

    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/life-stress-selection/{EMAIL}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["pool"] == [
        {"slug": "grief-release", "name": "Grief Release", "pattern": "grief pattern"},
        {"slug": "calm-clarity", "name": "Calm & Clarity", "pattern": "anxiety pattern"},
    ]
    assert body["selected"] == ["grief-release"]
    assert body["updated_at"] is not None


def test_200_no_selection_yet(wired):
    app_module, _db = wired
    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/life-stress-selection/{EMAIL}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["selected"] == []
    assert body["updated_at"] is None


def test_200_pool_is_raw_auto_pool_even_when_curated(monkeypatch, wired):
    """P2 review fix regression guard: `_life_stress_for` is curation-AWARE --
    once a curation exists for a client, it collapses to the curated subset
    (that's correct for the client-facing card). But this practitioner GET's
    `pool` must always reflect the FULL raw auto-pool from
    `life_stress.recommend`, uncollapsed, so the editor can be edited DOWN
    from the algorithm's full suggestion set. `curation` (from
    life_stress_curation.get) stays the separate pre-check source and still
    carries only the curated slugs. Pre-fix, `pool` came from
    `_life_stress_for` and would have wrongly collapsed to just
    ["grief-release"] here."""
    app_module, db_path = wired
    auto_pool = {
        "items": [
            {"slug": "grief-release", "name": "Grief Release", "note": "grief pattern"},
            {"slug": "calm-clarity", "name": "Calm & Clarity", "note": "anxiety pattern"},
            {"slug": "steady-ground", "name": "Steady Ground", "note": "overwhelm pattern"},
        ]
    }
    monkeypatch.setattr(life_stress, "recommend", lambda email, today: dict(auto_pool))

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        life_stress_curation.set(cx, EMAIL, PID, ["grief-release"], "practitioner note")

    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/life-stress-selection/{EMAIL}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["pool"] == [
        {"slug": "grief-release", "name": "Grief Release", "pattern": "grief pattern"},
        {"slug": "calm-clarity", "name": "Calm & Clarity", "pattern": "anxiety pattern"},
        {"slug": "steady-ground", "name": "Steady Ground", "pattern": "overwhelm pattern"},
    ]
    assert body["curation"] is not None
    assert body["curation"]["slugs"] == ["grief-release"]
