"""Task 3: GET /api/practitioner/condition-program/<patient_email> — the
practitioner condition-program composer candidate list.

Patterns borrowed from tests/test_continuity_routes.py: app-import + LOG_DB
monkeypatch + _practitioner_session_pid override, and gate-order assertions
(flag off -> 404, no pid -> 401, not authorized -> 403).
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import condition_programs
from dashboard import practitioner_programs as pp

PID = "doc1"
EMAIL = "pat@x.com"
CONDITION_KEY = "dry-amd"


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
    an always-authorized continuity_view.authorized_patient, a resolved
    condition key, and the composer flag ON. Seeds a dry-amd program with a
    diagnosis-implied client_default add modifier and a clinician-measured
    add modifier."""
    app_module = _app()
    db_path = tmp_path / "chat_log.db"
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        condition_programs.init_table(cx)
        pp.init_table(cx)
        condition_programs.upsert(
            cx, CONDITION_KEY, "Dry AMD Support", False,
            items=[{"slug": "wholomega", "name": "WholOmega"}],
            modifiers=[
                {"when": "drusen", "action": "add", "source": "diagnosis-implied",
                 "client_default": True,
                 "items": [{"slug": "drusen-support", "name": "Drusen Support"}]},
                {"when": "iop-measured", "action": "add", "source": "clinician-measured",
                 "items": [{"slug": "clinician-only", "name": "Clinician Only"}]},
            ])

    monkeypatch.setattr(app_module, "LOG_DB", db_path)
    monkeypatch.setenv("PROGRAM_COMPOSER_ENABLED", "1")
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: PID)
    from dashboard import continuity_view as cv
    monkeypatch.setattr(cv, "authorized_patient", lambda cx, pid, email: True)
    monkeypatch.setattr(app_module, "_client_condition_for", lambda email: CONDITION_KEY)
    return app_module, db_path


def test_flag_off_404s(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.delenv("PROGRAM_COMPOSER_ENABLED", raising=False)
    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/condition-program/{EMAIL}")
    assert r.status_code == 404


def test_requires_auth(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: None)
    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/condition-program/{EMAIL}")
    assert r.status_code == 401
    assert r.get_json()["ok"] is False


def test_403_when_not_authorized_with_no_data_in_body(monkeypatch, wired):
    app_module, _db = wired
    from dashboard import continuity_view as cv
    monkeypatch.setattr(cv, "authorized_patient", lambda cx, pid, email: False)
    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/condition-program/{EMAIL}")
    assert r.status_code == 403
    body = r.get_json()
    assert body["ok"] is False
    assert "candidates" not in body


def test_candidates_checked_rule(wired):
    app_module, _db = wired
    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/condition-program/{EMAIL}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["condition_key"] == CONDITION_KEY
    assert body["label"] == "Dry AMD Support"
    assert body["saved"] is None
    candidates = body["candidates"]

    base = [c for c in candidates if c["section"] == "base"]
    assert {c["slug"] for c in base} == {"wholomega"}
    assert all(c["checked"] is True for c in base)

    drusen = next(c for c in candidates if c["slug"] == "drusen-support")
    assert drusen["section"] == "modifier"
    assert drusen["when"] == "drusen"
    assert drusen["source"] == "diagnosis-implied"
    assert drusen["checked"] is True

    clinician = next(c for c in candidates if c["slug"] == "clinician-only")
    assert clinician["section"] == "modifier"
    assert clinician["when"] == "iop-measured"
    assert clinician["source"] == "clinician-measured"
    assert clinician["checked"] is False


def test_returns_saved_program_when_present(wired):
    app_module, db_path = wired
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        pp.upsert(cx, patient_email=EMAIL, practitioner_id=PID, condition_key=CONDITION_KEY,
                  items=[{"slug": "wholomega", "name": "WholOmega"}], note="start low")

    client = app_module.app.test_client()
    r = client.get(f"/api/practitioner/condition-program/{EMAIL}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["saved"] == {"items": [{"slug": "wholomega", "name": "WholOmega"}], "note": "start low"}


# --- Task 4: POST /api/practitioner/condition-program (save) ---

def _post_body(**overrides):
    body = {
        "token": "irrelevant-here",
        "patient_email": EMAIL,
        "condition_key": CONDITION_KEY,
        "items": [{"slug": "wholomega", "name": "WholOmega", "dose": "1/day"}],
        "note": "start low",
    }
    body.update(overrides)
    return body


def test_post_flag_off_404s(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.delenv("PROGRAM_COMPOSER_ENABLED", raising=False)
    client = app_module.app.test_client()
    r = client.post("/api/practitioner/condition-program", json=_post_body())
    assert r.status_code == 404


def test_post_requires_auth(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: None)
    client = app_module.app.test_client()
    r = client.post("/api/practitioner/condition-program", json=_post_body())
    assert r.status_code == 401
    assert r.get_json()["ok"] is False


def test_post_403_when_not_authorized_and_nothing_written(monkeypatch, wired):
    app_module, db_path = wired
    from dashboard import continuity_view as cv
    monkeypatch.setattr(cv, "authorized_patient", lambda cx, pid, email: False)
    client = app_module.app.test_client()
    r = client.post("/api/practitioner/condition-program", json=_post_body())
    assert r.status_code == 403
    assert r.get_json()["ok"] is False

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        assert pp.get(cx, EMAIL) is None


def test_post_authorized_save_persists(wired):
    app_module, db_path = wired
    client = app_module.app.test_client()
    r = client.post("/api/practitioner/condition-program", json=_post_body())
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["saved"] is True

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        saved = pp.get(cx, EMAIL)
    assert saved is not None
    assert saved["condition_key"] == CONDITION_KEY
    assert saved["items"] == [{"slug": "wholomega", "name": "WholOmega", "dose": "1/day"}]
    assert saved["note"] == "start low"
