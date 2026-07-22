"""Route tests for the practitioner product-review controls: submit-for-client,
per-client access toggle, all-clients toggle, and console access override.
Skips if app import needs secrets. Practitioner session + portal_data + roster
are monkeypatched so no real magic-link session is needed."""
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")
os.environ.setdefault("SUPPLEMENT_REVIEW_ENABLED", "on")

import pytest

try:
    import app
    import dashboard
    from dashboard import supplement_reviews as _sr
    from dashboard import continuity_view as _cv
    from dashboard import practitioner_portal as _pp
except Exception as e:  # pragma: no cover
    pytest.skip(f"app import needs secrets: {e}", allow_module_level=True)


def _setup(mp, tmp):
    dbp = str(tmp / "chat_log.db")
    mp.setenv("DATA_DIR", str(tmp))
    mp.setenv("SUPPLEMENT_REVIEW_ENABLED", "on")
    mp.setattr(app, "LOG_DB", dbp, raising=False)
    for o in (app, dashboard):
        mp.setattr(o, "CONSOLE_SECRET", "sek", raising=False)
    app._init_people_table()
    mp.setattr(app, "_practitioner_session_pid", lambda: 1, raising=False)
    mp.setattr(_pp, "portal_data", lambda pid, **k: {"email": "prac@x.com"}, raising=False)
    return dbp


def test_practitioner_submits_for_client(monkeypatch, tmp_path):
    dbp = _setup(monkeypatch, tmp_path)
    r = app.app.test_client().post("/api/practitioner/product-review/request",
                                   json={"client_email": "client@x.com", "product_name": "Fish Oil"})
    assert r.status_code == 200 and r.get_json()["client"] is True
    cx = sqlite3.connect(dbp)
    rows = _sr.list_for_email(cx, "client@x.com")
    cx.close()
    assert len(rows) == 1 and rows[0]["source"].startswith("practitioner:")


def test_practitioner_per_client_access_toggle(monkeypatch, tmp_path):
    dbp = _setup(monkeypatch, tmp_path)
    r = app.app.test_client().post("/api/practitioner/product-review/access",
                                   json={"client_email": "client@x.com", "enabled": False})
    assert r.status_code == 200 and r.get_json()["enabled"] is False
    cx = sqlite3.connect(dbp); _sr.init_table(cx)
    assert _sr.access_enabled(cx, "client@x.com") is False
    cx.close()


def test_practitioner_all_clients_toggle(monkeypatch, tmp_path):
    dbp = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(_cv, "roster",
                        lambda cx, pid: [{"email": "c1@x.com", "name": "C1"}, {"email": "c2@x.com", "name": "C2"}])
    r = app.app.test_client().post("/api/practitioner/product-review/access-all", json={"enabled": False})
    assert r.status_code == 200 and r.get_json()["clients"] == 2
    cx = sqlite3.connect(dbp); _sr.init_table(cx)
    assert _sr.access_enabled(cx, "c1@x.com") is False and _sr.access_enabled(cx, "c2@x.com") is False
    cx.close()


def test_console_access_override(monkeypatch, tmp_path):
    dbp = _setup(monkeypatch, tmp_path)
    c = app.app.test_client()
    r = c.post("/api/console/product-review/access",
               headers={"X-Console-Key": "sek"}, json={"email": "z@x.com", "enabled": False})
    assert r.status_code == 200 and r.get_json()["enabled"] is False
    assert c.post("/api/console/product-review/access", json={"email": "z@x.com", "enabled": False}).status_code == 401
