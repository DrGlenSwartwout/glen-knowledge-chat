# tests/test_patient_brand.py
"""Feature D (branded patient experience): the patient portal shows the patient's
attributed doctor's PUBLIC identity (name/practice/photo/logo/accent) — attribution-only,
NOT gated on practitioner_share_consent. Task 2 renders the band; this covers the
_patient_practitioner_brand() helper + api_client_portal payload wiring.

Real branding_json keys (confirmed from static/practitioner-settings.html): the color
picker saves brand_color_1 (header/primary) and brand_color_2, labeled "Brand color 2
(accent)" in the UI — so the helper's "accent" is branding["brand_color_2"].
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return importlib.import_module("app")


# ── Helper unit tests ───────────────────────────────────────────────────────

def test_brand_for_attributed_patient_with_branding(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_last_attributed_practitioner",
                        lambda email, **k: {"pid": "prac-42", "consent": 0})
    from dashboard import practitioner_settings as ps
    monkeypatch.setattr(ps, "get_settings", lambda cx, pid: {"branding": {
        "practice_name": "Vital Roots", "photo_url": "http://x/p.jpg",
        "logo_url": "", "brand_color_1": "#3d8a52", "brand_color_2": "#123456"}})
    monkeypatch.setattr(app, "_practitioner_display_name", lambda pid: "Dr. Jane Ríos")
    b = app._patient_practitioner_brand("pat@x.com")
    assert b["name"] == "Dr. Jane Ríos" and b["practice_name"] == "Vital Roots"
    assert b["photo_url"] == "http://x/p.jpg" and b["accent"] == "#123456"


def test_none_when_no_attributed_doctor(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_last_attributed_practitioner", lambda email, **k: None)
    assert app._patient_practitioner_brand("pat@x.com") is None


def test_none_when_branding_empty(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_last_attributed_practitioner",
                        lambda email, **k: {"pid": "prac-42", "consent": 1})
    from dashboard import practitioner_settings as ps
    monkeypatch.setattr(ps, "get_settings", lambda cx, pid: {"branding": {}})
    monkeypatch.setattr(app, "_practitioner_display_name", lambda pid: "Dr. Jane")
    assert app._patient_practitioner_brand("pat@x.com") is None   # no brand to show


def test_consent_independent(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_last_attributed_practitioner",
                        lambda email, **k: {"pid": "prac-42", "consent": 0})  # NOT consented
    from dashboard import practitioner_settings as ps
    monkeypatch.setattr(ps, "get_settings", lambda cx, pid: {"branding": {"practice_name": "Vital Roots"}})
    monkeypatch.setattr(app, "_practitioner_display_name", lambda pid: "Dr. Jane")
    assert app._patient_practitioner_brand("pat@x.com") is not None   # branding shows regardless of consent


def test_never_raises_on_failure(monkeypatch):
    app = _app()
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(app, "_last_attributed_practitioner", boom)
    assert app._patient_practitioner_brand("pat@x.com") is None   # swallowed -> None


# ── Route test: api_client_portal payload wiring ────────────────────────────

@pytest.fixture
def client(monkeypatch, tmp_path):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", str(tmp_path / "chat_log.db"))
    app._init_auth_tables()
    monkeypatch.setattr(app, "CONSOLE_SECRET", "test-secret")
    app.app.config["TESTING"] = True
    return app.app.test_client(), app


def _seed_portal(appmod, email, name, content=None):
    from dashboard import client_portal as cp
    content = content or {"greeting": "hi", "video": {}, "layers": [], "reorder_items": []}
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, content)
    cx.close()
    return token


def test_portal_payload_includes_brand_for_attributed_patient(client, monkeypatch):
    c, appmod = client
    tok = _seed_portal(appmod, "attributed@example.com", "Attributed Patient")
    monkeypatch.setattr(appmod, "_last_attributed_practitioner",
                        lambda email, **k: {"pid": "prac-1", "consent": 0})
    from dashboard import practitioner_settings as ps
    monkeypatch.setattr(ps, "get_settings", lambda cx, pid: {"branding": {
        "practice_name": "Sunrise Wellness", "photo_url": "http://x/doc.jpg",
        "logo_url": "http://x/logo.png", "brand_color_1": "#3d8a52", "brand_color_2": "#d4a843"}})
    monkeypatch.setattr(appmod, "_practitioner_display_name", lambda pid: "Dr. Jane Ríos")

    r = c.get(f"/api/portal/{tok}")
    assert r.status_code == 200
    j = r.get_json()
    assert j["practitioner_brand"] is not None
    assert j["practitioner_brand"]["name"] == "Dr. Jane Ríos"
    assert j["practitioner_brand"]["practice_name"] == "Sunrise Wellness"
    assert j["practitioner_brand"]["accent"] == "#d4a843"


def test_portal_payload_omits_brand_for_non_attributed_patient(client, monkeypatch):
    c, appmod = client
    tok = _seed_portal(appmod, "noattr@example.com", "No Attribution Patient")
    monkeypatch.setattr(appmod, "_last_attributed_practitioner", lambda email, **k: None)

    r = c.get(f"/api/portal/{tok}")
    assert r.status_code == 200
    j = r.get_json()
    assert not j.get("practitioner_brand")   # key absent or None -- never a real brand dict
