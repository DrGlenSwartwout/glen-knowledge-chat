import pytest
from dashboard import biofield_report_present as pres

REPORT = {"client": {"email": "a@b.com"}, "date": "2026-07-14"}


def test_disabled_returns_empty(monkeypatch):
    monkeypatch.delenv("LIFE_STRESS_ENABLED", raising=False)
    assert pres._life_stress(REPORT) == ""


def test_enabled_lists_essences_no_raw_stresses(monkeypatch):
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    calls = {}

    def _spy(email, day):
        calls["email"] = email
        return {"label": "Life Stress", "patterns": [],
                "items": [{"name": "Mimulus Flower Essence", "url": "", "note": "for the fear pattern in your scan"}]}

    monkeypatch.setattr(pres, "_ls_recommend", _spy)
    html = pres._life_stress(REPORT)
    assert calls["email"] == "a@b.com"          # email pulled from report["client"]["email"]
    assert "Mimulus Flower Essence" in html
    assert "Terrain Restore" in html
    assert "for the fear pattern" in html
    assert "<ER" not in html and "stress finding" not in html.lower()   # boundary: no raw stresses


def test_never_raises_on_bad_recommend(monkeypatch):
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(pres, "_ls_recommend", lambda e, d: (_ for _ in ()).throw(RuntimeError("boom")))
    assert pres._life_stress(REPORT) == ""
