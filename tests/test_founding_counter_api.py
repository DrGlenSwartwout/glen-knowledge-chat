# tests/test_founding_counter_api.py
import app as appmod
import dashboard.founding as founding


def test_status_open_and_closed(monkeypatch):
    monkeypatch.setattr(founding, "get_launch",
        lambda s: {"cap": 2500, "batch_label": "Founding Batch No. 1", "closes_at": ""} if s == "neuro-magnesium" else None)
    monkeypatch.setattr(founding, "remaining", lambda cx, s: 1847)
    monkeypatch.setattr(founding, "is_open", lambda cx, s, now_iso=None: True)
    c = appmod.app.test_client()
    r = c.get("/begin/founding/status/neuro-magnesium")
    assert r.status_code == 200
    d = r.get_json()
    assert d == {"open": True, "cap": 2500, "remaining": 1847, "batch_label": "Founding Batch No. 1"}

    r2 = c.get("/begin/founding/status/missing")
    assert r2.status_code == 404


def test_status_closed_remaining_zero(monkeypatch):
    monkeypatch.setattr(founding, "get_launch",
        lambda s: {"cap": 2500, "batch_label": "Founding Batch No. 1", "closes_at": ""} if s == "neuro-magnesium" else None)
    monkeypatch.setattr(founding, "remaining", lambda cx, s: 0)
    monkeypatch.setattr(founding, "is_open", lambda cx, s, now_iso=None: False)
    c = appmod.app.test_client()
    r = c.get("/begin/founding/status/neuro-magnesium")
    assert r.status_code == 200
    d = r.get_json()
    assert d["open"] is False
    assert d["remaining"] == 0
    assert d["cap"] == 2500
    assert d["batch_label"] == "Founding Batch No. 1"
