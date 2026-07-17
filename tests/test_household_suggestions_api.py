import importlib, sqlite3, sys, json
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _person(cx, email, first, last, address1="", zip=""):
    cx.execute("INSERT INTO people (email, first_name, last_name, address1, zip) "
               "VALUES (?,?,?,?,?)", (email, first, last, address1, zip))
    return cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]


def test_suggestions_returns_same_street_other(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        a = _person(cx, "a@x.com", "Ann", "Lee", "12 Palm St", "96720")
        b = _person(cx, "b@x.com", "Bo", "Reyes", "12 Palm St", "96720")
        cx.commit()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "testkey")
    c = appmod.app.test_client()
    r = c.get(f"/api/people/{a}/household-suggestions", headers={"X-Console-Key": "testkey"})
    body = r.get_json()
    ids = [s["person_id"] for s in body["suggestions"]]
    assert b in ids and a not in ids
