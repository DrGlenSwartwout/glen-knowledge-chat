import importlib, sqlite3, sys
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


def test_diff_surname_same_street_makes_candidate(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _person(cx, "a@x.com", "Ann", "Lee", "12 Palm St", "96720")
        _person(cx, "b@x.com", "Bo", "Reyes", " 12  Palm St ", "96720")  # dirty, diff surname
        cx.commit()
    appmod.detect_household_candidates()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sigs = [r[0] for r in cx.execute(
            "SELECT signal FROM household_candidates").fetchall()]
    assert "shared-street-address" in sigs


def test_empty_street_makes_no_street_candidate(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _person(cx, "a@x.com", "Ann", "Lee", "", "")
        _person(cx, "b@x.com", "Bo", "Reyes", "", "")
        cx.commit()
    appmod.detect_household_candidates()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sigs = [r[0] for r in cx.execute("SELECT signal FROM household_candidates").fetchall()]
    assert "shared-street-address" not in sigs
