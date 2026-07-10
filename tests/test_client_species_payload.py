import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_species as cs


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def app_db(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    with sqlite3.connect(tmp_db) as cx:
        cs.init_table(cx)
        cs.upsert(cx, "care@example.com", "Cat", "Sasha")
        cs.upsert(cx, "person@example.com", "Human", "")
    return app


def test_flag_off_returns_none(app_db, monkeypatch):
    monkeypatch.delenv("ANIMAL_GREETING_ENABLED", raising=False)
    assert app_db._client_species_for("care@example.com") is None


def test_flag_on_animal(app_db, monkeypatch):
    monkeypatch.setenv("ANIMAL_GREETING_ENABLED", "1")
    b = app_db._client_species_for("care@example.com")
    assert b == {"is_animal": True, "animal_name": "Sasha"}


def test_flag_on_human_returns_none(app_db, monkeypatch):
    monkeypatch.setenv("ANIMAL_GREETING_ENABLED", "1")
    assert app_db._client_species_for("person@example.com") is None


def test_flag_on_unknown_returns_none(app_db, monkeypatch):
    monkeypatch.setenv("ANIMAL_GREETING_ENABLED", "1")
    assert app_db._client_species_for("stranger@example.com") is None


def test_broken_lookup_never_raises(app_db, monkeypatch):
    monkeypatch.setenv("ANIMAL_GREETING_ENABLED", "1")
    monkeypatch.setattr(cs, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
    assert app_db._client_species_for("care@example.com") is None
