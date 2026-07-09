"""Guard the two-migrations divergence: prod's boot path must produce the column,
not just dashboard.customers.add_people_columns() in isolation."""
import sqlite3

import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    return appmod


def test_init_people_table_creates_pickup_default(appmod):
    appmod._init_people_table()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cols = {r[1] for r in cx.execute("PRAGMA table_info(people)").fetchall()}
    assert "pickup_default" in cols, f"prod boot path missing pickup_default: {sorted(cols)}"


def test_init_people_table_is_idempotent(appmod):
    appmod._init_people_table()
    appmod._init_people_table()          # must not raise
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cols = {r[1] for r in cx.execute("PRAGMA table_info(people)").fetchall()}
    assert "pickup_default" in cols
