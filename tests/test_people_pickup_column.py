import sqlite3

import dashboard.customers as C


def _people_db():
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE people (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        name TEXT DEFAULT '')""")
    cx.commit()
    return cx


def _cols(cx):
    return {r[1] for r in cx.execute("PRAGMA table_info(people)").fetchall()}


def test_migration_adds_pickup_default_and_address_columns():
    cx = _people_db()
    C.add_people_columns(cx)
    cols = _cols(cx)
    assert "pickup_default" in cols
    assert {"address1", "address2", "zip"} <= cols


def test_migration_is_idempotent():
    cx = _people_db()
    C.add_people_columns(cx)
    C.add_people_columns(cx)          # must not raise
    assert "pickup_default" in _cols(cx)


def test_pickup_default_defaults_to_zero():
    cx = _people_db()
    C.add_people_columns(cx)
    cx.execute("INSERT INTO people (email) VALUES ('a@b.com')")
    cx.commit()
    row = cx.execute("SELECT pickup_default FROM people WHERE email='a@b.com'").fetchone()
    assert row[0] == 0
