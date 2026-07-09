import sqlite3

import dashboard.customers as C


def _db():
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE people (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        name TEXT DEFAULT '', first_name TEXT DEFAULT '', last_name TEXT DEFAULT '',
        phone TEXT DEFAULT '', city TEXT DEFAULT '', state TEXT DEFAULT '',
        country TEXT DEFAULT '', order_count INTEGER DEFAULT 0,
        last_order_date TEXT DEFAULT '')""")
    cx.commit()
    C.add_people_columns(cx)
    return cx


def _person(cx, email):
    cx.execute("INSERT INTO people (email) VALUES (?)", (email,))
    cx.commit()
    return cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]


def test_unknown_email_is_not_pickup():
    """Fail toward CHARGING shipping: never free-ship on a guess."""
    cx = _db()
    assert C.pickup_default_for_email(cx, "nobody@example.com") is False
    assert C.pickup_default_for_email(cx, "") is False
    assert C.pickup_default_for_email(cx, None) is False


def test_set_and_read_round_trip():
    cx = _db()
    pid = _person(cx, "d@x.com")
    assert C.pickup_default_for_email(cx, "d@x.com") is False
    C.set_pickup_default(cx, pid, True)
    assert C.pickup_default_for_email(cx, "d@x.com") is True
    C.set_pickup_default(cx, pid, False)
    assert C.pickup_default_for_email(cx, "d@x.com") is False


def test_email_lookup_is_case_insensitive():
    cx = _db()
    pid = _person(cx, "d@x.com")
    C.set_pickup_default(cx, pid, True)
    assert C.pickup_default_for_email(cx, "D@X.COM") is True


def test_missing_column_reads_false_not_raise():
    """A pre-migration DB must resolve False, never explode a checkout."""
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL)")
    cx.execute("INSERT INTO people (email) VALUES ('d@x.com')")
    cx.commit()
    assert C.pickup_default_for_email(cx, "d@x.com") is False


def test_picker_cols_exposes_pickup_default():
    """A column absent from PICKER_COLS never reaches the browser."""
    assert "pickup_default" in C.PICKER_COLS
