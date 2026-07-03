"""dashboard.customers.rename_by_email — corrects a customer's display name across
BOTH the people record and all their orders (the invoice bills to the order name)."""
import sqlite3
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT, name TEXT, "
               "first_name TEXT, last_name TEXT, phone TEXT, notes TEXT, updated_at TEXT)")
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, name TEXT, updated_at TEXT)")
    return cx


def test_rename_updates_person_and_all_orders_without_clobbering():
    from dashboard import customers as C
    cx = _db()
    cx.execute("INSERT INTO people (email,name,first_name,last_name,phone,notes) VALUES (?,?,?,?,?,?)",
               ("des@x.com", "desiree dallaguardia", "desiree", "dallaguardia", "808", "keep me"))
    cx.execute("INSERT INTO orders (email,name) VALUES ('des@x.com','desiree dallaguardia')")
    cx.execute("INSERT INTO orders (email,name) VALUES ('des@x.com','desiree dallaguardia')")
    cx.execute("INSERT INTO orders (email,name) VALUES ('other@x.com','someone else')")
    cx.commit()

    res = C.rename_by_email(cx, "DES@x.com", name="Desiree' Dalla Guardia",
                            first_name="Desiree'", last_name="Dalla Guardia")
    assert res == {"people_updated": 1, "orders_updated": 2}

    p = cx.execute("SELECT * FROM people WHERE email='des@x.com'").fetchone()
    assert p["name"] == "Desiree' Dalla Guardia"
    assert p["last_name"] == "Dalla Guardia" and p["first_name"] == "Desiree'"
    assert p["phone"] == "808" and p["notes"] == "keep me"   # other fields untouched

    des_orders = {r["name"] for r in cx.execute("SELECT name FROM orders WHERE email='des@x.com'")}
    assert des_orders == {"Desiree' Dalla Guardia"}
    other = cx.execute("SELECT name FROM orders WHERE email='other@x.com'").fetchone()
    assert other["name"] == "someone else"   # a different customer is never touched


def test_rename_requires_email_and_name():
    from dashboard import customers as C
    cx = _db()
    with pytest.raises(ValueError):
        C.rename_by_email(cx, "", name="X")
    with pytest.raises(ValueError):
        C.rename_by_email(cx, "a@x.com", name="  ")
