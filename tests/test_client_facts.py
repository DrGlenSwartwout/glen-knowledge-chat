import sqlite3
from dashboard import client_facts as cf

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cf.init_table(cx); return cx

def test_default_empty():
    assert cf.get_facts(_cx(), "a@b.com") == {}

def test_set_and_get_bool_fact():
    cx = _cx()
    cf.set_fact(cx, "A@B.com", "on_areds2", True)
    assert cf.get_facts(cx, "a@b.com") == {"on_areds2": True}
    cf.set_fact(cx, "a@b.com", "on_areds2", False)
    assert cf.get_facts(cx, "a@b.com") == {"on_areds2": False}
