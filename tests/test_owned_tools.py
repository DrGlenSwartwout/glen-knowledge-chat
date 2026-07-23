import sqlite3
from dashboard import owned_tools as ot

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    ot.init_table(cx); return cx

def test_add_dedupes_and_lists():
    cx = _cx()
    assert ot.add(cx, "a@b.com", "Red Light Panel", "Joovv")["created"] is True
    assert ot.add(cx, "a@b.com", "red light panel", "Joovv")["created"] is False  # dedupe
    rows = ot.list_for(cx, "a@b.com")
    assert len(rows) == 1 and rows[0]["brand"] == "Joovv"

def test_owned_slugs_only_mapped():
    cx = _cx()
    ot.add(cx, "a@b.com", "Water Ionizer", "OtherCo", slug="water-ionizer")
    ot.add(cx, "a@b.com", "Random Gadget", "OtherCo")   # no slug
    assert ot.owned_slugs(cx, "a@b.com") == {"water-ionizer"}

def test_remove_scoped_to_email():
    cx = _cx()
    tid = ot.add(cx, "a@b.com", "Kloud", "X", slug="kloud")["id"]
    assert ot.remove(cx, "other@b.com", tid)["removed"] is False   # not your row
    assert ot.remove(cx, "a@b.com", tid)["removed"] is True
