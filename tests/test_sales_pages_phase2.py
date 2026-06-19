import sqlite3
from dashboard import sales_pages as sp

def _cx():
    return sqlite3.connect(":memory:")

def test_upsert_then_get_section_roundtrip():
    cx = _cx()
    assert sp.get_section(cx, "longevity", "intro") is None
    sp.upsert_section(cx, "longevity", "intro", "Hello world.", model="m1")
    assert sp.get_section(cx, "longevity", "intro") == "Hello world."

def test_upsert_accretes_sections_in_one_row():
    cx = _cx()
    sp.upsert_section(cx, "energy", "intro", "A.")
    sp.upsert_section(cx, "energy", "description", "B.")
    page = sp.get_page(cx, "energy")
    assert page["content"] == {"intro": "A.", "description": "B."}
    assert page["state"] == "draft"

def test_get_page_missing_returns_none():
    assert sp.get_page(_cx(), "nope") is None
