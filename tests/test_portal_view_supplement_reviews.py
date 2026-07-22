"""Portal-block tests for the free product review feature. Mirrors
tests/test_portal_view_ambassador.py: exercise the block function directly on an
in-memory db. Key rule: a review's TEXT is exposed only once confirmed."""
import sqlite3

from dashboard import portal_view as pv
from dashboard import supplement_reviews as sr


def _cx():
    cx = sqlite3.connect(":memory:")
    sr.init_table(cx)
    return cx


def test_block_off_when_flag_off():
    cx = _cx()
    sr.create_request(cx, "a@x.com", "Fish Oil", "OmegaCo")
    assert pv._supplement_reviews_block(cx, "a@x.com", enabled=False) == {"status": "off"}


def test_block_empty_when_no_reviews():
    cx = _cx()
    assert pv._supplement_reviews_block(cx, "a@x.com", enabled=True) == {"status": "empty", "reviews": []}


def test_block_lists_reviews_and_hides_unconfirmed_text():
    cx = _cx()
    # requested (no text), ai_draft (has text, must stay hidden), confirmed (text visible)
    sr.create_request(cx, "a@x.com", "Requested Prod", "B1")
    d = sr.create_request(cx, "a@x.com", "Draft Prod", "B2"); sr.set_draft(cx, d["id"], "SECRET DRAFT")
    c = sr.create_request(cx, "a@x.com", "Confirmed Prod", "B3"); sr.set_draft(cx, c["id"], "PUBLISHED REVIEW"); sr.set_status(cx, c["id"], "confirmed")

    block = pv._supplement_reviews_block(cx, "a@x.com", enabled=True)
    assert block["status"] == "has_reviews"
    by_name = {r["product_name"]: r for r in block["reviews"]}
    assert set(by_name) == {"Requested Prod", "Draft Prod", "Confirmed Prod"}
    # unconfirmed rows never carry review text
    assert "review" not in by_name["Requested Prod"]
    assert "review" not in by_name["Draft Prod"]
    # confirmed row reveals the text
    assert by_name["Confirmed Prod"]["review"] == "PUBLISHED REVIEW"
    assert by_name["Draft Prod"]["status"] == "ai_draft"


def test_block_blank_email_safe():
    cx = _cx()
    assert pv._supplement_reviews_block(cx, "", enabled=True) == {"status": "empty", "reviews": []}


def test_get_portal_view_threads_block():
    # minimal people row; sub-blocks are defensive so this composes without other tables
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT, name TEXT, first_name TEXT, last_name TEXT, roles TEXT)")
    cx.execute("INSERT INTO people (id,email,name,roles) VALUES (1,'a@x.com','A','[]')")
    sr.init_table(cx)
    sr.create_request(cx, "a@x.com", "P", "B")
    view = pv.get_portal_view(cx, 1, supplement_review_enabled=True)
    assert "supplement_review" in view and view["supplement_review"]["status"] == "has_reviews"
    # default (flag omitted) is off
    view2 = pv.get_portal_view(cx, 1)
    assert view2["supplement_review"] == {"status": "off"}
