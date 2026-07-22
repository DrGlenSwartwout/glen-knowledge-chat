"""Access-model tests: per-client free-review access, default ON (opt-out), and
its effect on the portal block. Pure sqlite, no app import."""
import sqlite3

from dashboard import portal_view as pv
from dashboard import supplement_reviews as sr


def _cx():
    cx = sqlite3.connect(":memory:")
    sr.init_table(cx)
    return cx


def test_access_default_on():
    cx = _cx()
    assert sr.access_enabled(cx, "a@x.com") is True   # no row = enabled


def test_set_access_off_then_on():
    cx = _cx()
    assert sr.set_access(cx, "A@x.com", False, by="prac")["enabled"] is False
    assert sr.access_enabled(cx, "a@x.com") is False   # normalized, opted off
    sr.set_access(cx, "a@x.com", True, by="prac")
    assert sr.access_enabled(cx, "a@x.com") is True


def test_block_hidden_when_access_off():
    cx = _cx()
    sr.create_request(cx, "a@x.com", "P", "B")
    assert pv._supplement_reviews_block(cx, "a@x.com", enabled=True)["status"] == "has_reviews"
    sr.set_access(cx, "a@x.com", False)
    assert pv._supplement_reviews_block(cx, "a@x.com", enabled=True) == {"status": "off"}


def test_access_blank_email_safe():
    cx = _cx()
    assert sr.access_enabled(cx, "") is True
    assert sr.set_access(cx, "", True)["email"] is None
