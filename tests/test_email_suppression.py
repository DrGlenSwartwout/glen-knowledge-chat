import sqlite3
from dashboard import email_suppression as es


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    es.init_table(cx)
    return cx


def test_add_and_is_suppressed_case_insensitive():
    cx = _cx()
    es.add(cx, "Dead@Domain.com", "hard", "NXDOMAIN", "bounce-scan")
    assert es.is_suppressed(cx, "dead@domain.com") is True
    assert es.is_suppressed(cx, "  DEAD@domain.com ") is True
    assert es.is_suppressed(cx, "other@x.com") is False
    assert es.is_suppressed(cx, "") is False


def test_add_is_idempotent():
    cx = _cx()
    es.add(cx, "a@b.com", "hard", "no such user", "bounce-scan")
    es.add(cx, "a@b.com", "hard", "no such user", "bounce-scan")
    assert len(es.list_recent(cx)) == 1


def test_is_suppressed_no_table_is_false():
    cx = sqlite3.connect(":memory:")  # table never created
    assert es.is_suppressed(cx, "x@y.com") is False
