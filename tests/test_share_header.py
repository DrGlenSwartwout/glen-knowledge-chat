import sqlite3
import pytest
from dashboard import share_header as sh


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    sh.init_share_headers_table(cx)
    return cx


def test_init_is_idempotent():
    cx = _cx()
    sh.init_share_headers_table(cx)  # second call must not raise


def test_upsert_lands_as_pending():
    cx = _cx()
    row = sh.upsert_header(cx, "a@b.com", "Ann", "Been working with Glen since March.")
    assert row["status"] == "pending"


def test_get_approved_returns_none_while_pending():
    cx = _cx()
    sh.upsert_header(cx, "a@b.com", "Ann", "Hello there.")
    assert sh.get_approved(cx, "a@b.com") is None


def test_get_approved_returns_row_after_approval():
    cx = _cx()
    sh.upsert_header(cx, "a@b.com", "Ann", "Hello there.")
    sh.approve(cx, "a@b.com")
    row = sh.get_approved(cx, "a@b.com")
    assert row["display_name"] == "Ann"
    assert row["body"] == "Hello there."


def test_editing_an_approved_header_resets_to_pending():
    cx = _cx()
    sh.upsert_header(cx, "a@b.com", "Ann", "First version.")
    sh.approve(cx, "a@b.com")
    sh.upsert_header(cx, "a@b.com", "Ann", "Second version.")
    assert sh.get_approved(cx, "a@b.com") is None


def test_rejected_header_never_renders():
    cx = _cx()
    sh.upsert_header(cx, "a@b.com", "Ann", "Hello there.")
    sh.reject(cx, "a@b.com")
    assert sh.get_approved(cx, "a@b.com") is None


@pytest.mark.parametrize("raw,gone", [
    ("<script>alert(1)</script>hi", "<script>"),
    ("visit https://evil.example.com now", "https://"),
    ("mail me at a@b.com", "@"),
    ("call 808-555-1212", "555"),
    ("call 5551212 now", "5551212"),
    ("call 555 1212 now", "1212"),
    ("+1 (808) 555-1212", "555"),
    ("evil.com/path", "evil.com"),
    ("evil.com", "evil.com"),
    ("HTTPS://EVIL.COM", "EVIL.COM"),
    ("javascript:alert(1)", "javascript:"),
    ("data:text/html,x", "data:"),
    ("a+tag@sub.example.co.uk", "@"),
    ("</script>bye", "</script>"),
    ("<img src=x>hi", "<img"),
    ("<<script>>alert(1)", "<script>"),
    ("<scr<x>ipt>alert(1)</scr<x>ipt>", "<scr"),
])
def test_sanitize_strips_dangerous_content(raw, gone):
    assert gone not in sh.sanitize(raw)


@pytest.mark.parametrize("raw", [
    "I have felt better for 6 months",
    "5 < 10 and it cost 42",
    "5 < 10 > 3",
    "Started in 2026",
])
def test_sanitize_preserves_legitimate_prose(raw):
    assert sh.sanitize(raw) == raw


def test_body_over_280_chars_is_rejected():
    cx = _cx()
    with pytest.raises(ValueError):
        sh.upsert_header(cx, "a@b.com", "Ann", "x" * 281)
