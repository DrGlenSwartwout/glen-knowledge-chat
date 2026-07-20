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
])
def test_sanitize_strips_dangerous_content(raw, gone):
    assert gone not in sh.sanitize(raw)


def test_body_over_280_chars_is_rejected():
    cx = _cx()
    with pytest.raises(ValueError):
        sh.upsert_header(cx, "a@b.com", "Ann", "x" * 281)
