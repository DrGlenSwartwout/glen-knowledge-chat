import sqlite3
from dashboard import sales_pages as sp


def _cx():
    return sqlite3.connect(":memory:")


def test_set_state_approved_stamps_by_and_time():
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "hello")
    sp.set_state(cx, "x", "approved", by="Glen")
    page = sp.get_page(cx, "x")
    assert page["state"] == "approved"
    row = cx.execute(
        "SELECT approved_at, approved_by FROM sales_pages WHERE product_slug='x'").fetchone()
    assert row[1] == "Glen" and row[0]  # approved_by set, approved_at non-empty


def test_set_state_draft_does_not_stamp_approver():
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "hello")
    sp.set_state(cx, "x", "approved", by="Glen")
    sp.set_state(cx, "x", "draft")
    page = sp.get_page(cx, "x")
    assert page["state"] == "draft"


def test_list_draft_pages_includes_content_excludes_empty():
    cx = _cx()
    sp.upsert_section(cx, "with-copy", "intro", "hello")
    sp.init_table(cx)
    # a row with empty content_json should be excluded
    cx.execute("INSERT INTO sales_pages (product_slug, content_json) VALUES ('empty','{}')")
    cx.commit()
    rows = sp.list_draft_pages(cx)
    slugs = [r["slug"] for r in rows]
    assert "with-copy" in slugs and "empty" not in slugs
    row = next(r for r in rows if r["slug"] == "with-copy")
    assert row["state"] == "draft" and row["sections"] == ["intro"]
