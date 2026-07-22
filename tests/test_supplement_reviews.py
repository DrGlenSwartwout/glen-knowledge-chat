"""Data-module tests for the free-product-review store (supplement_reviews).
Pure sqlite, no app import, no email. Mirrors tests/test_analysis_requests.py harness.
State machine: requested -> ai_draft -> confirmed (+ rejected side-state). Never downgrades."""
import sqlite3

from dashboard import supplement_reviews as sr


def _cx():
    cx = sqlite3.connect(":memory:")
    sr.init_table(cx)
    return cx


def test_init_idempotent():
    cx = _cx()
    sr.init_table(cx)  # second call must not raise
    assert sr.list_for_email(cx, "nobody@x.com") == []


def test_create_and_idempotent():
    cx = _cx()
    a = sr.create_request(cx, "K@x.com", "Neuro Magnesium", "Acme", source="portal")
    assert a["created"] is True and a["status"] == "requested" and a["id"] > 0
    # same product, different email casing/whitespace -> same row, not created again
    b = sr.create_request(cx, "  k@x.com ", "neuro magnesium", "  acme ", source="public")
    assert b["created"] is False and b["id"] == a["id"] and b["status"] == "requested"
    assert len(sr.list_for_email(cx, "k@x.com")) == 1


def test_distinct_products_are_separate_rows():
    cx = _cx()
    sr.create_request(cx, "a@x.com", "Product One", "BrandA")
    sr.create_request(cx, "a@x.com", "Product Two", "BrandA")
    assert len(sr.list_for_email(cx, "a@x.com")) == 2


def test_draft_then_confirm_flow():
    cx = _cx()
    r = sr.create_request(cx, "a@x.com", "Fish Oil", "OmegaCo")
    assert sr.set_draft(cx, r["id"], "Review body: contains soy lecithin.")["status"] == "ai_draft"
    row = sr.get(cx, r["id"])
    assert row["status"] == "ai_draft" and "soy lecithin" in row["review_text"]
    assert sr.set_status(cx, r["id"], "confirmed")["status"] == "confirmed"
    assert sr.get(cx, r["id"])["status"] == "confirmed"


def test_never_downgrade():
    cx = _cx()
    r = sr.create_request(cx, "a@x.com", "P", "B")
    sr.set_draft(cx, r["id"], "draft")
    sr.set_status(cx, r["id"], "confirmed")
    # attempts to move a confirmed review backward are no-ops
    assert sr.set_status(cx, r["id"], "ai_draft")["status"] == "confirmed"
    assert sr.set_status(cx, r["id"], "requested")["status"] == "confirmed"
    # set_draft must not overwrite a confirmed review's content
    sr.set_draft(cx, r["id"], "OVERWRITE ATTEMPT")
    assert sr.get(cx, r["id"])["review_text"] == "draft"
    assert sr.get(cx, r["id"])["status"] == "confirmed"


def test_reject_allowed_before_confirm_not_after():
    cx = _cx()
    r = sr.create_request(cx, "a@x.com", "P", "B")
    assert sr.set_status(cx, r["id"], "rejected")["status"] == "rejected"
    r2 = sr.create_request(cx, "b@x.com", "P", "B")
    sr.set_draft(cx, r2["id"], "d")
    sr.set_status(cx, r2["id"], "confirmed")
    # cannot reject something already confirmed
    assert sr.set_status(cx, r2["id"], "rejected")["status"] == "confirmed"


def test_pending_queue_excludes_terminal():
    cx = _cx()
    a = sr.create_request(cx, "a@x.com", "PA", "B")          # requested
    b = sr.create_request(cx, "b@x.com", "PB", "B"); sr.set_draft(cx, b["id"], "d")  # ai_draft
    c = sr.create_request(cx, "c@x.com", "PC", "B"); sr.set_draft(cx, c["id"], "d"); sr.set_status(cx, c["id"], "confirmed")
    d = sr.create_request(cx, "d@x.com", "PD", "B"); sr.set_status(cx, d["id"], "rejected")
    q = {row["id"] for row in sr.pending_queue(cx)}
    assert q == {a["id"], b["id"]}


def test_list_for_email_shape():
    cx = _cx()
    r = sr.create_request(cx, "a@x.com", "Vit D3", "SunCo", source="public")
    rows = sr.list_for_email(cx, "a@x.com")
    assert rows[0]["product_name"] == "Vit D3" and rows[0]["product_brand"] == "SunCo"
    assert rows[0]["status"] == "requested" and rows[0]["source"] == "public"


def test_blank_skipped():
    cx = _cx()
    assert sr.create_request(cx, "", "P", "B")["created"] is False
    assert sr.create_request(cx, "a@x.com", "  ", "B")["created"] is False
