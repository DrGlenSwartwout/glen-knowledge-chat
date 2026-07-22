import sqlite3
from dashboard import supplement_reviews as sr

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    sr.init_table(cx); return cx

def test_add_listed_then_set_meta_and_promote():
    cx = _cx()
    r = sr.add_listed(cx, "a@b.com", "Magnesium Glycinate", "Acme", reason="sleep", importance=8)
    assert r["created"] is True and r["status"] == "listed"
    key = sr._key("Magnesium Glycinate", "Acme")
    sr.set_meta(cx, "a@b.com", key, reason="sleep + cramps", importance=9)
    row = [x for x in sr.list_for_email(cx, "a@b.com")][0]
    assert row["reason"] == "sleep + cramps" and row["importance"] == 9
    # request-review promotes listed -> requested via the existing pipeline
    assert sr.create_request(cx, "a@b.com", "Magnesium Glycinate", "Acme")["status"] == "requested"

def test_remove_only_before_review_exists():
    cx = _cx()
    sr.add_listed(cx, "a@b.com", "Fish Oil", "Acme")
    key = sr._key("Fish Oil", "Acme")
    assert sr.remove(cx, "a@b.com", key)["removed"] is True
    # a confirmed review is protected from client removal
    sr.create_request(cx, "a@b.com", "Vit D", "Acme")
    rid = sr.list_for_email(cx, "a@b.com")[0]["id"]
    sr.set_draft(cx, rid, "review text"); sr.set_status(cx, rid, "confirmed")
    assert sr.remove(cx, "a@b.com", sr._key("Vit D", "Acme"))["removed"] is False
