import sqlite3
from dashboard import remedies_block, supplement_reviews as sr

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    sr.init_table(cx); return cx

def test_block_off_when_disabled():
    assert remedies_block.build_block(None, "a@b.com", False) == {"enabled": False}

def test_external_hides_unconfirmed_review_text():
    cx = _cx()
    sr.add_listed(cx, "a@b.com", "Fish Oil", "Acme", reason="heart", importance=6)
    rid = sr.list_for_email(cx, "a@b.com")[0]["id"]
    sr.create_request(cx, "a@b.com", "Fish Oil", "Acme")
    sr.set_draft(cx, rid, "SECRET draft not for client")
    blk = remedies_block.build_block(cx, "a@b.com", True)
    ext = blk["external"][0]
    assert ext["reason"] == "heart" and ext["importance"] == 6
    assert "review" not in ext or ext.get("review") in (None, "")   # ai_draft text stays hidden
