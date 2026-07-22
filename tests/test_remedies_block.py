import sqlite3
from dashboard import remedies_block, supplement_reviews as sr, portal_recommendations as pr

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

def test_ranked_reason_never_leaks_operator_note(monkeypatch):
    # Drive the real _build_ranked path (through build_block) but control what
    # portal_recommendations.build_sections hands back, so we can plant a product
    # carrying an internal-only operator_note and prove it never reaches `reason`.
    def fake_build_sections(product_sources, notes, section_state, resolve_product, *, top_n=5):
        return [{
            "source": "biofield",
            "products": [{
                "product_key": "x",
                "name": "X",
                "url": "",
                "operator_note": "INTERNAL do not show",
                "client_note": "",
            }],
        }]
    monkeypatch.setattr(pr, "build_sections", fake_build_sections)

    cx = _cx()
    blk = remedies_block.build_block(cx, "a@b.com", True)
    ranked = blk["ranked"]
    assert len(ranked) == 1
    assert ranked[0]["reason"] == ""
    assert "INTERNAL" not in ranked[0]["reason"]
