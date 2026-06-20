import sqlite3
import sys
from pathlib import Path

import pytest


def _m():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import ingredient_pages
        return ingredient_pages
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    _m().init_table(cx)
    return cx


def test_section_scores_state(tmp_path):
    m = _m(); cx = _cx(tmp_path)
    m.upsert_section(cx, "zinc", "what_it_is", "An essential mineral.")
    assert m.get_section(cx, "zinc", "what_it_is") == "An essential mineral."
    m.set_scores(cx, "zinc", 9, 12)          # 12 clamps to 10
    p = m.get_page(cx, "zinc")
    assert p["research_score"] == 9 and p["traditional_score"] == 10
    m.set_traditional_use(cx, "zinc", [{"system": "TCM", "formula": "X", "uses": "y", "forms": "powder"}])
    m.set_related_forms(cx, "zinc", [{"name": "Zinc Oxide", "slug": "zinc-oxide", "verdict": "inferior", "note": "n"}])
    p = m.get_page(cx, "zinc")
    assert p["traditional_use"][0]["system"] == "TCM" and p["related_forms"][0]["verdict"] == "inferior"
    m.set_state(cx, "zinc", "approved", by="glen")
    assert m.get_page(cx, "zinc")["state"] == "approved"
    # bare query after a getter must still return tuples (no row_factory leak)
    assert isinstance(cx.execute("SELECT 1").fetchone(), tuple)


def test_requests_and_notify(tmp_path):
    m = _m(); cx = _cx(tmp_path)
    m.record_request(cx, "zinc", "a@x.com")
    m.record_request(cx, "zinc", "a@x.com")   # idempotent
    m.record_request(cx, "zinc", "b@x.com")
    assert {r["email"] for r in m.requesters_to_email(cx, "zinc")} == {"a@x.com", "b@x.com"}
    sent = []
    m.notify_on_approve(cx, "zinc", "Zinc", "https://x.test",
                        send=lambda to, subject, body: sent.append((to, body)) or True)
    assert len(sent) == 2 and all("/begin/ingredient/zinc" in b for _, b in sent)
    assert m.requesters_to_email(cx, "zinc") == []   # all marked emailed
    m.notify_on_approve(cx, "zinc", "Zinc", "https://x.test", send=lambda *a: sent.append(a) or True)
    assert len(sent) == 2   # at-most-once
