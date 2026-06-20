"""Begin #4a - biofield_reveals store: ai_draft -> confirmed, idempotent draft."""
import sqlite3
import sys
from pathlib import Path
import pytest


def _mod():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import biofield_reveals
        return biofield_reveals
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    _mod().init_table(cx)
    return cx


def test_upsert_creates_draft(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19",
                         {"name": "Cistus Shield", "slug": "cistus-shield", "meaning": "Calm the terrain."},
                         [{"kind": "binder"}, {"kind": "mineral"}], "e4l-matcher")
    row = m.get(cx, rid)
    assert row["status"] == "ai_draft"
    assert row["top"]["name"] == "Cistus Shield"
    assert len(row["blurred"]) == 2


def test_upsert_updates_while_draft(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "One"}, [], "s")
    rid2 = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Two"}, [], "s")
    assert rid == rid2
    assert m.get(cx, rid)["top"]["name"] == "Two"


def test_confirmed_not_overwritten(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "One"}, [], "s")
    m.approve(cx, rid, "glen", "hash123")
    m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Two"}, [], "s")
    row = m.get(cx, rid)
    assert row["status"] == "confirmed"
    assert row["top"]["name"] == "One"


def test_set_top_stays_draft(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "One"}, [], "s")
    m.set_top(cx, rid, {"name": "Edited", "meaning": "new"})
    row = m.get(cx, rid)
    assert row["status"] == "ai_draft" and row["top"]["name"] == "Edited"


def test_approve_and_token_lookup(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "One"}, [], "s")
    assert m.approve(cx, rid, "glen", "hashABC") is True
    row = m.get_by_token_hash(cx, "hashABC")
    assert row["id"] == rid and row["status"] == "confirmed" and row["approved_by"] == "glen"


def test_list_drafts_only_drafts(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    r1 = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "A"}, [], "s")
    r2 = m.upsert_draft(cx, "b@x.com", "2026-06-19", {"name": "B"}, [], "s")
    m.approve(cx, r1, "glen", "h1")
    drafts = m.list_drafts(cx)
    assert [d["id"] for d in drafts] == [r2]
