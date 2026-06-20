"""Begin #4a rev - biofield_reveals store: interpretation + remedies + first_approved."""
import sqlite3, sys
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


def _interp():
    return {"greeting": "Aloha", "body": "Your terrain reading."}


def _remedies():
    return [{"name": "Cistus Shield", "slug": "cistus-shield", "meaning": "Calm the terrain."},
            {"name": "Binder", "slug": "binder", "meaning": "Bind and clear."}]


def test_upsert_new_then_update(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid, is_new = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    assert is_new is True
    row = m.get(cx, rid)
    assert row["interpretation"]["greeting"] == "Aloha"
    assert len(row["remedies"]) == 2 and row["first_approved"] is False
    rid2, is_new2 = m.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Hi", "body": "v2"}, _remedies(), "s")
    assert rid2 == rid and is_new2 is False
    assert m.get(cx, rid)["interpretation"]["greeting"] == "Hi"


def test_no_overwrite_after_approval(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid, _ = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    m.approve_first(cx, rid, "glen")
    m.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "X", "body": "X"}, [], "s")
    row = m.get(cx, rid)
    assert row["first_approved"] is True
    assert row["interpretation"]["greeting"] == "Aloha"  # unchanged after approval


def test_approve_first_and_list_pending(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    r1, _ = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    r2, _ = m.upsert(cx, "b@x.com", "2026-06-19", _interp(), _remedies(), "s")
    assert m.approve_first(cx, r1, "glen") is True
    pending = m.list_pending(cx)
    assert [p["id"] for p in pending] == [r2]
    assert m.get(cx, r1)["approved_by"] == "glen"


def test_token_lookup(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid, _ = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    m.set_token(cx, rid, "H:tok")
    assert m.get_by_token_hash(cx, "H:tok")["id"] == rid


def test_edit_interpretation_and_remedies(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid, _ = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    m.set_interpretation(cx, rid, {"greeting": "Edited", "body": "new"})
    m.set_remedies(cx, rid, [{"name": "Only", "slug": "only", "meaning": "m"}])
    row = m.get(cx, rid)
    assert row["interpretation"]["greeting"] == "Edited" and len(row["remedies"]) == 1
