# tests/test_biofield_reveal_send.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load(mod):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"{mod} not importable: {e}")


def test_set_notified_and_list_approved_unnotified(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        r1, _ = br.upsert(cx, "a@x.com", "2026-06-20", {"body": "x"}, [], "s")  # approved, unnotified
        r2, _ = br.upsert(cx, "b@x.com", "2026-06-20", {"body": "x"}, [], "s")  # approved, notified
        r3, _ = br.upsert(cx, "c@x.com", "2026-06-20", {"body": "x"}, [], "s")  # not approved
        br.approve_first(cx, r1, "glen")
        br.approve_first(cx, r2, "glen")
        br.set_notified(cx, r2)
        ids = [r["id"] for r in br.list_approved_unnotified(cx)]
    assert r1 in ids and r2 not in ids and r3 not in ids
    with sqlite3.connect(db) as cx:
        row = br.get(cx, r2)
    assert row["notified_at"]
