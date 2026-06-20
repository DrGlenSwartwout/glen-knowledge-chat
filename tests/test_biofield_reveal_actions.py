import sqlite3, sys
from pathlib import Path
import pytest


def _mods():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import biofield_reveals, biofield_reveal_actions
        return biofield_reveals, biofield_reveal_actions
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    from dashboard import biofield_reveals
    biofield_reveals.init_table(cx)
    return cx


class _Actor:
    name = "glen"


def test_approve_flips_first_approved_no_email(tmp_path):
    br, acts = _mods(); cx = _cx(tmp_path)
    rid, _ = br.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Hi"}, [{"name": "C"}], "s")
    sent = []
    acts.configure(send=lambda *a, **k: sent.append(a))
    acts._exec_approve({"id": rid}, {"cx": cx, "actor": _Actor()})
    assert br.get(cx, rid)["first_approved"] is True
    assert sent == []  # no email on approve (it went out at ingest)


def test_edit_updates_interpretation_and_remedies(tmp_path):
    br, acts = _mods(); cx = _cx(tmp_path)
    rid, _ = br.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Old"}, [{"name": "A"}], "s")
    acts._exec_edit({"id": rid, "greeting": "New", "body": "b",
                     "remedies": [{"name": "B", "slug": "b", "meaning": "m"}]},
                    {"cx": cx, "actor": _Actor()})
    row = br.get(cx, rid)
    assert row["interpretation"]["greeting"] == "New" and row["remedies"][0]["name"] == "B"
    assert row["first_approved"] is False
