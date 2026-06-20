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
    cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
    from dashboard import biofield_reveals
    biofield_reveals.init_table(cx)
    return cx


class _Actor:
    name = "glen"


def test_approve_confirms_mints_token_and_sends(tmp_path):
    br, acts = _mods()
    cx = _cx(tmp_path)
    rid = br.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Cistus"}, [], "s")
    sent = []
    acts.configure(base_url="https://x.test",
                   send=lambda to, subject, body: sent.append((to, subject, body)) or True,
                   hash_token=lambda t: "H:" + t, mint_token=lambda: "TOK123")
    acts._exec_approve({"id": rid}, {"cx": cx, "actor": _Actor()})
    row = br.get(cx, rid)
    assert row["status"] == "confirmed"
    at = cx.execute("SELECT email, purpose FROM auth_tokens WHERE token_hash=?", ("H:TOK123",)).fetchone()
    assert at == ("a@x.com", "biofield_reveal")
    assert len(sent) == 1 and "/begin/biofield/TOK123" in sent[0][2]


def test_approve_never_fails_on_send_error(tmp_path):
    br, acts = _mods()
    cx = _cx(tmp_path)
    rid = br.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "C"}, [], "s")
    def _boom(*a, **k):
        raise RuntimeError("smtp down")
    acts.configure(base_url="https://x.test", send=_boom,
                   hash_token=lambda t: "H:" + t, mint_token=lambda: "TOK")
    acts._exec_approve({"id": rid}, {"cx": cx, "actor": _Actor()})  # must not raise
    assert br.get(cx, rid)["status"] == "confirmed"


def test_edit_updates_top_stays_draft(tmp_path):
    br, acts = _mods()
    cx = _cx(tmp_path)
    rid = br.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Old"}, [], "s")
    acts._exec_edit({"id": rid, "name": "New", "meaning": "warm"}, {"cx": cx, "actor": _Actor()})
    row = br.get(cx, rid)
    assert row["status"] == "ai_draft" and row["top"]["name"] == "New" and row["top"]["meaning"] == "warm"
