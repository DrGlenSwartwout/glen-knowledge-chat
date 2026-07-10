"""Approving a studio-credit claim must not deadlock on its own connection.

The console hands the action a live sqlite connection (`ctx['cx']`, opened in
app.bos_action). approve_claim -> _studio_credit_grant_and_notify ->
_grant_membership writes on that connection, leaving a write transaction open.
_mint_membership_magic_link then opened a SECOND connection to the same file and
tried to insert, which SQLite answers with `database is locked` -- so the
Approve button always failed.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))


@pytest.fixture
def approve_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    from dashboard import studio_credit as sc

    sent = []
    monkeypatch.setattr(appmod, "_send_inquiry_email",
                        lambda **kw: sent.append(kw) or True)

    # Exactly how app.bos_action builds it: a plain connection, no _db_lock.
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    appmod.init_membership_tables(cx)
    sc.migrate(cx)
    cx.commit()
    yield appmod, sc, cx, sent
    cx.close()


def _tokens(cx, purpose):
    return cx.execute("SELECT COUNT(*) FROM auth_tokens WHERE purpose=?",
                      (purpose,)).fetchone()[0]


def test_approve_claim_does_not_lock_the_database(approve_env):
    appmod, sc, cx, sent = approve_env
    claim = sc.add_claim(cx, email="studio@example.com", invoice_ref="inv-1",
                         proof_note="", source="console", created_by="glen")
    cx.commit()

    res = sc.approve_claim(cx, claim["id"], decided_by="glen",
                           grant_fn=appmod._studio_credit_grant_and_notify)

    assert res["ok"] is True
    assert res["membership_id"]


def test_approve_claim_grants_membership_and_mints_exactly_one_link(approve_env):
    appmod, sc, cx, sent = approve_env
    claim = sc.add_claim(cx, email="studio@example.com", invoice_ref="inv-1",
                         proof_note="", source="console", created_by="glen")
    cx.commit()

    sc.approve_claim(cx, claim["id"], decided_by="glen",
                     grant_fn=appmod._studio_credit_grant_and_notify)

    grants = cx.execute("SELECT COUNT(*) FROM memberships WHERE email=?",
                        ("studio@example.com",)).fetchone()[0]
    assert grants == 1
    assert _tokens(cx, "membership_magic_link") == 1
    assert len(sent) == 1 and sent[0]["to_email"] == "studio@example.com"


def test_approved_claim_and_its_magic_link_commit_together(approve_env):
    """The token rides the caller's transaction, so it lands only if the
    approval does. A committed claim with no usable link is worse than neither."""
    appmod, sc, cx, sent = approve_env
    claim = sc.add_claim(cx, email="studio@example.com", invoice_ref="inv-1",
                         proof_note="", source="console", created_by="glen")
    cx.commit()

    sc.approve_claim(cx, claim["id"], decided_by="glen",
                     grant_fn=appmod._studio_credit_grant_and_notify)

    # A separate connection sees both, or neither.
    other = sqlite3.connect(appmod.LOG_DB)
    try:
        status = other.execute("SELECT status FROM studio_credit_claims WHERE id=?",
                               (claim["id"],)).fetchone()[0]
        toks = other.execute(
            "SELECT COUNT(*) FROM auth_tokens WHERE purpose='membership_magic_link'"
        ).fetchone()[0]
    finally:
        other.close()
    assert status == "approved"
    assert toks == 1


def test_mint_still_works_without_a_caller_connection(approve_env):
    """The standalone path (coaching sign-in, admin grant) opens its own
    connection and must keep working."""
    appmod, _sc, cx, _sent = approve_env
    tok = appmod._mint_membership_magic_link("solo@example.com")
    assert appmod._validate_membership_magic_link(tok) == "solo@example.com"
