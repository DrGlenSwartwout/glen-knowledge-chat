import sqlite3, sys
from pathlib import Path
import pytest


def _mod():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import studio_credit
        return studio_credit
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    _mod().migrate(cx)
    return cx


def test_add_claim_creates_pending(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    c = m.add_claim(cx, email="Buyer@X.com", invoice_ref="INV-9",
                    proof_note="emailed 6/20", source="console", created_by="glen")
    assert c["status"] == "pending"
    assert c["email"] == "buyer@x.com"          # lowercased
    assert c["invoice_ref"] == "INV-9"
    assert c["source"] == "console"
    assert c["id"]
    got = m.get(cx, c["id"])
    assert got["email"] == "buyer@x.com" and got["status"] == "pending"


def test_list_claims_filters_by_status(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    m.add_claim(cx, email="a@x.com", source="console")
    m.add_claim(cx, email="b@x.com", source="console")
    assert len(m.list_claims(cx)) == 2
    assert len(m.list_claims(cx, status="pending")) == 2
    assert m.list_claims(cx, status="approved") == []
