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


def _seed_memberships_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS memberships "
        "(id TEXT PRIMARY KEY, email TEXT, granted_at TEXT, expires_at TEXT, "
        " granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT)")
    cx.commit()


def _insert_membership(cx, email, source, granted_days_ago):
    from datetime import datetime, timedelta
    import uuid
    g = (datetime.utcnow() - timedelta(days=granted_days_ago)).isoformat() + "Z"
    e = (datetime.utcnow() - timedelta(days=granted_days_ago) + timedelta(days=30)).isoformat() + "Z"
    cx.execute("INSERT INTO memberships VALUES (?,?,?,?,?,?,?,?)",
               (str(uuid.uuid4()), email.lower(), g, e, source, source, "", ""))
    cx.commit()


class _GrantSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, cx, email, days):
        self.calls.append((email, days))
        mid = "mem-" + str(len(self.calls))
        _insert_membership(cx, email, "studio_credit", 0)  # simulate the real grant row
        return {"membership_id": mid, "magic_link_url": "https://x/coaching/auth/tok"}


def test_approve_grants_30_day_studio_credit(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    res = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    assert res["ok"] is True and res["membership_id"] == "mem-1"
    assert spy.calls == [("a@x.com", 30)]
    assert m.get(cx, c["id"])["status"] == "approved"


def test_double_approve_is_idempotent(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    res2 = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    assert res2.get("already") is True and res2["membership_id"] == "mem-1"
    assert len(spy.calls) == 1   # not granted again


def test_within_year_blocks_without_force(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    _insert_membership(cx, "a@x.com", "studio_credit", 100)  # 100 days ago
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    res = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    assert res["ok"] is False and res["warning"] == "granted_within_year"
    assert spy.calls == []
    assert m.get(cx, c["id"])["status"] == "pending"   # unchanged
    res2 = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy, force=True)
    assert res2["ok"] is True and spy.calls == [("a@x.com", 30)]


def test_grant_older_than_year_is_allowed(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    _insert_membership(cx, "a@x.com", "studio_credit", 400)  # >365 days ago
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    res = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    assert res["ok"] is True and spy.calls == [("a@x.com", 30)]


def test_reject_grants_nothing(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    res = m.reject_claim(cx, c["id"], decided_by="glen", reason="no invoice")
    assert res["ok"] is True
    assert spy.calls == []
    row = m.get(cx, c["id"])
    assert row["status"] == "rejected" and row["decision_note"] == "no invoice"
