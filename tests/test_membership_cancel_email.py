# tests/test_membership_cancel_email.py
import importlib, re, sqlite3, sys
from datetime import datetime
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        cx.execute(
            "CREATE TABLE IF NOT EXISTS auth_tokens "
            "(token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return db


def _mock_paid_session(app_module, monkeypatch, email="t@x.com"):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "biofield_trial", "email": email}, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})


def test_fulfill_sends_welcome_email_with_working_cancel_link(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_session(app_module, monkeypatch)
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body, **kw: sent.append((to, subj, body)) or True)
    res = app_module._fulfill_biofield_trial("cs_1")
    assert res.get("created") is True
    assert len(sent) == 1, f"expected exactly one email, got {len(sent)}"
    to, subj, body = sent[0]
    assert to == "t@x.com"
    m = re.search(r"/membership/cancel/(\S+)", body)
    assert m, "no cancel link in email body"
    tok = m.group(1)
    # the emailed link actually cancels the membership
    r = app_module.app.test_client().get(f"/membership/cancel/{tok}")
    assert r.status_code == 200
    with sqlite3.connect(db) as cx:
        status = cx.execute(
            "SELECT status FROM subscriptions WHERE email='t@x.com' AND kind='membership'").fetchone()
    assert status is not None and status[0] == "cancelled", f"got {status}"


def test_welcome_email_sent_exactly_once(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_session(app_module, monkeypatch)
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body, **kw: sent.append(to) or True)
    app_module._fulfill_biofield_trial("cs_1")
    app_module._fulfill_biofield_trial("cs_1")
    assert len(sent) == 1, f"expected one email across two fulfills, got {len(sent)}"


def test_email_failure_does_not_break_fulfillment(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_session(app_module, monkeypatch)
    def _boom(*a, **k):
        raise RuntimeError("smtp down")
    monkeypatch.setattr(app_module, "_send_inquiry_email", _boom)
    res = app_module._fulfill_biofield_trial("cs_1")
    assert res.get("created") is True, f"fulfillment should still succeed, got {res}"
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 1
        assert cx.execute("SELECT COUNT(*) FROM memberships WHERE email='t@x.com'").fetchone()[0] == 1


def test_cancel_token_ttl_extended(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_session(app_module, monkeypatch)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    app_module._fulfill_biofield_trial("cs_1")
    with sqlite3.connect(db) as cx:
        row = cx.execute(
            "SELECT created_at, expires_at FROM auth_tokens "
            "WHERE purpose='membership_cancel' AND email='t@x.com'").fetchone()
    assert row, "no membership_cancel token minted"
    created = datetime.fromisoformat(row[0])
    expires = datetime.fromisoformat(row[1])
    days = (expires - created).days
    assert days >= 365, f"cancel token TTL too short: {days} days"
    assert days == app_module.MEMBERSHIP_CANCEL_TTL_DAYS
