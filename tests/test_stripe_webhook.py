# tests/test_stripe_webhook.py
import importlib, sqlite3, sys, json, time, hmac, hashlib
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
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return db


def _mock_paid_trial(app_module, monkeypatch, email="t@x.com", succeeded=True):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "biofield_trial", "email": email, "token": "tk"}, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1",
                    "status": "succeeded" if succeeded else "requires_payment_method"})


def _sign(payload: bytes, secret: str, ts: int) -> str:
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


def test_verify_webhook_accepts_valid():
    sp = _load("dashboard.stripe_pay")
    body = json.dumps({"type": "checkout.session.completed", "data": {"object": {"id": "cs_1"}}}).encode()
    ts = int(time.time())
    ev = sp.verify_webhook(body, _sign(body, "whsec_test", ts), "whsec_test")
    assert ev and ev["type"] == "checkout.session.completed" and ev["data"]["object"]["id"] == "cs_1"


def test_verify_webhook_rejects_tampered_body():
    sp = _load("dashboard.stripe_pay")
    body = b'{"type":"checkout.session.completed"}'
    ts = int(time.time())
    sig = _sign(body, "whsec_test", ts)
    assert sp.verify_webhook(b'{"type":"evil"}', sig, "whsec_test") is None


def test_verify_webhook_rejects_wrong_secret():
    sp = _load("dashboard.stripe_pay")
    body = b'{"a":1}'
    ts = int(time.time())
    assert sp.verify_webhook(body, _sign(body, "whsec_test", ts), "whsec_other") is None


def test_verify_webhook_rejects_stale():
    sp = _load("dashboard.stripe_pay")
    body = b'{"a":1}'
    old = int(time.time()) - 10000
    assert sp.verify_webhook(body, _sign(body, "whsec_test", old), "whsec_test", tolerance=300) is None


def test_verify_webhook_rejects_malformed_header():
    sp = _load("dashboard.stripe_pay")
    assert sp.verify_webhook(b'{}', "garbage", "whsec_test") is None
    assert sp.verify_webhook(b'{}', "", "whsec_test") is None


def test_fulfill_creates_membership_and_grant(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_trial(app_module, monkeypatch)
    res = app_module._fulfill_biofield_trial("cs_x")
    assert res["ok"] is True
    with sqlite3.connect(db) as cx:
        subs = cx.execute("SELECT amount_cents, status, kind FROM subscriptions WHERE email='t@x.com'").fetchall()
        grants = cx.execute("SELECT source FROM memberships WHERE email='t@x.com'").fetchall()
    assert subs == [(9900, "active", "membership")] and grants == [("biofield_trial",)]
    assert app_module._active_membership_for_email("t@x.com") is not None


def test_fulfill_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_trial(app_module, monkeypatch)
    app_module._fulfill_biofield_trial("cs_x")
    r2 = app_module._fulfill_biofield_trial("cs_x")
    assert r2.get("already") is True
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 1
        assert cx.execute("SELECT COUNT(*) FROM memberships WHERE email='t@x.com'").fetchone()[0] == 1


def test_fulfill_unpaid_creates_nothing(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_trial(app_module, monkeypatch, succeeded=False)
    res = app_module._fulfill_biofield_trial("cs_x")
    assert res["ok"] is False
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 0


def test_fulfill_non_trial_noop(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session", lambda s: {"metadata": {"kind": "reorder"}})
    res = app_module._fulfill_biofield_trial("cs_x")
    assert res["ok"] is False and res.get("reason") == "not_trial"
