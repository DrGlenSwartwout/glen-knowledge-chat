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
