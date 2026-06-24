import hashlib
import hmac
import json
from dashboard import easypost as EP


def test_create_tracker_requires_key(monkeypatch):
    monkeypatch.delenv("EASYPOST_API_KEY", raising=False)
    try:
        EP.create_tracker("TN123")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_verify_webhook_accepts_valid():
    secret = "ephook_test"
    body = json.dumps({"description": "tracker.updated",
                       "result": {"status": "delivered", "tracking_code": "TN9"}}).encode()
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    ev = EP.verify_webhook(body, "hmac-sha256-hex=" + sig, secret)
    assert ev and ev["result"]["tracking_code"] == "TN9"


def test_verify_webhook_rejects_bad_sig():
    body = json.dumps({"description": "tracker.updated"}).encode()
    assert EP.verify_webhook(body, "hmac-sha256-hex=deadbeef", "ephook_test") is None
    assert EP.verify_webhook(body, "", "ephook_test") is None
    assert EP.verify_webhook(body, "x", "") is None
