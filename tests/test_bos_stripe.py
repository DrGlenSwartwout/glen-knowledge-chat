import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


class _Resp:
    def __init__(self, j): self._j = j
    def raise_for_status(self): pass
    def json(self): return self._j


def test_refund_full_and_partial(monkeypatch):
    from dashboard import stripe_pay as S
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    captured = {}
    def _post(url, data=None, auth=None, timeout=None):
        captured["url"] = url; captured["data"] = data
        return _Resp({"id": "re_1", "status": "succeeded", "amount": data.get("amount", 0)})
    monkeypatch.setattr(S.requests, "post", _post)
    # full refund: no amount
    r = S.refund("pi_123")
    assert captured["url"].endswith("/v1/refunds")
    assert captured["data"]["payment_intent"] == "pi_123" and "amount" not in captured["data"]
    assert r["status"] == "succeeded"
    # partial refund: amount in cents
    S.refund("pi_123", amount_cents=2500)
    assert captured["data"]["amount"] == 2500


def test_get_session_includes_payment_intent(monkeypatch):
    from dashboard import stripe_pay as S
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setattr(S.requests, "get",
                        lambda url, auth=None, timeout=None: _Resp(
                            {"id": "cs_1", "payment_status": "paid", "amount_total": 7000,
                             "metadata": {"invoice_id": "9"}, "payment_intent": "pi_9"}))
    sess = S.get_session("cs_1")
    assert sess["payment_intent"] == "pi_9"
    assert sess["payment_status"] == "paid"
