# tests/test_stripe_vault.py
from dashboard import stripe_pay


class _Resp:
    def __init__(self, d): self._d = d
    def json(self): return self._d


def test_checkout_session_save_card_params(monkeypatch):
    captured = {}
    def fake_post(path, params):           # match the real helper's (path, params) shape
        captured["path"] = path; captured["params"] = params
        return {"id": "cs_1", "url": "https://stripe/x"}
    monkeypatch.setattr(stripe_pay, "_post", fake_post)
    stripe_pay.create_checkout_session(
        7000, customer_email="a@x.com", description="d", metadata={"k": "v"},
        success_url="s", cancel_url="c", save_card=True)
    p = captured["params"]
    assert p["mode"] == "payment"
    assert p["customer_creation"] == "always"
    assert p["payment_intent_data[setup_future_usage]"] == "off_session"


def test_charge_off_session_params(monkeypatch):
    captured = {}
    def fake_post(path, params):
        captured["path"] = path; captured["params"] = params
        return {"id": "pi_1", "status": "succeeded"}
    monkeypatch.setattr(stripe_pay, "_post", fake_post)
    out = stripe_pay.charge_off_session("cus_1", "pm_1", 5000,
                                        description="cycle", metadata={"sub": "9"})
    assert captured["path"].endswith("/payment_intents")
    assert captured["params"]["off_session"] == "true"
    assert captured["params"]["confirm"] == "true"
    assert captured["params"]["customer"] == "cus_1"
    assert captured["params"]["payment_method"] == "pm_1"
    assert out["status"] == "succeeded"


def test_charge_off_session_card_declined(monkeypatch):
    def fake_post(path, params):
        return {"error": {"type": "card_error", "code": "card_declined",
                          "decline_code": "insufficient_funds"}}
    monkeypatch.setattr(stripe_pay, "_post", fake_post)
    out = stripe_pay.charge_off_session("cus_1", "pm_1", 5000, description="x", metadata={})
    assert out["status"] == "failed"
    assert out["decline_code"] == "insufficient_funds"
