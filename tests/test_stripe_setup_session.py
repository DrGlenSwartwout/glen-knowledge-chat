import dashboard.stripe_pay as sp

def test_create_setup_session(monkeypatch):
    captured = {}
    monkeypatch.setattr(sp, "_post", lambda path, params: captured.update(path=path, params=params) or {"id": "cs_1", "url": "https://stripe/x"})
    out = sp.create_setup_session(customer_email="p@x.com",
                                  metadata={"kind": "studio_bridge", "email": "p@x.com"},
                                  success_url="https://h/return", cancel_url="https://h/cancel")
    assert out["url"].startswith("https://stripe/")
    assert captured["path"] == "/checkout/sessions"
    assert captured["params"]["mode"] == "setup"
    assert captured["params"]["customer_email"] == "p@x.com"
    assert captured["params"]["success_url"] == "https://h/return"
    assert captured["params"]["cancel_url"] == "https://h/cancel"
    assert captured["params"]["metadata[kind]"] == "studio_bridge"
    assert captured["params"]["metadata[email]"] == "p@x.com"

def test_get_setup_intent(monkeypatch):
    monkeypatch.setattr(sp, "_get", lambda path: {"id": "si_1", "customer": "cus_1", "payment_method": "pm_1"} if path == "/setup_intents/si_1" else {})
    si = sp.get_setup_intent("si_1")
    assert si["customer"] == "cus_1" and si["payment_method"] == "pm_1"
