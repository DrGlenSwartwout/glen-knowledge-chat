import dashboard.stripe_pay as sp


def test_create_setup_session_attaches_new_customer(monkeypatch):
    # A setup-mode Checkout Session only saves the card to a Customer when one is
    # supplied; without it the vaulted card can't be charged off-session later
    # (founding on-ship, studio bridge). So the session MUST be bound to a customer.
    monkeypatch.setattr(sp, "_get", lambda path: {"data": []})   # no existing customer
    posts = []
    def fake_post(path, params):
        posts.append((path, params))
        return {"id": "cus_new"} if path == "/customers" else {"id": "cs_1", "url": "https://stripe/x"}
    monkeypatch.setattr(sp, "_post", fake_post)

    out = sp.create_setup_session(customer_email="p@x.com",
                                  metadata={"kind": "studio_bridge", "email": "p@x.com"},
                                  success_url="https://h/return", cancel_url="https://h/cancel")

    assert out["url"].startswith("https://stripe/")
    # a customer was created for the email
    assert ("/customers", {"email": "p@x.com"}) in posts
    # the session is mode=setup and bound to that customer
    sess = next(p for p in posts if p[0] == "/checkout/sessions")[1]
    assert sess["mode"] == "setup"
    assert sess["customer"] == "cus_new"
    assert "customer_email" not in sess          # can't pass both customer + customer_email
    assert sess["success_url"] == "https://h/return"
    assert sess["cancel_url"] == "https://h/cancel"
    assert sess["metadata[kind]"] == "studio_bridge"
    assert sess["metadata[email]"] == "p@x.com"


def test_create_setup_session_reuses_existing_customer(monkeypatch):
    # Don't pile up duplicate Stripe customers when the same email reserves again.
    monkeypatch.setattr(sp, "_get", lambda path: {"data": [{"id": "cus_existing"}]})
    posts = []
    monkeypatch.setattr(sp, "_post",
        lambda path, params: posts.append((path, params)) or {"id": "cs", "url": "https://stripe/y"})

    sp.create_setup_session(customer_email="p@x.com", metadata={},
                            success_url="s", cancel_url="c")

    assert all(p[0] != "/customers" for p in posts)   # no new customer created
    sess = next(p for p in posts if p[0] == "/checkout/sessions")[1]
    assert sess["customer"] == "cus_existing"


def test_get_setup_intent(monkeypatch):
    monkeypatch.setattr(sp, "_get", lambda path: {"id": "si_1", "customer": "cus_1", "payment_method": "pm_1"} if path == "/setup_intents/si_1" else {})
    si = sp.get_setup_intent("si_1")
    assert si["customer"] == "cus_1" and si["payment_method"] == "pm_1"
