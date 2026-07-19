"""Tests for opt-in Stripe shipping-address collection on physical checkouts
(dashboard.stripe_pay) and the shipping_details -> order-address mapping used to
enrich retail/reorder orders after payment."""


class _Resp:
    def __init__(self, j):
        self._j = j

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


def test_checkout_params_collect_shipping_true_adds_param():
    from dashboard.stripe_pay import _checkout_params
    p = _checkout_params(50000, customer_email="d@x.com", description="Order #1042",
                         metadata={"invoice_id": "INV1"},
                         success_url="https://s/ok", cancel_url="https://s/no",
                         collect_shipping=True)
    assert p["shipping_address_collection[allowed_countries][0]"] == "US"


def test_checkout_params_default_false_omits_param():
    from dashboard.stripe_pay import _checkout_params
    p = _checkout_params(50000, customer_email="d@x.com", description="Order #1042",
                         metadata={"invoice_id": "INV1"},
                         success_url="https://s/ok", cancel_url="https://s/no")
    assert "shipping_address_collection[allowed_countries][0]" not in p


def test_create_checkout_session_collect_shipping_true(monkeypatch):
    from dashboard import stripe_pay as S
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    captured = {}

    def _post(url, data=None, auth=None, timeout=None):
        captured["data"] = data
        return _Resp({"id": "cs_1", "url": "https://checkout.stripe.com/cs_1"})

    monkeypatch.setattr(S.requests, "post", _post)
    S.create_checkout_session(
        5000, customer_email="a@b.com", description="Remedy Match order #1",
        metadata={"invoice_id": "INV1", "kind": "retail"},
        success_url="https://s/ok", cancel_url="https://s/no",
        collect_shipping=True)
    assert captured["data"]["shipping_address_collection[allowed_countries][0]"] == "US"


def test_create_checkout_session_default_no_shipping_collection(monkeypatch):
    from dashboard import stripe_pay as S
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    captured = {}

    def _post(url, data=None, auth=None, timeout=None):
        captured["data"] = data
        return _Resp({"id": "cs_1", "url": "https://checkout.stripe.com/cs_1"})

    monkeypatch.setattr(S.requests, "post", _post)
    S.create_checkout_session(
        5000, customer_email="a@b.com", description="Some service checkout",
        metadata={"invoice_id": "INV1"},
        success_url="https://s/ok", cancel_url="https://s/no")
    assert "shipping_address_collection[allowed_countries][0]" not in captured["data"]


def test_get_session_includes_shipping_details(monkeypatch):
    from dashboard import stripe_pay as S
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setattr(
        S.requests, "get",
        lambda url, auth=None, timeout=None: _Resp({
            "id": "cs_1", "payment_status": "paid", "amount_total": 5000,
            "metadata": {"invoice_id": "9"}, "payment_intent": "pi_9",
            "shipping_details": {
                "name": "Jane Buyer",
                "address": {"line1": "123 Main St", "line2": "Apt 4",
                            "city": "Honolulu", "state": "HI",
                            "postal_code": "96815", "country": "US"}}}))
    sess = S.get_session("cs_1")
    assert sess["shipping_details"]["name"] == "Jane Buyer"
    assert sess["shipping_details"]["address"]["line1"] == "123 Main St"


def test_shipping_details_to_address_maps_fields():
    from dashboard.stripe_pay import shipping_details_to_address
    sd = {
        "name": "Jane Buyer",
        "address": {"line1": "123 Main St", "line2": "Apt 4",
                    "city": "Honolulu", "state": "HI",
                    "postal_code": "96815", "country": "US"},
    }
    addr = shipping_details_to_address(sd)
    assert addr == {
        "name": "Jane Buyer", "street": "123 Main St", "address2": "Apt 4",
        "city": "Honolulu", "state": "HI", "zip": "96815", "country": "US",
    }


def test_shipping_details_to_address_none_returns_empty():
    from dashboard.stripe_pay import shipping_details_to_address
    assert shipping_details_to_address(None) == {}


def test_shipping_details_to_address_missing_returns_empty():
    from dashboard.stripe_pay import shipping_details_to_address
    # No shipping collected on this checkout (e.g. service checkouts) -- Stripe
    # returns shipping_details: null, and a session dict may simply omit the key.
    assert shipping_details_to_address({}) == {}
    assert shipping_details_to_address({"address": {}}) == {}
    # No line1 (address present but street missing) -- still treated as absent.
    assert shipping_details_to_address({"name": "X", "address": {"city": "Honolulu"}}) == {}
