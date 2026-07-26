"""Tests for dashboard.stripe_pay._checkout_params (pure form-param builder)."""

from dashboard import stripe_pay


def test_checkout_params_shape():
    from dashboard.stripe_pay import _checkout_params
    p = _checkout_params(50000, customer_email="d@x.com", description="Order #1042",
                         metadata={"invoice_id": "INV1", "customer_id": "C1", "skip": None},
                         success_url="https://s/ok", cancel_url="https://s/no")
    assert p["mode"] == "payment"
    assert p["line_items[0][price_data][currency]"] == "usd"
    assert p["line_items[0][price_data][unit_amount]"] == "50000"
    assert p["line_items[0][price_data][product_data][name]"] == "Order #1042"
    assert p["customer_email"] == "d@x.com"
    assert p["success_url"] == "https://s/ok"
    assert p["cancel_url"] == "https://s/no"
    # metadata mirrored onto the payment intent; None values skipped
    assert p["metadata[invoice_id]"] == "INV1"
    assert p["payment_intent_data[metadata][invoice_id]"] == "INV1"
    assert "metadata[skip]" not in p


def test_price_params_payment_mode():
    p = stripe_pay._price_checkout_params(
        "price_cert", mode="payment", customer_email="a@x.com",
        metadata={"kind": "course_purchase", "email": "a@x.com", "product": "onetime"},
        success_url="https://c/s", cancel_url="https://c/c")
    assert p["mode"] == "payment"
    assert p["line_items[0][price]"] == "price_cert"
    assert p["line_items[0][quantity]"] == "1"
    assert p["customer_email"] == "a@x.com"
    assert p["metadata[kind]"] == "course_purchase"
    assert not any(k.startswith("subscription_data") for k in p)


def test_price_params_subscription_metadata():
    p = stripe_pay._price_checkout_params(
        "price_mem", mode="subscription", customer_email="",
        metadata={"kind": "course_purchase", "product": "membership"},
        success_url="https://c/s", cancel_url="https://c/c",
        subscription_metadata={"kind": "course_membership", "email": "m@x.com"})
    assert p["mode"] == "subscription"
    assert p["subscription_data[metadata][kind]"] == "course_membership"
    assert p["subscription_data[metadata][email]"] == "m@x.com"
    assert "customer_email" not in p  # empty email omitted


def test_create_price_checkout_session_posts_and_maps(monkeypatch):
    seen = {}
    monkeypatch.setattr(stripe_pay, "_post",
                        lambda path, params: seen.update(path=path, params=params) or
                        {"id": "cs_1", "url": "https://stripe/cs_1"})
    out = stripe_pay.create_price_checkout_session(
        "price_x", mode="payment", customer_email="a@x.com", metadata={"kind": "course_purchase"},
        success_url="https://c/s", cancel_url="https://c/c")
    assert out == {"id": "cs_1", "url": "https://stripe/cs_1"}
    assert seen["path"] == "/checkout/sessions"


def test_get_subscription_maps(monkeypatch):
    monkeypatch.setattr(stripe_pay, "_get", lambda path: {
        "id": "sub_1", "status": "active", "current_period_end": 1700000000,
        "customer": "cus_1", "metadata": {"kind": "course_membership"}})
    out = stripe_pay.get_subscription("sub_1")
    assert out["current_period_end"] == 1700000000
    assert out["metadata"]["kind"] == "course_membership"


def test_get_subscription_period_end_falls_back_to_item(monkeypatch):
    # Stripe API 2025-03-31.basil+: no top-level current_period_end; it lives on the item.
    monkeypatch.setattr(stripe_pay, "_get", lambda path: {
        "id": "sub_2", "status": "active", "customer": "cus_2",
        "items": {"data": [{"current_period_end": 1800000000}]},
        "metadata": {"kind": "course_membership"}})
    out = stripe_pay.get_subscription("sub_2")
    assert out["current_period_end"] == 1800000000  # fell back to items.data[0], not None
