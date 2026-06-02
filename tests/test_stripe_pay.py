"""Tests for dashboard.stripe_pay._checkout_params (pure form-param builder)."""


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
