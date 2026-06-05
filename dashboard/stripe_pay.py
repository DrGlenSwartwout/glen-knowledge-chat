"""Minimal Stripe Checkout client — raw HTTPS via requests (no SDK dependency),
mirroring dashboard/money.py. Stripe is the card rail on top of QBO invoices:
create a hosted Checkout Session for an amount, then verify payment on return.
"""

import os

import requests

STRIPE_API = "https://api.stripe.com/v1"


def _key() -> str:
    k = os.environ.get("STRIPE_SECRET_KEY")
    if not k:
        raise RuntimeError("STRIPE_SECRET_KEY not set")
    return k


def _checkout_params(amount_cents, *, customer_email, description, metadata,
                     success_url, cancel_url, currency="usd") -> dict:
    """Pure: build the form params for a one-time-payment Checkout Session."""
    p = {
        "mode": "payment",
        "success_url": success_url,
        "cancel_url": cancel_url,
        "line_items[0][quantity]": "1",
        "line_items[0][price_data][currency]": currency,
        "line_items[0][price_data][unit_amount]": str(int(amount_cents)),
        "line_items[0][price_data][product_data][name]": description or "Remedy Match order",
    }
    if customer_email:
        p["customer_email"] = customer_email
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        p[f"metadata[{k}]"] = str(v)
        p[f"payment_intent_data[metadata][{k}]"] = str(v)
    return p


def create_checkout_session(amount_cents, *, customer_email, description, metadata,
                            success_url, cancel_url) -> dict:
    params = _checkout_params(amount_cents, customer_email=customer_email,
                              description=description, metadata=metadata,
                              success_url=success_url, cancel_url=cancel_url)
    r = requests.post(f"{STRIPE_API}/checkout/sessions", data=params,
                      auth=(_key(), ""), timeout=20)
    r.raise_for_status()
    j = r.json()
    return {"id": j.get("id"), "url": j.get("url")}


def get_session(session_id) -> dict:
    r = requests.get(f"{STRIPE_API}/checkout/sessions/{session_id}",
                     auth=(_key(), ""), timeout=20)
    r.raise_for_status()
    j = r.json()
    return {"id": j.get("id"), "payment_status": j.get("payment_status"),
            "amount_total": j.get("amount_total"), "metadata": j.get("metadata") or {},
            "payment_intent": j.get("payment_intent")}


def refund(payment_intent, amount_cents=None):
    """Issue a Stripe refund against a PaymentIntent. amount_cents=None = full
    refund. Returns {id, status, amount}. Raises on a Stripe error."""
    data = {"payment_intent": str(payment_intent)}
    if amount_cents is not None:
        data["amount"] = int(amount_cents)
    r = requests.post(f"{STRIPE_API}/refunds", data=data, auth=(_key(), ""), timeout=20)
    r.raise_for_status()
    j = r.json()
    return {"id": j.get("id"), "status": j.get("status"), "amount": j.get("amount")}
