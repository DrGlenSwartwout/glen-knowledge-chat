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


def _post(path: str, params: dict) -> dict:
    """POST form-encoded params to the Stripe API. Returns the parsed JSON dict."""
    url = path if path.startswith("http") else f"{STRIPE_API}{path}"
    r = requests.post(url, data=params, auth=(_key(), ""), timeout=20)
    r.raise_for_status()
    return r.json()


def _get(path: str) -> dict:
    """GET from the Stripe API. Returns the parsed JSON dict."""
    url = path if path.startswith("http") else f"{STRIPE_API}{path}"
    r = requests.get(url, auth=(_key(), ""), timeout=20)
    r.raise_for_status()
    return r.json()


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
                            success_url, cancel_url, save_card=False) -> dict:
    params = _checkout_params(amount_cents, customer_email=customer_email,
                              description=description, metadata=metadata,
                              success_url=success_url, cancel_url=cancel_url)
    if save_card:
        params["customer_creation"] = "always"
        params["payment_intent_data[setup_future_usage]"] = "off_session"
    j = _post("/checkout/sessions", params)
    return {"id": j.get("id"), "url": j.get("url")}


def _find_or_create_customer(email: str) -> str:
    """Find a Stripe Customer by email, or create one. Returns the customer id ("" on
    failure). A setup-mode Checkout Session only saves the card to a Customer when one
    is supplied, so off-session charges later (founding on-ship, studio bridge) need it."""
    email = (email or "").strip()
    if email:
        try:
            from urllib.parse import quote
            res = _get(f"/customers?email={quote(email)}&limit=1")
            data = res.get("data") or []
            if data and data[0].get("id"):
                return data[0]["id"]
        except Exception:
            pass
    j = _post("/customers", {"email": email} if email else {})
    return j.get("id") or ""


def create_setup_session(*, customer_email, metadata, success_url, cancel_url) -> dict:
    """Stripe Checkout in mode=setup — vaults a card with NO charge (for later
    off-session use). Binds the session to a Customer so the saved card can be
    charged off-session later; without a customer Stripe leaves SetupIntent.customer
    null and the on-ship charge has nothing to bill. Returns {id, url}."""
    customer_id = _find_or_create_customer(customer_email)
    params = {
        "mode": "setup",
        "success_url": success_url,
        "cancel_url": cancel_url,
        "payment_method_types[0]": "card",
    }
    if customer_id:
        params["customer"] = customer_id          # can't pass both customer + customer_email
    elif customer_email:
        params["customer_email"] = customer_email
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        params[f"metadata[{k}]"] = str(v)
    j = _post("/checkout/sessions", params)
    return {"id": j.get("id"), "url": j.get("url")}


def get_setup_intent(si_id: str) -> dict:
    """Retrieve a SetupIntent. Returns {id, customer, payment_method, status}."""
    j = _get(f"/setup_intents/{si_id}")
    return {"id": j.get("id"), "customer": j.get("customer"),
            "payment_method": j.get("payment_method"), "status": j.get("status")}


def get_session(session_id) -> dict:
    j = _get(f"/checkout/sessions/{session_id}")
    return {"id": j.get("id"), "payment_status": j.get("payment_status"),
            "amount_total": j.get("amount_total"), "metadata": j.get("metadata") or {},
            "payment_intent": j.get("payment_intent"),
            "setup_intent": j.get("setup_intent")}


def get_payment_intent(pi_id: str) -> dict:
    """Retrieve a PaymentIntent. Returns {id, customer, payment_method, status}."""
    j = _get(f"/payment_intents/{pi_id}")
    return {"id": j.get("id"), "customer": j.get("customer"),
            "payment_method": j.get("payment_method"), "status": j.get("status")}


def charge_off_session(customer_id, payment_method_id, amount_cents, *,
                       description, metadata) -> dict:
    """Charge a vaulted card off-session. Returns {id, status, decline_code?, error?}.
    status: 'succeeded' | 'requires_action' | 'failed'."""
    params = {
        "amount": str(int(amount_cents)), "currency": "usd",
        "customer": customer_id, "payment_method": payment_method_id,
        "off_session": "true", "confirm": "true", "description": description or "",
    }
    for k, v in (metadata or {}).items():
        params[f"metadata[{k}]"] = str(v)

    def _failed(err):
        return {"id": None, "status": "failed",
                "decline_code": err.get("decline_code") or err.get("code"),
                "error": err.get("message")}
    try:
        resp = _post("/payment_intents", params)
    except requests.HTTPError as e:
        # A real card decline is HTTP 402 with {"error": {...}} — _post raised before
        # returning, so parse the error off the response here instead of propagating.
        try:
            err = (e.response.json() or {}).get("error") or {}
        except Exception:
            err = {}
        return _failed(err)
    err = resp.get("error")          # defensive: some errors arrive 200-with-body
    if err:
        return _failed(err)
    return {"id": resp.get("id"), "status": resp.get("status")}


def refund(payment_intent, amount_cents=None):
    """Issue a Stripe refund against a PaymentIntent. amount_cents=None = full
    refund. Returns {id, status, amount}. Raises on a Stripe error."""
    data = {"payment_intent": str(payment_intent)}
    if amount_cents is not None:
        data["amount"] = int(amount_cents)
    j = _post("/refunds", data)
    return {"id": j.get("id"), "status": j.get("status"), "amount": j.get("amount")}


def verify_webhook(payload, sig_header, secret, tolerance=300):
    """Verify a Stripe webhook signature. payload = the raw request body (bytes or str).
    Returns the parsed event dict on success, None on any failure (bad/missing/stale
    signature, wrong secret, unparseable body). Pure; no network."""
    import hmac, hashlib, json, time
    try:
        payload_b = payload.encode("utf-8") if isinstance(payload, str) else payload
        items = dict(p.split("=", 1) for p in (sig_header or "").split(",") if "=" in p)
        ts, v1 = items.get("t"), items.get("v1")
        if not ts or not v1:
            return None
        expected = hmac.new(secret.encode("utf-8"), f"{ts}.".encode("utf-8") + payload_b,
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, v1):
            return None
        if tolerance and abs(time.time() - int(ts)) > tolerance:
            return None
        return json.loads(payload_b.decode("utf-8"))
    except Exception:
        return None
