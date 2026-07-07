"""Client fee (biofield-analysis courtesy) helpers for the local Intake app.

Pure functions + thin best-effort network calls to the prod console
`/api/console/client-prices` endpoint. The panel on /author/<id> uses these to
see and set a client's courtesy price; prod `client_prices` stays the single
source the pricer reads. Never raises into the page — network failures degrade
to an "unavailable" state.
"""
import json as _json
import os
import urllib.parse
import urllib.request

BIOFIELD_SLUG = "biofield-analysis"
STANDARD_CENTS = 30000   # mirrors the prod biofield-analysis product ($300 charge)
VALUE_CENTS = 99700      # mirrors the product's stated value ($997)


def dollars_to_cents(v):
    """Dollars (str/int/float) -> integer cents. Rejects negatives and non-numbers."""
    if v is None:
        raise ValueError("amount required")
    try:
        d = float(str(v).strip())
    except (TypeError, ValueError):
        raise ValueError("invalid amount")
    if d < 0:
        raise ValueError("amount must be non-negative")
    return int(round(d * 100))


def cents_to_dollars(cents):
    """Integer cents -> a display dollar string ('300', '697.50')."""
    cents = int(cents)
    return f"{cents // 100}" if cents % 100 == 0 else f"{cents / 100:.2f}"


def parse_courtesy(resp):
    """Pull the biofield-analysis entry out of a client-prices GET response."""
    for row in (resp or {}).get("prices", []) or []:
        if row.get("slug") == BIOFIELD_SLUG:
            return {"courtesy_cents": int(row["price_cents"]), "note": row.get("note") or ""}
    return {"courtesy_cents": None, "note": ""}


def build_fee_state(email, fee_get):
    """The state the panel renders from. Only calls fee_get when an email exists."""
    email = (email or "").strip()
    state = {"email": email, "has_email": bool(email), "available": False,
             "courtesy_cents": None, "note": "",
             "standard_cents": STANDARD_CENTS, "value_cents": VALUE_CENTS}
    if not email:
        return state
    got = fee_get(email) or {}
    state["available"] = bool(got.get("available"))
    state["courtesy_cents"] = got.get("courtesy_cents")
    state["note"] = got.get("note") or ""
    return state
