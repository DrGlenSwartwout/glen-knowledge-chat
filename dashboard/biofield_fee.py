"""Client fee (biofield-analysis courtesy) helpers for the local Intake app.

Pure functions + thin best-effort network calls to the prod console
`/api/console/client-prices` endpoint. The panel on /author/<id> uses these to
see and set a client's courtesy price; prod `client_prices` stays the single
source the pricer reads. Never raises into the page: network failures degrade
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


def _console():
    """(base_url, key) or (None, None) when no CONSOLE_SECRET is set."""
    key = os.environ.get("CONSOLE_SECRET")
    if not key:
        return None, None
    base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
    return base, key


def _request(method, base, key, body=None):
    """POST/DELETE to the client-prices endpoint with a JSON body. Returns parsed JSON."""
    url = f"{base}/api/console/client-prices?key=" + urllib.parse.quote(key)
    data = _json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"X-Console-Key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return _json.loads(r.read().decode() or "{}")


def default_fee_get(email):
    email = (email or "").strip().lower()
    base, key = _console()
    if not email or not base:
        return {"available": False, "courtesy_cents": None, "note": ""}
    try:
        url = (f"{base}/api/console/client-prices?key=" + urllib.parse.quote(key)
               + "&email=" + urllib.parse.quote(email))
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        # Page-load path: keep this snappy, don't let a slow/unreachable prod stall the page.
        with urllib.request.urlopen(req, timeout=6) as r:
            resp = _json.loads(r.read().decode() or "{}")
        out = parse_courtesy(resp)
        out["available"] = True
        return out
    except Exception:
        return {"available": False, "courtesy_cents": None, "note": ""}


def default_fee_set(email, cents, note=""):
    email = (email or "").strip().lower()
    base, key = _console()
    if not email or not base:
        return {"ok": False}
    try:
        resp = _request("POST", base, key, {"email": email, "slug": BIOFIELD_SLUG,
                                            "price_cents": int(cents), "note": note or ""})
        return {"ok": bool(resp.get("ok", True))}
    except Exception:
        return {"ok": False}


def default_fee_clear(email):
    email = (email or "").strip().lower()
    base, key = _console()
    if not email or not base:
        return {"ok": False}
    try:
        resp = _request("DELETE", base, key, {"email": email, "slug": BIOFIELD_SLUG})
        return {"ok": bool(resp.get("ok", True))}
    except Exception:
        return {"ok": False}
