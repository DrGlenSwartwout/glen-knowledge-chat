"""EasyPost carrier boundary, behind the EASYPOST_API_KEY feature flag.

When the key is absent (current state), the Orders module falls back to a manual
USPS Click-N-Ship handoff. When set, buy_label() purchases the lowest USPS rate
and returns the label URL + tracking number. The pure helpers are unit-tested;
the live API call is production-only."""
import json
import os
import urllib.request

CLICKNSHIP_URL = "https://cns.usps.com"
_API = "https://api.easypost.com/v2"
_DEFAULT_OZ = 4  # base weight per parcel
_PER_ITEM_OZ = 4  # rough per-bottle weight

# Ship-from defaults to Remedy Match LLC's Hilo address (the same address used in
# the email footers). EasyPost rejects a shipment with no from_address, so this is
# the fallback when the dispatch ctx supplies none. Override per-field via env.
_SHIP_FROM = {
    "name": "Remedy Match LLC",
    "street": "351 Wailuku Drive",
    "city": "Hilo",
    "state": "HI",
    "zip": "96720",
    "country": "US",
    "phone": "",
}


def is_configured():
    return bool(os.environ.get("EASYPOST_API_KEY"))


def ship_from():
    """The ship-from address. Defaults to the Remedy Match Hilo address; each field
    is overridable via SHIP_FROM_NAME / _STREET / _CITY / _STATE / _ZIP / _COUNTRY /
    _PHONE env vars (so the address can move without a code change)."""
    out = dict(_SHIP_FROM)
    for field in out:
        env = os.environ.get("SHIP_FROM_" + field.upper())
        if env:
            out[field] = env
    return out


def build_shipment(order, from_address):
    """Pure: build the EasyPost shipment payload from an order dict + the ship-from
    address. Weight is estimated from item count (ounces)."""
    addr = order.get("address") or {}
    fa = from_address or {}
    n_items = sum(int(i.get("qty", 1) or 1) for i in (order.get("items") or [])) or 1
    return {
        "to_address": {
            "name": order.get("name") or order.get("email") or "Customer",
            "street1": addr.get("street", ""), "city": addr.get("city", ""),
            "state": addr.get("state", ""), "zip": addr.get("zip", ""),
            "country": addr.get("country", "US"),
        },
        "from_address": {
            "name": fa.get("name", ""), "street1": fa.get("street", ""),
            "city": fa.get("city", ""), "state": fa.get("state", ""),
            "zip": fa.get("zip", ""), "country": fa.get("country", "US"),
            "phone": fa.get("phone", ""),
        },
        "parcel": {"weight": _DEFAULT_OZ + _PER_ITEM_OZ * n_items},
    }


def buy_label(order, from_address):
    """Live: create a shipment, buy the lowest rate. Production-only (requires
    EASYPOST_API_KEY). Returns {tracking_number, label_url} or raises."""
    key = os.environ.get("EASYPOST_API_KEY")
    if not key:
        raise RuntimeError("EASYPOST_API_KEY not set")
    import base64
    auth = base64.b64encode((key + ":").encode()).decode()
    payload = json.dumps({"shipment": build_shipment(order, from_address)}).encode()
    req = urllib.request.Request(_API + "/shipments", data=payload, method="POST",
                                 headers={"Authorization": "Basic " + auth,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        shp = json.loads(r.read())
    rates = shp.get("rates") or []
    if not rates:
        raise RuntimeError("no rates returned")
    lowest = min(rates, key=lambda x: float(x.get("rate", "9999")))
    buy = json.dumps({"rate": {"id": lowest["id"]}}).encode()
    breq = urllib.request.Request(_API + "/shipments/" + shp["id"] + "/buy", data=buy,
                                  method="POST",
                                  headers={"Authorization": "Basic " + auth,
                                           "Content-Type": "application/json"})
    with urllib.request.urlopen(breq, timeout=30) as r:
        bought = json.loads(r.read())
    return {"tracking_number": bought.get("tracking_code", ""),
            "label_url": (bought.get("postage_label") or {}).get("label_url", "")}
