"""Shipping-label carrier boundary.

ShipStation API is preferred when SHIPENGINE_API_KEY is present; EasyPost remains
supported for existing installations and for its tracker/webhook integration.
With neither label key, Orders falls back to USPS Click-N-Ship."""
import json
import os
import urllib.request

CLICKNSHIP_URL = "https://cns.usps.com"
_API = "https://api.easypost.com/v2"
_SHIPENGINE_API = "https://api.shipengine.com/v1"
_DEFAULT_OZ = 4  # base weight per parcel
_PER_ITEM_OZ = 4  # rough per-bottle weight
_PREDEFINED_PACKAGE_BY_BOX_SIZE = {
    # Remedy Match packs the internal Small Box inside USPS's padded flat-rate
    # envelope. The carrier-facing package must therefore be the envelope.
    "S": "FlatRatePaddedEnvelope",
    "M": "MediumFlatRateBox",
    "L": "LargeFlatRateBox",
}
_SHIPENGINE_PACKAGE_BY_BOX_SIZE = {
    "S": "flat_rate_padded_envelope",
    "M": "medium_flat_rate_box",
    "L": "large_flat_rate_box",
}

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
    """EasyPost tracker API readiness (kept separate from label readiness)."""
    return bool(os.environ.get("EASYPOST_API_KEY"))


def label_api_configured():
    """True when either supported provider can purchase a shipping label."""
    return bool(os.environ.get("SHIPENGINE_API_KEY") or os.environ.get("EASYPOST_API_KEY"))


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
    parcel = {"weight": _DEFAULT_OZ + _PER_ITEM_OZ * n_items}
    box_size = str(order.get("shipping_box_size") or order.get("box_size") or "").upper()
    predefined = _PREDEFINED_PACKAGE_BY_BOX_SIZE.get(box_size)
    if predefined:
        parcel["predefined_package"] = predefined
    return {
        "to_address": {
            "name": addr.get("name") or order.get("name") or order.get("email") or "Customer",
            "street1": addr.get("street", ""), "street2": addr.get("address2", ""),
            "city": addr.get("city", ""),
            "state": addr.get("state", ""), "zip": addr.get("zip", ""),
            "country": addr.get("country", "US"),
        },
        "from_address": {
            "name": fa.get("name", ""), "street1": fa.get("street", ""),
            "city": fa.get("city", ""), "state": fa.get("state", ""),
            "zip": fa.get("zip", ""), "country": fa.get("country", "US"),
            "phone": fa.get("phone", ""),
        },
        "parcel": parcel,
    }


def build_shipengine_label(order, from_address, carrier_id):
    """Pure: build one USPS Priority Mail label request for ShipStation API."""
    addr = order.get("address") or {}
    fa = from_address or {}
    n_items = sum(int(i.get("qty", 1) or 1) for i in (order.get("items") or [])) or 1
    box_size = str(order.get("shipping_box_size") or order.get("box_size") or "").upper()
    package_code = _SHIPENGINE_PACKAGE_BY_BOX_SIZE.get(box_size, "package")
    return {
        "shipment": {
            "carrier_id": carrier_id,
            "service_code": "usps_priority_mail",
            "ship_to": {
                "name": addr.get("name") or order.get("name") or order.get("email") or "Customer",
                "address_line1": addr.get("street", ""),
                "address_line2": addr.get("address2", ""),
                "city_locality": addr.get("city", ""),
                "state_province": addr.get("state", ""),
                "postal_code": addr.get("zip", ""),
                "country_code": addr.get("country", "US"),
                "phone": order.get("phone", ""),
            },
            "ship_from": {
                "name": fa.get("name", ""),
                "address_line1": fa.get("street", ""),
                "city_locality": fa.get("city", ""),
                "state_province": fa.get("state", ""),
                "postal_code": fa.get("zip", ""),
                "country_code": fa.get("country", "US"),
                "phone": fa.get("phone", ""),
            },
            "packages": [{
                "package_code": package_code,
                "weight": {"value": _DEFAULT_OZ + _PER_ITEM_OZ * n_items, "unit": "ounce"},
            }],
        },
        "label_format": "pdf",
        "label_layout": "4x6",
        "label_download_type": "url",
    }


def _shipengine_request(path, *, data=None):
    key = os.environ.get("SHIPENGINE_API_KEY")
    if not key:
        raise RuntimeError("SHIPENGINE_API_KEY not set")
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        _SHIPENGINE_API + path, data=body, method=("POST" if body is not None else "GET"),
        headers={"API-Key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _shipengine_usps_carrier_id():
    explicit = os.environ.get("SHIPENGINE_USPS_CARRIER_ID", "").strip()
    if explicit:
        return explicit
    carriers = _shipengine_request("/carriers")
    for carrier in carriers:
        code = str(carrier.get("carrier_code") or "").lower()
        name = str(carrier.get("friendly_name") or carrier.get("carrier_name") or "").lower()
        if code in ("stamps_com", "usps") or "stamps" in name or name == "usps":
            return carrier.get("carrier_id")
    raise RuntimeError("ShipStation USPS/Stamps.com carrier not found")


def _buy_shipengine_label(order, from_address):
    payload = build_shipengine_label(order, from_address, _shipengine_usps_carrier_id())
    bought = _shipengine_request("/labels", data=payload)
    downloads = bought.get("label_download") or {}
    return {
        "tracking_number": bought.get("tracking_number", ""),
        "label_url": downloads.get("pdf") or downloads.get("href") or "",
        "estimated_delivery_date": bought.get("estimated_delivery_date"),
        "provider": "shipengine",
    }


def buy_label(order, from_address):
    """Purchase a label, preferring ShipStation API over legacy EasyPost."""
    if os.environ.get("SHIPENGINE_API_KEY"):
        return _buy_shipengine_label(order, from_address)
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


def _auth_header():
    key = os.environ.get("EASYPOST_API_KEY")
    if not key:
        raise RuntimeError("EASYPOST_API_KEY not set")
    import base64
    return "Basic " + base64.b64encode((key + ":").encode()).decode()


def create_tracker(tracking_code, carrier=None):
    """Live: register an EasyPost Tracker for a tracking number (any carrier; no
    label purchase needed). Returns {tracker_id, status}. Production-only."""
    body = {"tracker": {"tracking_code": tracking_code}}
    if carrier:
        body["tracker"]["carrier"] = carrier
    req = urllib.request.Request(_API + "/trackers", data=json.dumps(body).encode(),
                                 method="POST",
                                 headers={"Authorization": _auth_header(),
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        t = json.loads(r.read())
    return {"tracker_id": t.get("id", ""), "status": (t.get("status") or "")}


def get_tracker(tracker_id):
    """Live: fetch current tracker state (backstop poll for missed webhooks)."""
    req = urllib.request.Request(_API + "/trackers/" + tracker_id, method="GET",
                                 headers={"Authorization": _auth_header()})
    with urllib.request.urlopen(req, timeout=30) as r:
        t = json.loads(r.read())
    return {"tracker_id": t.get("id", ""), "status": (t.get("status") or ""),
            "tracking_code": t.get("tracking_code", "")}


def verify_webhook(payload, sig_header, secret):
    """Verify an EasyPost webhook HMAC. payload = raw body (bytes/str). EasyPost signs
    X-Hmac-Signature = 'hmac-sha256-hex=<hexdigest of body with the webhook secret>'.
    Returns the parsed event dict on success, None on any failure. Pure; no network."""
    import hmac as _hmac, hashlib as _hashlib, json as _json
    try:
        if not (sig_header and secret):
            return None
        payload_b = payload.encode("utf-8") if isinstance(payload, str) else payload
        provided = sig_header.split("=", 1)[1] if "=" in sig_header else sig_header
        expected = _hmac.new(secret.encode("utf-8"), payload_b, _hashlib.sha256).hexdigest()
        if not _hmac.compare_digest(expected, provided):
            return None
        return _json.loads(payload_b.decode("utf-8"))
    except Exception:
        return None
