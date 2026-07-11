"""The GrooveKart webhook must read GrooveKart's ACTUAL field names.

GrooveKart (PrestaShop) sends flat customer_* fields + delivery/invoice blocks +
a products[] list — NOT customer.email / line_items. The old handler read the
wrong names, so every order 400'd on "No email in payload". These tests pin the
mapping against a REAL captured order payload (tests/fixtures/).

Runs under the Doppler harness (imports app):
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 -m pytest tests/test_groovekart_webhook_mapping.py
"""
import importlib, json, sys
from pathlib import Path
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
FIXTURE = repo_root / "tests" / "fixtures" / "groovekart_order_webhook.json"

try:
    import app
except Exception as _e:  # pragma: no cover
    pytest.skip(f"app import requires real secrets: {_e}", allow_module_level=True)


def _spy(monkeypatch, tmp_path):
    """Isolate the webhook from network + DB; return capture dicts."""
    monkeypatch.setattr(app, "LOG_DB", str(tmp_path / "chat_log.db"))
    cap = {"onboard": None, "puts": [], "posts": [], "orders": []}
    monkeypatch.setattr(app, "ghl_onboard_contact",
                        lambda **k: (cap.__setitem__("onboard", k), {"contact_id": "C1"})[1])
    monkeypatch.setattr(app, "_ghl_put", lambda path, body: cap["puts"].append((path, body)))
    monkeypatch.setattr(app, "_ghl_post", lambda path, body: cap["posts"].append((path, body)))
    monkeypatch.setattr(app, "_log_inbound_lead", lambda *a, **k: None)
    monkeypatch.setattr(app, "_attribute_conversion_by_email", lambda *a, **k: None)
    monkeypatch.setattr(app, "_ingest_order", lambda **k: cap["orders"].append(k))
    return cap


def test_real_payload_onboards_with_correct_fields(monkeypatch, tmp_path):
    cap = _spy(monkeypatch, tmp_path)
    payload = json.loads(FIXTURE.read_text())
    r = app.app.test_client().post("/webhook/groovekart", json=payload)
    assert r.status_code == 200
    o = cap["onboard"]
    assert o["email"] == "buyer@example.com"           # customer_email, was unread (redacted fixture)
    assert o["first_name"] == "Test" and o["last_name"] == "Buyer"
    assert o["phone"] == "0000000000"                  # invoice.phone_mobile (delivery empty)
    assert o["source_tag"] == "source:gk-purchase"
    # address PUT: delivery is empty in this digital order -> falls back to invoice
    addr = [b for p, b in cap["puts"] if p == "/contacts/C1"]
    assert addr and addr[0]["address1"] == "123 Test St" and addr[0]["postalCode"] == "00000"
    # products captured in the note AND the ingested order
    assert any("DENAS PCM 6 Manual in English" in b["body"] for _p, b in cap["posts"])
    assert cap["orders"] and cap["orders"][0]["items"] == [{"name": "DENAS PCM 6 Manual in English"}]


def test_shipping_address_preferred_over_billing(monkeypatch, tmp_path):
    cap = _spy(monkeypatch, tmp_path)
    payload = {
        "id": 9, "customer_email": "buyer@x.com",
        "customer_firstname": "Pat", "customer_lastname": "Kim",
        "invoice":  {"address": "1 Billing St", "city": "BillTown", "state_name": "NY",
                     "postcode": "10001", "phone_mobile": "111"},
        "delivery": {"address": "9 Ship Ave", "city": "ShipCity", "state_name": "CA",
                     "postcode": "90001", "phone_mobile": "222"},
        "products": [{"product_name": "Widget"}],
    }
    r = app.app.test_client().post("/webhook/groovekart", json=payload)
    assert r.status_code == 200
    assert cap["onboard"]["phone"] == "222"            # delivery phone, not billing 111
    addr = [b for p, b in cap["puts"] if p == "/contacts/C1"][0]
    assert addr["address1"] == "9 Ship Ave" and addr["state"] == "CA" and addr["postalCode"] == "90001"


def test_still_400_when_truly_no_email(monkeypatch, tmp_path):
    _spy(monkeypatch, tmp_path)
    r = app.app.test_client().post("/webhook/groovekart", json={"id": 1, "products": []})
    assert r.status_code == 400
