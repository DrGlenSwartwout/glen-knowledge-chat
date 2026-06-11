"""Project B — automated GET tax on invoices.

Covers the pure tax module, the create_invoice override injection, ship-to
normalization, the AST diagnostic guard, and the retail checkout routing tax by
ship-to state. Tax is config-gated (TAX_ENABLED) and ships disabled.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable in this env: {e}")


def _enable_tax(monkeypatch, retail="0.045", wholesale="0.005"):
    monkeypatch.setenv("TAX_ENABLED", "true")
    monkeypatch.setenv("GET_RETAIL_RATE", retail)
    monkeypatch.setenv("GET_WHOLESALE_RATE", wholesale)


# ── compute_get_cents ─────────────────────────────────────────────────────────

def test_get_disabled_is_zero(monkeypatch):
    monkeypatch.setenv("TAX_ENABLED", "false")
    from dashboard import tax
    assert tax.compute_get_cents(10000, channel="retail", ship_to_state="HI") == 0


def test_get_hi_retail(monkeypatch):
    _enable_tax(monkeypatch)
    from dashboard import tax
    assert tax.compute_get_cents(10000, channel="retail", ship_to_state="hi") == 450


def test_get_hi_wholesale_with_resale(monkeypatch):
    _enable_tax(monkeypatch)
    from dashboard import tax
    assert tax.compute_get_cents(10000, channel="wholesale",
                                 ship_to_state="HI", resale_ok=True) == 50


def test_get_wholesale_without_resale_falls_back_to_retail(monkeypatch):
    _enable_tax(monkeypatch)
    from dashboard import tax
    assert tax.compute_get_cents(10000, channel="wholesale",
                                 ship_to_state="HI", resale_ok=False) == 450


def test_get_out_of_state_is_zero(monkeypatch):
    _enable_tax(monkeypatch)
    from dashboard import tax
    assert tax.compute_get_cents(10000, channel="retail", ship_to_state="CA") == 0
    assert tax.compute_get_cents(10000, channel="retail", ship_to_state="") == 0


# ── create_invoice injection ──────────────────────────────────────────────────

def test_create_invoice_injects_tax(monkeypatch):
    from dashboard import qbo_billing as qb
    captured = {}
    monkeypatch.setattr(qb, "_post", lambda path, body: (captured.update(body) or {"Invoice": body}))
    qb.create_invoice({"Id": "C1"}, [{"name": "X", "amount": 10.0, "qty": 1, "item_id": "I1"}],
                      tax_cents=450)
    assert captured["TxnTaxDetail"] == {"TotalTax": 4.5}
    assert captured["GlobalTaxCalculation"] == "TaxExcluded"


def test_create_invoice_no_tax_when_zero(monkeypatch):
    from dashboard import qbo_billing as qb
    captured = {}
    monkeypatch.setattr(qb, "_post", lambda path, body: (captured.update(body) or {"Invoice": body}))
    qb.create_invoice({"Id": "C1"}, [{"name": "X", "amount": 10.0, "qty": 1, "item_id": "I1"}],
                      tax_cents=0)
    assert "TxnTaxDetail" not in captured


# ── ship-to normalization + AST diagnostic ────────────────────────────────────

def test_normalize_ship_address():
    app_module = _load_app()
    out = app_module._normalize_ship_address(
        {"street": "1 Aloha St", "city": "Hilo", "state": "hi", "zip": "96720"}, fallback_name="Jo")
    assert out["state"] == "HI" and out["city"] == "Hilo" and out["name"] == "Jo"
    assert app_module._normalize_ship_address({}) == {}


def test_taxprefs_requires_console_key(monkeypatch):
    app_module = _load_app()
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "s3cret")
    from dashboard import money
    monkeypatch.setattr(money, "qb_refresh", lambda: "tok")
    monkeypatch.setattr(money, "qb_get",
                        lambda tok, path: {"Preferences": {"TaxPrefs": {"PartnerTaxEnabled": True}}})
    client = app_module.app.test_client()
    assert client.get("/admin/qbo/taxprefs").status_code == 401
    r = client.get("/admin/qbo/taxprefs?key=s3cret")
    assert r.status_code == 200
    body = r.get_json().get("data", r.get_json())
    assert body["partner_tax_enabled"] is True


# ── retail checkout routes tax by ship-to state ───────────────────────────────

def _stub_retail_checkout(app_module, monkeypatch, tmp_path):
    import sqlite3, begin_funnel
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        begin_funnel.record_unlock(cx, session_id="m", trigger="tos",
                                   email="buyer@x.com", tos=True, tos_version="v")
    monkeypatch.setattr(app_module, "_get_product",
                        lambda slug: {"name": "Test", "info_only": False, "qbo_item_id": None})
    monkeypatch.setattr(app_module, "_qty_unit_cents", lambda p, qty: 10000)
    monkeypatch.setattr(app_module, "_ingest_order", lambda **k: None)
    from dashboard import qbo_billing as qb
    seen = {}

    def _fake_invoice(cust, lines, **kw):
        seen["tax_cents"] = kw.get("tax_cents")
        return {"Id": "INV1", "SyncToken": "0", "DocNumber": "1", "TotalAmt": 100.0}

    monkeypatch.setattr(qb, "find_or_create_customer", lambda e, n: {"Id": "C1"})
    monkeypatch.setattr(qb, "create_invoice", _fake_invoice)
    monkeypatch.setattr(qb, "get_invoice_pay_link", lambda inv: "")
    return seen


def test_checkout_taxes_hawaii(monkeypatch, tmp_path):
    app_module = _load_app()
    _enable_tax(monkeypatch)
    seen = _stub_retail_checkout(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "m")
    r = client.post("/begin/checkout/test", json={
        "email": "buyer@x.com", "method": "zelle", "qty": 1,
        "address": {"street": "1 Aloha", "city": "Hilo", "state": "HI", "zip": "96720"}})
    assert r.status_code == 200
    assert seen["tax_cents"] == 450   # 10000c * 4.5%


def test_checkout_no_tax_out_of_state(monkeypatch, tmp_path):
    app_module = _load_app()
    _enable_tax(monkeypatch)
    seen = _stub_retail_checkout(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "m")
    r = client.post("/begin/checkout/test", json={
        "email": "buyer@x.com", "method": "zelle", "qty": 1,
        "address": {"street": "5 Main", "city": "Reno", "state": "NV", "zip": "89501"}})
    assert r.status_code == 200
    assert seen["tax_cents"] == 0
