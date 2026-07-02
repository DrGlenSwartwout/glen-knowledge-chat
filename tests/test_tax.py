"""Project B — Hawai'i GET, absorb-and-track.

GET is computed and recorded on the order ledger (orders.get_cents) for
remittance — NOT added to the invoice. Covers the rate engine, the (dormant)
invoice override hook, the checkout recording GET on the order, the order ledger
round-trip, and the period filing report.
"""

import importlib
import sqlite3
import sys
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


# ── dormant invoice override hook still works (latent pass-through) ────────────

def test_create_invoice_override_hook(monkeypatch):
    from dashboard import qbo_billing as qb
    bodies = []
    monkeypatch.setattr(qb, "_post", lambda path, body: (bodies.append(dict(body)) or {"Invoice": body}))
    qb.create_invoice({"Id": "C1"}, [{"name": "X", "amount": 10.0, "qty": 1, "item_id": "I1"}],
                      tax_cents=450)
    assert bodies[0]["TxnTaxDetail"] == {"TotalTax": 4.5}
    qb.create_invoice({"Id": "C1"}, [{"name": "X", "amount": 10.0, "qty": 1, "item_id": "I1"}])
    assert "TxnTaxDetail" not in bodies[1]   # default: no tax on the invoice


# ── order ledger round-trip + report ──────────────────────────────────────────

def test_upsert_order_records_get_cents():
    from dashboard import orders
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    orders.init_orders_table(cx)
    oid = orders.upsert_order(cx, source="funnel", external_ref="r1", channel="retail",
                              total_cents=10000, get_cents=450, address={"state": "HI"})
    row = cx.execute("SELECT get_cents FROM orders WHERE id=?", (oid,)).fetchone()
    assert row["get_cents"] == 450


def test_get_tax_report_buckets():
    from dashboard import orders
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    orders.init_orders_table(cx)
    orders.upsert_order(cx, source="funnel", external_ref="r1", channel="retail",
                        total_cents=10000, get_cents=450, address={"state": "HI"})
    orders.upsert_order(cx, source="wholesale", external_ref="w1", channel="wholesale",
                        total_cents=20000, get_cents=100, address={"state": "HI"})
    orders.upsert_order(cx, source="funnel", external_ref="o1", channel="retail",
                        total_cents=5000, get_cents=0, address={"state": "CA"})
    orders.upsert_order(cx, source="funnel", external_ref="u1", channel="retail",
                        total_cents=3000, get_cents=0, address={})
    rep = orders.get_tax_report(cx, date_from="2000-01-01", date_to="2100-01-01")
    assert rep["hi_retail"] == {"orders": 1, "gross_cents": 10000, "get_cents": 450}
    assert rep["hi_wholesale"]["get_cents"] == 100
    assert rep["out_of_state"]["gross_cents"] == 5000 and rep["out_of_state"]["get_cents"] == 0
    assert rep["unknown_state"]["orders"] == 1
    assert rep["total_get_cents"] == 550


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
                        lambda tok, path: {"Preferences": {"TaxPrefs": {"PartnerTaxEnabled": False}}})
    client = app_module.app.test_client()
    assert client.get("/admin/qbo/taxprefs").status_code == 401
    assert client.get("/admin/qbo/taxprefs?key=s3cret").status_code == 200


def test_get_report_requires_console_key(monkeypatch, tmp_path):
    app_module = _load_app()
    import dashboard
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "s3cret")
    client = app_module.app.test_client()
    assert client.get("/admin/tax/get-report?from=2026-01-01&to=2026-12-31").status_code == 401
    r = client.get("/admin/tax/get-report?from=2026-01-01&to=2026-12-31&key=s3cret")
    assert r.status_code == 200
    assert "hi_retail" in (r.get_json().get("data") or r.get_json())


# ── retail checkout RECORDS GET on the order, does NOT tax the invoice ─────────

def _stub_retail_checkout(app_module, monkeypatch, tmp_path):
    import begin_funnel
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        begin_funnel.record_unlock(cx, session_id="m", trigger="tos",
                                   email="buyer@x.com", tos=True, tos_version="v")
    monkeypatch.setattr(app_module, "_get_product",
                        lambda slug: {"slug": "test", "name": "Test", "info_only": False,
                                      "qbo_item_id": None, "price_cents": 10000})
    monkeypatch.setattr(app_module._shipping, "quote", lambda b: {"shipping_cents": 0})
    seen = {}
    monkeypatch.setattr(app_module, "_ingest_order",
                        lambda **k: seen.update({"order_get_cents": k.get("get_cents")}))
    from dashboard import qbo_billing as qb

    def _fake_invoice(cust, lines, **kw):
        seen["invoice_tax_cents"] = kw.get("tax_cents")   # must be None — no invoice tax
        return {"Id": "INV1", "SyncToken": "0", "DocNumber": "1", "TotalAmt": 100.0}

    monkeypatch.setattr(qb, "find_or_create_customer", lambda e, n: {"Id": "C1"})
    monkeypatch.setattr(qb, "create_invoice", _fake_invoice)
    monkeypatch.setattr(qb, "get_invoice_pay_link", lambda inv: "")
    return seen


def test_checkout_records_get_on_order_not_invoice(monkeypatch, tmp_path):
    app_module = _load_app()
    _enable_tax(monkeypatch)
    seen = _stub_retail_checkout(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "m")
    r = client.post("/begin/checkout/test", json={
        "email": "buyer@x.com", "method": "zelle", "qty": 1,
        "address": {"street": "1 Aloha", "city": "Hilo", "state": "HI", "zip": "96720"}})
    assert r.status_code == 200
    assert seen["order_get_cents"] == 450          # recorded on the order
    assert seen["invoice_tax_cents"] is None        # NOT charged on the invoice


def test_checkout_no_get_out_of_state(monkeypatch, tmp_path):
    app_module = _load_app()
    _enable_tax(monkeypatch)
    seen = _stub_retail_checkout(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "m")
    r = client.post("/begin/checkout/test", json={
        "email": "buyer@x.com", "method": "zelle", "qty": 1,
        "address": {"street": "5 Main", "city": "Reno", "state": "NV", "zip": "89501"}})
    assert r.status_code == 200
    assert seen["order_get_cents"] == 0
