"""Tests for dashboard.dropship_checkout — patient-paid (dispensary) client order.

Stubs follow the same pattern as test_dropship_checkout.py: monkeypatch
qb.find_or_create_customer, qb.create_invoice, and tax.compute_get_cents.

Margin math (1 bottle, S=$70, uncertified):
  base = blended_unit_price_cents(qty=1, modules=0) = $50.00 (5000 cents)
  fee  = 33% * (7000 - 5000)                       = $6.60  (660 cents)
  margin = 7000 - 5000 - 660                        = $13.40 (1340 cents)
"""

from dashboard import dropship_checkout as dc


# ── practitioner_price_for ───────────────────────────────────────────────────

def test_practitioner_price_for_defaults_to_retail(monkeypatch):
    """When no stored price exists, practitioner_price_for returns the retail price."""
    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    # Stub the internal price lookup to always return retail (default stub behaviour)
    monkeypatch.setattr(dc, "_practitioner_price_cents", lambda pid, slug, retail: retail)
    assert dc.practitioner_price_for("p1", "brain-boost") == 7000


# ── build_client_order ───────────────────────────────────────────────────────

def test_build_client_order_charges_patient_credits_margin(monkeypatch):
    """1 bottle @ S=$70: patient is charged $70, margin = $13.40 (1340 cents)."""
    cart = [{"slug": "brain-boost", "qty": 1}]
    prac = {"id": "p1", "modules_completed": 0, "dispensary_code": "abc"}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}

    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    # S = $70 retail (practitioner hasn't overridden it)
    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: 7000)

    monkeypatch.setattr(dc.qb, "find_or_create_customer",
                        lambda *a, **k: {"Id": "PATC"})
    cap = {}
    monkeypatch.setattr(dc.qb, "create_invoice",
        lambda cust, lines, **k: cap.update(cust=cust, lines=lines) or
        {"Id": "INV", "TotalAmt": 70.0})

    import dashboard.tax as _tax
    monkeypatch.setattr(_tax, "compute_get_cents",
        lambda s, *, channel, ship_to_state, resale_ok=False: 0)

    out = dc.build_client_order(cart, prac, patient=patient, method="card")

    assert out["ok"] is True
    assert out["source"] == "dispensary"
    assert out["customer_id"] == "PATC"          # the PATIENT is the QBO customer
    assert out["ship_to"]["name"] == "Pat"        # ships to the patient
    # 1 bottle @ S=$70, base $50, fee 33%*(7000-5000)=660 -> margin 1340
    assert out["margin_cents"] == 1340
    assert cap["lines"][0]["amount"] == 70.0      # patient is charged S, not wholesale


def test_build_client_order_empty_cart_rejected(monkeypatch):
    """Empty cart and zero-qty cart are both rejected."""
    prac = {"id": "p1", "modules_completed": 0}
    patient = {"email": "pat@x.com", "ship": {"state": "CA", "country": "US"}}
    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: 7000)
    monkeypatch.setattr(dc.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C"})
    monkeypatch.setattr(dc.qb, "create_invoice",
        lambda *a, **k: {"Id": "I", "TotalAmt": 0.0})
    import dashboard.tax as _tax
    monkeypatch.setattr(_tax, "compute_get_cents",
        lambda s, *, channel, ship_to_state, resale_ok=False: 0)

    assert dc.build_client_order([], prac, patient=patient, method="card")["ok"] is False
    assert dc.build_client_order([{"slug": "a", "qty": 0}], prac,
                                  patient=patient, method="card")["ok"] is False


# ── _practitioner_price_cents — real settings store ──────────────────────────

def test_practitioner_price_cents_uses_stored_markup(tmp_path, monkeypatch):
    """_practitioner_price_cents reads a stored 20 % markup: 7000 * 1.20 = 8400."""
    import sqlite3
    from dashboard import practitioner_settings as ps
    from dashboard import dropship_checkout as dc

    db_path = str(tmp_path / "chat_log.db")
    # Seed the settings DB with a 20% markup for practitioner "p1".
    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row
    ps.init_settings_table(cx)
    ps.set_pricing(cx, "p1", {"default_markup_pct": 20, "overrides": {}})
    cx.close()

    monkeypatch.setattr(dc, "_LOG_DB", db_path)
    assert dc._practitioner_price_cents("p1", "brain-boost", 7000) == 8400


def test_practitioner_price_cents_no_settings_returns_retail(tmp_path, monkeypatch):
    """With no stored settings, _practitioner_price_cents returns retail (≥ MAP)."""
    import sqlite3
    from dashboard import practitioner_settings as ps
    from dashboard import dropship_checkout as dc

    db_path = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row
    ps.init_settings_table(cx)
    cx.close()

    monkeypatch.setattr(dc, "_LOG_DB", db_path)
    # No settings row → markup=0 → price = retail (7000) → clamped to max(7000, 6700)
    assert dc._practitioner_price_cents("p1", "brain-boost", 7000) == 7000


def test_practitioner_price_cents_fallback_on_error(monkeypatch):
    """Any DB error falls back to max(retail, MAP) without raising."""
    from dashboard import dropship_checkout as dc

    monkeypatch.setattr(dc, "_LOG_DB", "/nonexistent/path/chat_log.db")
    # Fallback: max(7000, 6700) = 7000
    result = dc._practitioner_price_cents("p1", "brain-boost", 7000)
    assert result == 7000


def test_build_client_order_get_recorded_not_charged(monkeypatch):
    """GET comes back in the result but is never added to invoice lines."""
    cart = [{"slug": "brain-boost", "qty": 1}]
    prac = {"id": "p1", "modules_completed": 0}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "HI", "country": "US"}}

    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: 7000)
    monkeypatch.setattr(dc.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C"})
    cap = {}
    monkeypatch.setattr(dc.qb, "create_invoice",
        lambda cust, lines, **k: cap.update(lines=lines) or
        {"Id": "INV", "TotalAmt": 70.0})
    import dashboard.tax as _tax
    monkeypatch.setattr(_tax, "compute_get_cents",
        lambda s, *, channel, ship_to_state, resale_ok=False: 400)

    out = dc.build_client_order(cart, prac, patient=patient, method="card")

    assert out["get_cents"] == 400
    names = " ".join(
        l.get("name", "") + l.get("description", "") for l in cap["lines"]
    ).lower()
    assert "tax" not in names and "get" not in names   # no GET line on the invoice
