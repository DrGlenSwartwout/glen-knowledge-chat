"""Tests for dashboard.dropship_checkout — drop-ship wholesale pricing + order building.

QBO and wallet are stubbed using the same pattern as test_wholesale_checkout.py.
"""

from dashboard import dropship_checkout as dc


# ── dropship_line_cents ───────────────────────────────────────────────────────

def test_dropship_unit_price_is_base_plus_retail_fee():
    # base from blended curve; fee = 33% of (retail - base); drop-ship unit = base + fee
    # 1 bottle, uncertified: base $50.00, retail $70.00 -> fee 33%*(7000-5000)=660 -> unit 5660
    line = dc.dropship_line_cents(retail_cents=7000, qty=1, modules=0,
                                  settings=dc._settings())
    assert line["base_cents"] == 5000
    assert line["fee_cents"] == 660
    assert line["unit_cents"] == 5660          # what the practitioner pays per bottle
    assert line["line_cents"] == 5660          # x qty 1


def test_dropship_unit_uses_blended_volume_and_cert():
    # 12 bottles, fully certified: base $42.76, retail $70 -> fee 33%*(7000-4276)=899 -> 5175
    line = dc.dropship_line_cents(retail_cents=7000, qty=12, modules=12,
                                  settings=dc._settings())
    assert line["base_cents"] == 4276
    assert line["fee_cents"] == 899
    assert line["unit_cents"] == 5175
    assert line["line_cents"] == 5175 * 12


def test_dropship_fee_zero_when_retail_equals_base():
    line = dc.dropship_line_cents(retail_cents=5000, qty=1, modules=0, settings=dc._settings())
    assert line["fee_cents"] == 0 and line["unit_cents"] == 5000


# ── build_dropship_order ──────────────────────────────────────────────────────

def test_build_dropship_order_invoices_practitioner_ships_patient(monkeypatch):
    cart = [{"slug": "brain-boost", "qty": 6}]
    prac = {"id": "p1", "modules_completed": 0, "email": "doc@x.com", "name": "Doc"}
    patient_ship = {"name": "Pat", "state": "CA", "country": "US", "address1": "1 St"}

    # stubs
    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)

    cap = {}
    monkeypatch.setattr(dc.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(dc.qb, "create_invoice",
        lambda cust, lines, **k: cap.update(lines=lines, kw=k) or
        {"Id": "INV", "SyncToken": "0", "DocNumber": "1001", "TotalAmt": 339.60})

    # stub wallet — no balance to redeem
    import dashboard.wallet as _wallet
    monkeypatch.setattr(_wallet, "redeem_for_order", lambda pid, total, inv: 0)
    monkeypatch.setattr(_wallet, "earn_fee_free", lambda pid, charged, inv: 0)

    # stub tax
    import dashboard.tax as _tax
    monkeypatch.setattr(_tax, "compute_get_cents",
        lambda subtotal, *, channel, ship_to_state, resale_ok=False: 0)

    out = dc.build_dropship_order(cart, prac, patient_ship=patient_ship, method="zelle")

    assert out["ok"] is True
    assert out["ship_to"]["name"] == "Pat"            # ships to the PATIENT
    assert out["source"] == "dropship"
    # 6 bottles uncertified: invoice carries unit qty
    assert cap["lines"][0]["qty"] == 6
