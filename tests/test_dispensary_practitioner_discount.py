"""Task 6 — dispensary patient orders route through the practitioner-effective
discount engine.

When `build_client_order` is called with an `effective_settings` block, the
patient RECEIVES the practitioner's best-of volume discount off S (reducing the
practitioner's margin), clamped to the house ceilings and floored via
`unit_floor_cents`. When `effective_settings=None`, behavior MUST be byte-
identical to today (flat S, unchanged margin).

Stubbing mirrors tests/test_client_order.py: monkeypatch
practitioner_price_for / qb.find_or_create_customer / qb.create_invoice /
tax.compute_get_cents so the call runs with no network. `_get_product` /
`_qty_eligible` come from `app` (lazily imported inside build_client_order);
we monkeypatch `_get_product` on the app module so the eligible-SKU is
deterministic and use the real `_qty_eligible`.
"""

from dashboard import dropship_checkout as dc
from dashboard import practitioner_pricing as ppx
from dashboard import pricing as _pricing


S = 6997  # practitioner's selling price for the SKU (>= MAP)


def _stub_qb_tax(monkeypatch, cap):
    monkeypatch.setattr(dc.qb, "find_or_create_customer", lambda *a, **k: {"Id": "PATC"})
    monkeypatch.setattr(
        dc.qb, "create_invoice",
        lambda cust, lines, **k: cap.update(cust=cust, lines=lines)
        or {"Id": "INV", "TotalAmt": 0.0},
    )
    import dashboard.tax as _tax
    monkeypatch.setattr(
        _tax, "compute_get_cents",
        lambda s, *, channel, ship_to_state, resale_ok=False: 0,
    )


def _stub_eligible_product(monkeypatch, slug="vol-sku"):
    """Make app._get_product(slug) return a volume-eligible FF; keep the real
    _qty_eligible so eligibility is judged the same way production judges it."""
    import app as appmod
    prod = {"slug": slug, "qty_pricing": True, "info_only": False, "price_cents": S}
    monkeypatch.setattr(appmod, "_get_product", lambda s: dict(prod))
    return prod


# ── 1. Baseline preserved: effective_settings=None → flat S, unchanged margin ──

def test_no_config_leaves_price_at_S(monkeypatch):
    cart = [{"slug": "vol-sku", "qty": 1}]
    prac = {"id": "p1", "modules_completed": 0, "dispensary_code": "abc"}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}

    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: S)
    cap = {}
    _stub_qb_tax(monkeypatch, cap)

    out = dc.build_client_order(cart, prac, patient=patient, method="card",
                                effective_settings=None)

    # Patient charged flat S; margin identical to today's quote_line margin.
    q = dc._pp.quote_line(selling_cents=S, qty=1, modules=0, settings=dc._settings())
    assert out["ok"] is True
    assert cap["lines"][0]["amount"] == 69.97
    assert out["subtotal_cents"] == S
    assert out["margin_cents"] == q["margin_cents"]


# ── 2. Discount applied: dialed open_total discounts the patient off S ─────────

def test_dialed_open_total_discounts_patient_off_S(monkeypatch):
    QTY = 12
    cart = [{"slug": "vol-sku", "qty": QTY}]
    prac = {"id": "p1", "modules_completed": 0, "dispensary_code": "abc"}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}

    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: S)
    prod = _stub_eligible_product(monkeypatch)
    cap = {}
    _stub_qb_tax(monkeypatch, cap)

    # Standard schedule with open_total dialed to full (same_sku left OFF).
    eff = ppx.effective_settings(
        {"standard": {"open_total": {"enabled": True, "dial": 1.0}}},
        program_member=False, settings={},
    )

    out = dc.build_client_order(cart, prac, patient=patient, method="card",
                                effective_settings=eff, program_member=False)

    # Ground the expectation entirely in the same pricing helpers the engine uses.
    exp_pct = _pricing.open_total_pct(QTY, eff)
    assert exp_pct > 0  # sanity: the dial actually produced a discount
    floor = _pricing.unit_floor_cents(prod, S, eff, "discount")
    exp_paid = _pricing.apply_discount(S, exp_pct, floor)
    assert exp_paid < S  # patient actually pays less than flat S

    q = dc._pp.quote_line(selling_cents=S, qty=QTY, modules=0, settings=dc._settings())
    exp_margin_unit = q["margin_cents"] - (S - exp_paid)

    assert cap["lines"][0]["amount"] == exp_paid / 100.0
    assert out["subtotal_cents"] == exp_paid * QTY
    # Margin drops by exactly (S - paid_unit) * qty.
    assert out["margin_cents"] == exp_margin_unit * QTY
    assert out["margin_cents"] == q["margin_cents"] * QTY - (S - exp_paid) * QTY


# ── 3. same_sku (per-line) discount also flows through ─────────────────────────

def test_dialed_same_sku_discounts_per_line(monkeypatch):
    QTY = 6
    cart = [{"slug": "vol-sku", "qty": QTY}]
    prac = {"id": "p1", "modules_completed": 0, "dispensary_code": "abc"}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}

    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: S)
    prod = _stub_eligible_product(monkeypatch)
    cap = {}
    _stub_qb_tax(monkeypatch, cap)

    eff = ppx.effective_settings(
        {"standard": {"same_sku": {"enabled": True, "dial": 1.0}}},
        program_member=False, settings={},
    )
    out = dc.build_client_order(cart, prac, patient=patient, method="card",
                                effective_settings=eff, program_member=False)

    exp_pct = _pricing.same_sku_pct(QTY, eff)
    assert exp_pct > 0
    floor = _pricing.unit_floor_cents(prod, S, eff, "discount")
    exp_paid = _pricing.apply_discount(S, exp_pct, floor)
    assert cap["lines"][0]["amount"] == exp_paid / 100.0
    assert out["subtotal_cents"] == exp_paid * QTY


# ── 4. Ineligible (non-FF) SKU gets NO discount even with a dialed config ──────

def test_ineligible_sku_not_discounted(monkeypatch):
    cart = [{"slug": "plain", "qty": 12}]
    prac = {"id": "p1", "modules_completed": 0, "dispensary_code": "abc"}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}

    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: S)
    import app as appmod
    # Not a functional formulation → _qty_eligible False.
    monkeypatch.setattr(appmod, "_get_product",
                        lambda s: {"slug": s, "qty_pricing": False, "price_cents": S})
    cap = {}
    _stub_qb_tax(monkeypatch, cap)

    eff = ppx.effective_settings(
        {"standard": {"open_total": {"enabled": True, "dial": 1.0}}},
        program_member=False, settings={},
    )
    out = dc.build_client_order(cart, prac, patient=patient, method="card",
                                effective_settings=eff, program_member=False)
    assert cap["lines"][0]["amount"] == 69.97          # flat S, no discount
    assert out["subtotal_cents"] == S * 12


# ── 5. Clamp/ceiling: dial cannot exceed the house ceiling curve ───────────────

def test_dial_clamped_to_ceiling(monkeypatch):
    # An over-dial (dial > 1) is clamped to 1.0 == the house ceiling curve, so the
    # effective pct at any qty never exceeds the ceiling.
    eff_full = ppx.effective_settings(
        {"standard": {"open_total": {"enabled": True, "dial": 1.0}}},
        program_member=False, settings={},
    )
    eff_over = ppx.effective_settings(
        {"standard": {"open_total": {"enabled": True, "dial": 2.0}}},
        program_member=False, settings={},
    )
    ceil = ppx.ceilings({})["open_total"]
    for qty in (1, 6, 12, 18, 30):
        p_full = _pricing.open_total_pct(qty, eff_full)
        p_over = _pricing.open_total_pct(qty, eff_over)
        assert p_over == p_full          # over-dial clamped to full ceiling curve
        assert p_full <= ceil            # never exceeds the house ceiling
    # Flat beyond the last anchor: at high qty the pct pins at the ceiling.
    assert _pricing.open_total_pct(30, eff_full) == ceil

    # And the discounted unit price never drops below the discount floor.
    QTY = 30
    cart = [{"slug": "vol-sku", "qty": QTY}]
    prac = {"id": "p1", "modules_completed": 0}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}
    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: S)
    prod = _stub_eligible_product(monkeypatch)
    cap = {}
    _stub_qb_tax(monkeypatch, cap)
    out = dc.build_client_order(cart, prac, patient=patient, method="card",
                                effective_settings=eff_over, program_member=False)
    floor = _pricing.unit_floor_cents(prod, S, eff_full, "discount")
    assert out["subtotal_cents"] >= floor * QTY
