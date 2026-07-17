"""Tests for dashboard.dropship_checkout.build_client_order — patient channel-locked
points (Task 2): fee-capped redemption applied as a paid-only discount.

Paid-only (Stage 4): build_client_order creates NO QBO invoice at checkout time --
the discount now lands in the returned ``qbo_payload["discount_cents"]`` (booked
later by the return-handler), not in a create_invoice kwarg.

Stubs mirror test_client_order.py:
  - _retail_for -> 7000
  - practitioner_price_for -> 7000 (fixed S; quote_line runs for real)
  - qb.find_or_create_customer / qb.create_invoice -> must NOT be called
  - tax.compute_get_cents -> 0

Economics (S=$70, uncertified, blended base at the order qty):
  1 bottle:  base $50, fee 33%*(7000-5000)=660, margin 1340.
  Total service fee (cap) = sum(fee_cents * qty) over the cart.
"""

from dashboard import dropship_checkout as dc


def _common_stubs(monkeypatch):
    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: 7000)

    def boom(*a, **k):
        raise AssertionError("build_client_order must not touch QBO invoicing (paid-only)")
    monkeypatch.setattr(dc.qb, "find_or_create_customer", boom)
    monkeypatch.setattr(dc.qb, "create_invoice", boom)
    import dashboard.tax as _tax
    monkeypatch.setattr(_tax, "compute_get_cents",
        lambda s, *, channel, ship_to_state, resale_ok=False: 0)


def test_zero_redemption_unchanged(monkeypatch):
    """points_to_redeem_cents=0 → discount_cents=0, redeemed=0, margin intact."""
    cart = [{"slug": "brain-boost", "qty": 1}]
    prac = {"id": "p1", "modules_completed": 0}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}
    _common_stubs(monkeypatch)

    out = dc.build_client_order(cart, prac, patient=patient, method="card",
                               points_to_redeem_cents=0, points_balance_cents=0)

    assert out["ok"] is True
    assert out["points_redeemed_cents"] == 0
    assert "subtotal_cents" in out
    assert out["margin_cents"] > 0
    assert out["qbo_payload"]["discount_cents"] == 0


def test_large_redemption_capped_at_total_fee(monkeypatch):
    """Large redeem + large balance → capped at the order's total service fee;
    margin is UNCHANGED vs the no-redemption call (practitioner keeps full margin)."""
    cart = [{"slug": "brain-boost", "qty": 1}]
    prac = {"id": "p1", "modules_completed": 0}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}

    # Baseline: no redemption — capture margin + fee.
    _common_stubs(monkeypatch)
    base = dc.build_client_order(cart, prac, patient=patient, method="card",
                                points_to_redeem_cents=0, points_balance_cents=0)
    baseline_margin = base["margin_cents"]
    total_fee = 660  # 1 bottle: 33% * (7000 - 5000)

    _common_stubs(monkeypatch)
    out = dc.build_client_order(cart, prac, patient=patient, method="card",
                               points_to_redeem_cents=999999,
                               points_balance_cents=999999)

    assert out["points_redeemed_cents"] == total_fee
    assert out["qbo_payload"]["discount_cents"] == total_fee
    assert out["margin_cents"] == baseline_margin   # margin unchanged; RM absorbs


def test_balance_is_binding_cap(monkeypatch):
    """Large redeem but small balance (150) → redeemed=150, discount=150."""
    cart = [{"slug": "brain-boost", "qty": 1}]
    prac = {"id": "p1", "modules_completed": 0}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}
    _common_stubs(monkeypatch)

    out = dc.build_client_order(cart, prac, patient=patient, method="card",
                               points_to_redeem_cents=999999,
                               points_balance_cents=150)

    assert out["points_redeemed_cents"] == 150
    assert out["qbo_payload"]["discount_cents"] == 150
