"""Partner Program 'Your standing' block — pure pricing math, no DB."""
from dashboard import practitioner_portal as pp


def test_partner_block_reports_floor_from_modules():
    b = pp.partner_block(4)
    assert b["modules_completed"] == 4
    assert b["floor_cents"] == 3500                 # 4000 - 4*125
    assert b["margin_low_cents"] == 6997 - 3500     # margin at $69.97 MAP
    assert b["margin_high_cents"] == 7997 - 3500    # margin at $79.97 SRP


def test_partner_block_certified_margin_range_is_64_to_69():
    b = pp.partner_block(12)
    assert b["floor_cents"] == 2500
    assert b["margin_low_pct"] == 64                # (6997-2500)/6997
    assert b["margin_high_pct"] == 69               # (7997-2500)/7997


def test_partner_block_uncertified_and_clamping():
    assert pp.partner_block(0)["floor_cents"] == 4000
    assert pp.partner_block(99)["floor_cents"] == 2500   # clamped to 12 modules
    assert pp.partner_block(None)["modules_completed"] == 0


def test_partner_block_carries_credit_figures():
    b = pp.partner_block(8, wellness_credit_cents=1234, dispensary_credit_cents=5600)
    assert b["wellness_credit_cents"] == 1234
    assert b["dispensary_credit_cents"] == 5600
