from dashboard import practitioner_pricing as pp

def test_defaults():
    s = pp.load_settings({})
    assert s["fee_pct"] == 0.33
    assert s["map_default_cents"] == 6700

def test_drop_ship_base_matches_blended_curve():
    # q1 = $50 for everyone; 12 uncertified ~ $47.11; 12 fully-certified ~ $42.76
    assert pp.drop_ship_base_cents(1, 0) == 5000
    assert pp.drop_ship_base_cents(1, 12) == 5000
    assert pp.drop_ship_base_cents(12, 0) == 4711
    assert pp.drop_ship_base_cents(12, 12) == 4276
    assert pp.drop_ship_base_cents(40, 12) == 2500   # floor at 2B, fully certified

def test_service_fee_flat_33pct_of_markup():
    s = pp.load_settings({})
    assert pp.service_fee_cents(8000, 5000, s) == 990    # 33% of (80-50)=30 -> $9.90
    assert pp.service_fee_cents(5000, 5000, s) == 0      # no markup -> no fee
    assert pp.service_fee_cents(4000, 5000, s) == 0      # negative markup clamps to 0
