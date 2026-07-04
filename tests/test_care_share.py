from dashboard import care_share as cs


def test_rate_tiers():
    assert cs.rate(0) == 0.30
    assert abs(cs.rate(6) - 0.40) < 1e-9
    assert cs.rate(12) == 0.50


def test_rate_clamps():
    assert cs.rate(-5) == 0.30
    assert cs.rate(99) == 0.50


def test_share_cents_tiers():
    assert cs.share_cents(9900, 0) == 2970
    assert cs.share_cents(9900, 6) == 3960
    assert cs.share_cents(9900, 12) == 4950


def test_share_cents_prepay_lump():
    # 12-month prepay at full cert: 50% of $990.00
    assert cs.share_cents(99000, 12) == 49500


def test_share_cents_rounds_half():
    # 40% of 9901 = 3960.4 -> 3960
    assert cs.share_cents(9901, 6) == 3960
