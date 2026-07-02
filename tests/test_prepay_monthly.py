import dashboard.prepay as prepay

def test_public_tiers_exclude_3mo():
    keys = [t["key"] for t in prepay.tiers_public()]
    assert keys == ["1mo", "6mo", "12mo"]

def test_monthly_total_and_savings():
    assert prepay.monthly_total_cents("6mo") == 9900 * 6          # 59400
    assert prepay.upfront_savings_cents("6mo") == 59400 - 54600   # 4800
    assert prepay.monthly_total_cents("12mo") == 9900 * 12        # 118800
    assert prepay.upfront_savings_cents("12mo") == 118800 - 99000 # 19800

def test_commitment_flag_and_fields():
    by = {t["key"]: t for t in prepay.tiers_public()}
    assert by["6mo"]["commitment"] is True and by["12mo"]["commitment"] is True
    assert by["1mo"]["commitment"] is False
    assert by["6mo"]["monthly_cents"] == 9900
    assert by["6mo"]["monthly_total_cents"] == 59400
    assert by["6mo"]["upfront_savings_cents"] == 4800
