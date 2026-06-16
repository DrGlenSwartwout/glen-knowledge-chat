from dashboard import wallet


def test_personal_earn_fee_free_rate():
    assert wallet.PERSONAL_EARN_FEE_FREE_PCT == 0.035


def test_personal_earn_cents_zelle_and_wise():
    assert wallet.personal_earn_cents(10000, "zelle") == 350
    assert wallet.personal_earn_cents(10000, "wise") == 350
    assert wallet.personal_earn_cents(10000, "Wise") == 350   # case-insensitive


def test_personal_earn_cents_card_and_zero():
    assert wallet.personal_earn_cents(10000, "card") == 0
    assert wallet.personal_earn_cents(0, "zelle") == 0
    assert wallet.personal_earn_cents(10000, "") == 0
