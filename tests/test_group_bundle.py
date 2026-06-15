from dashboard import group_bundle as gb


def test_included_group_months_one_per_program_month_capped_at_3():
    assert gb.included_group_months(0) == 0
    assert gb.included_group_months(1) == 1
    assert gb.included_group_months(2) == 2
    assert gb.included_group_months(3) == 3
    assert gb.included_group_months(6) == 3
    assert gb.included_group_months(12) == 3


def test_included_group_months_handles_junk():
    assert gb.included_group_months(None) == 0
    assert gb.included_group_months(-4) == 0
    assert gb.included_group_months("2") == 2


def test_membership_amount_default():
    assert gb.MEMBERSHIP_AMOUNT_CENTS == 9900
    assert gb.MEMBERSHIP_CADENCE_MONTHS == 1
