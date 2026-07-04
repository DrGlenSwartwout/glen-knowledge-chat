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


def test_credit_for_charge_credits_attributed():
    seen = {}
    def earn(pid, cents, *, event_ref):
        seen.update(pid=pid, cents=cents, event_ref=event_ref); return cents
    sub = {"id": 7, "order_count": 3, "attributed_practitioner_id": "prac-42"}
    out = cs.credit_for_charge(sub, charge_cents=9900,
                               earn=earn, resolve_modules=lambda pid: 12)
    assert out == 4950
    assert seen == {"pid": "prac-42", "cents": 4950, "event_ref": "care_share:7:3"}


def test_credit_for_charge_no_attribution():
    called = []
    sub = {"id": 7, "order_count": 3, "attributed_practitioner_id": None}
    out = cs.credit_for_charge(sub, charge_cents=9900,
                               earn=lambda *a, **k: called.append(1),
                               resolve_modules=lambda pid: 12)
    assert out == 0 and called == []


def test_credit_for_charge_owner_not_a_practitioner():
    sub = {"id": 7, "order_count": 3, "attributed_practitioner_id": "someone"}
    out = cs.credit_for_charge(sub, charge_cents=9900,
                               earn=lambda *a, **k: 1/0,   # must not be called
                               resolve_modules=lambda pid: None)
    assert out == 0
