import sqlite3
from dashboard import coach_subscriptions as _cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _cs.init_sub_tables(cx)
    return cx


def test_tiers():
    assert _cs.TIERS["rae"]["amount_cents"] == 10000
    assert _cs.TIERS["glen"]["amount_cents"] == 20000
    assert _cs.TIERS["rae"]["service"] == "evox" and _cs.TIERS["glen"]["service"] == "biofield"


def test_create_and_get():
    cx = _cx()
    _cs.create_sub(cx, email="M@x.com", tier="rae", customer_id="cus_1",
                   payment_method_id="pm_1", next_charge_at="2026-08-05")
    s = _cs.get_sub(cx, "m@x.com")
    assert s["tier"] == "rae" and s["status"] == "active" and s["next_charge_at"] == "2026-08-05"
    assert s["stripe_customer_id"] == "cus_1" and s["payment_method_id"] == "pm_1"


def test_due_only_active_and_past():
    cx = _cx()
    _cs.create_sub(cx, email="due@x.com", tier="rae", customer_id="c", payment_method_id="p",
                   next_charge_at="2026-07-01")            # past → due
    _cs.create_sub(cx, email="future@x.com", tier="glen", customer_id="c", payment_method_id="p",
                   next_charge_at="2026-12-01")            # future → not due
    _cs.create_sub(cx, email="cx@x.com", tier="rae", customer_id="c", payment_method_id="p",
                   next_charge_at="2026-07-01"); _cs.set_status(cx, "cx@x.com", "canceled")
    emails = [d["member_email"] for d in _cs.due(cx, "2026-07-15")]
    assert emails == ["due@x.com"]                          # not future, not canceled


def test_mark_charged_advances_and_resets():
    cx = _cx()
    _cs.create_sub(cx, email="m@x.com", tier="rae", customer_id="c", payment_method_id="p",
                   next_charge_at="2026-07-01")
    _cs.mark_failed(cx, "m@x.com")
    assert _cs.get_sub(cx, "m@x.com")["fail_count"] == 1
    assert _cs.get_sub(cx, "m@x.com")["status"] == "past_due"
    _cs.mark_charged(cx, "m@x.com", "2026-08-01")
    s = _cs.get_sub(cx, "m@x.com")
    assert s["next_charge_at"] == "2026-08-01" and s["fail_count"] == 0 and s["last_charged_at"]


def test_record_charge_ledger():
    cx = _cx()
    _cs.record_charge(cx, email="m@x.com", tier="rae", amount_cents=10000, pi_id="pi_1",
                      status="succeeded")
    row = cx.execute("SELECT * FROM coach_sub_charges").fetchone()
    assert row["pi_id"] == "pi_1" and row["amount_cents"] == 10000
