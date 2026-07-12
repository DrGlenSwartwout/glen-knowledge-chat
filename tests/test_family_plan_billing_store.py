import sqlite3
from dashboard import family_plan as fp


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    fp.init_family_plan_table(cx)
    return cx


def test_due_excludes_future_cancelled_and_comp():
    cx = _cx()
    fp.activate(cx, "due@x.com", next_charge_at="2026-07-01",
                customer_id="c", payment_method_id="p")            # past -> due
    fp.activate(cx, "future@x.com", next_charge_at="2026-12-01",
                customer_id="c", payment_method_id="p")            # future -> not due
    fp.activate(cx, "cxl@x.com", next_charge_at="2026-07-01",
                customer_id="c", payment_method_id="p")
    fp.set_status(cx, "cxl@x.com", "cancelled")                    # cancelled -> not due
    fp.activate(cx, "comp@x.com", next_charge_at=None, source="comp")  # comp -> never due
    emails = [d["caregiver_email"] for d in fp.due(cx, "2026-07-15")]
    assert emails == ["due@x.com"]


def test_mark_charged_advances_and_resets():
    cx = _cx()
    fp.activate(cx, "m@x.com", next_charge_at="2026-07-01",
                customer_id="c", payment_method_id="p")
    fp.mark_failed(cx, "m@x.com")
    assert fp.get(cx, "m@x.com")["fail_count"] == 1
    assert fp.get(cx, "m@x.com")["status"] == "past_due"
    fp.mark_charged(cx, "m@x.com", "2026-08-01")
    s = fp.get(cx, "m@x.com")
    assert s["next_charge_at"] == "2026-08-01" and s["fail_count"] == 0
    assert s["status"] == "active" and s["last_charged_at"]


def test_record_charge_ledger():
    cx = _cx()
    fp.record_charge(cx, caregiver_email="m@x.com", amount_cents=14700,
                     pi_id="pi_1", status="succeeded")
    row = cx.execute("SELECT * FROM family_sub_charges").fetchone()
    assert row["pi_id"] == "pi_1" and row["amount_cents"] == 14700
    assert row["caregiver_email"] == "m@x.com" and row["status"] == "succeeded"
