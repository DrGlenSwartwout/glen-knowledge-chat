import sqlite3
from dashboard import biofield_store as bs, biofield_gate as bg


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    bs.init_table(cx); return cx


def test_gate_unlocks_only_when_all_green():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    st = bg.gate_state(cx, "p@x.com", has_intake=lambda e: False)
    assert st["paid"] and not st["booking_unlocked"]
    assert st["items"]["photo"]["status"] == "needed"
    assert st["items"]["scan"]["status"] == "needed"
    assert st["items"]["intake"]["status"] == "needed"

    bs.set_photo_on_file(cx, "p@x.com", "x.jpg")
    bs.set_scan_confirmed(cx, "p@x.com", True)
    st = bg.gate_state(cx, "p@x.com", has_intake=lambda e: True)  # intake auto-detected
    assert st["items"]["photo"]["status"] == "green"
    assert st["items"]["scan"]["status"] == "green"
    assert st["items"]["intake"]["status"] == "green"
    assert st["booking_unlocked"] is True


def test_intake_green_via_self_confirm_even_without_auto():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    bs.set_intake_confirmed(cx, "p@x.com", True)
    st = bg.gate_state(cx, "p@x.com", has_intake=lambda e: False)
    assert st["items"]["intake"]["status"] == "green"


def test_not_paid_never_unlocks():
    cx = _cx()
    st = bg.gate_state(cx, "nobody@x.com", has_intake=lambda e: True)
    assert st["paid"] is False
    assert st["booking_unlocked"] is False
    # all items still reported (needed) so the page can render
    assert set(st["items"].keys()) == {"photo", "intake", "scan"}


def test_already_booked_still_reports_unlocked_true():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    bs.set_photo_on_file(cx, "p@x.com", "x.jpg")
    bs.set_scan_confirmed(cx, "p@x.com", True)
    bs.set_booked(cx, "p@x.com")
    st = bg.gate_state(cx, "p@x.com", has_intake=lambda e: True)
    assert st["booking_unlocked"] is True
    assert st["booked"] is True
