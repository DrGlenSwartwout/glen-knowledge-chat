import sqlite3
from dashboard import biofield_store as bs


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    bs.init_table(cx); return cx


def test_seed_paid_and_flags():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    r = bs.get(cx, "p@x.com")
    assert r["paid_at"] and r["paid_via"] == "stripe" and r["order_ref"] == "INV1"
    assert not r["photo_on_file"]
    bs.set_photo_on_file(cx, "p@x.com", "data/biofield-photos/p_x_com.jpg")
    bs.set_intake_confirmed(cx, "p@x.com", True)
    bs.set_scan_confirmed(cx, "p@x.com", True)
    r = bs.get(cx, "p@x.com")
    assert r["photo_on_file"] and r["intake_confirmed"] and r["scan_confirmed"]
    assert r["photo_path"].endswith(".jpg")


def test_seed_paid_idempotent():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    first = bs.get(cx, "p@x.com")["paid_at"]
    bs.seed_paid(cx, "p@x.com", via="pb", order_ref="INV2")  # must NOT overwrite paid_at
    assert bs.get(cx, "p@x.com")["paid_at"] == first


def test_set_flag_creates_row_if_missing():
    cx = _cx()
    # setting a flag before paid should still work (bare row created)
    bs.set_photo_on_file(cx, "new@x.com", "x.jpg")
    r = bs.get(cx, "new@x.com")
    assert r is not None and r["photo_on_file"] and not r["paid_at"]


def test_set_booked():
    cx = _cx()
    bs.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    bs.set_booked(cx, "p@x.com")
    assert bs.get(cx, "p@x.com")["booked_at"]


def test_get_missing_returns_none():
    cx = _cx()
    assert bs.get(cx, "nobody@x.com") is None
