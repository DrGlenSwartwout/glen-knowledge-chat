import sqlite3
from dashboard import masterclass as mc

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    mc.init_masterclass_tables(cx); return cx

def test_event_create_get_zoom_price():
    cx = _cx()
    eid = mc.create_event(cx, topic="Terrain 101", description="d",
                          start_ts="2026-07-10T18:00:00", duration_min=60,
                          price_cents=5000, member_price_cents=0)
    ev = mc.get_event(cx, eid)
    assert ev["topic"] == "Terrain 101" and ev["price_cents"] == 5000
    assert mc.price_for(ev, is_member=True) == 0
    assert mc.price_for(ev, is_member=False) == 5000
    mc.set_zoom(cx, eid, "https://zoom.us/j/123", "123")
    assert mc.get_event(cx, eid)["zoom_join_url"] == "https://zoom.us/j/123"

def test_register_upsert_and_mark_paid():
    cx = _cx()
    eid = mc.create_event(cx, topic="T", description="", start_ts="2026-07-10T18:00:00",
                          duration_min=60, price_cents=5000, member_price_cents=0)
    mc.register(cx, eid, "A@x.com", "A", is_member=False, amount_cents=5000, paid=False)
    assert mc.is_registered(cx, eid, "a@x.com") is False        # pending, not paid
    mc.mark_paid(cx, eid, "a@x.com")
    assert mc.is_registered(cx, eid, "A@x.com") is True          # lowercased
    # re-register (upsert) doesn't duplicate
    mc.register(cx, eid, "a@x.com", "A", is_member=True, amount_cents=0, paid=True)
    n = cx.execute("SELECT COUNT(*) FROM masterclass_registrations WHERE event_id=?", (eid,)).fetchone()[0]
    assert n == 1 and mc.is_registered(cx, eid, "a@x.com") is True
