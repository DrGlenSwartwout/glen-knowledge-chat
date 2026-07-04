from dashboard import evox

def test_readiness_complete_all_true():
    assert evox.readiness_complete(
        {"pc_ok": True, "cradle_ok": True, "headset_ok": True, "zyto_ok": True}) is True

def test_readiness_incomplete_when_any_false():
    assert evox.readiness_complete(
        {"pc_ok": True, "cradle_ok": False, "headset_ok": True, "zyto_ok": True}) is False

def test_readiness_incomplete_when_missing_key():
    assert evox.readiness_complete({"pc_ok": True}) is False


import sqlite3
def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    evox.init_evox_tables(cx); return cx

def test_readiness_roundtrip_and_complete():
    cx = _cx()
    assert evox.get_readiness(cx, "A@x.com")["complete"] is False
    for item in ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok"):
        st = evox.set_readiness_item(cx, "a@x.com", item, True)
    assert st["complete"] is True
    assert evox.get_readiness(cx, "a@x.com")["complete"] is True   # email lowercased

def test_cradle_source_recorded():
    cx = _cx()
    st = evox.set_readiness_item(cx, "b@x.com", "cradle_ok", True, cradle_source="buy")
    assert st["cradle_source"] == "buy"


from datetime import date, datetime

def test_parse_office_hours():
    assert evox.parse_office_hours("1-4:09:00-16:00") == (1, 4, "09:00", "16:00")

def test_slot_grid_weekday_in_window():
    slots = evox.slot_grid(date(2026, 7, 6), "1-4:09:00-16:00")   # Mon
    assert slots[0] == "2026-07-06T09:00:00"
    assert slots[-1] == "2026-07-06T15:00:00"   # last 60-min slot starts 15:00, ends 16:00
    assert len(slots) == 7

def test_slot_grid_weekday_out_of_window():
    assert evox.slot_grid(date(2026, 7, 5), "1-4:09:00-16:00") == []   # Sunday

def test_available_excludes_busy_and_booked_and_past():
    days = [date(2026, 7, 6)]
    now = datetime(2026, 7, 6, 10, 30)                      # 09:00 & 10:00 are past
    busy = [("2026-07-06T13:00:00", "2026-07-06T14:00:00")] # blocks 13:00
    booked = {"2026-07-06T12:00:00"}                        # blocks 12:00
    slots = evox.available_slots(days, "1-4:09:00-16:00", busy, booked, now)
    assert slots == ["2026-07-06T11:00:00", "2026-07-06T14:00:00", "2026-07-06T15:00:00"]

def test_available_allday_busy_blocks_whole_day():
    days = [date(2026, 7, 6)]
    now = datetime(2026, 7, 6, 0, 0)
    slots = evox.available_slots(days, "1-4:09:00-16:00", [("2026-07-06", "")], set(), now)
    assert slots == []

def test_available_skips_unparseable_busy():
    days = [date(2026, 7, 6)]                       # Mon
    now = datetime(2026, 7, 6, 0, 0)
    busy = [("garbage", "also-bad"),                # unparseable -> ignored
            ("2026-07-06T13:00:00", "2026-07-06T14:00:00")]  # blocks 13:00
    slots = evox.available_slots(days, "1-4:09:00-16:00", busy, set(), now)
    assert slots == ["2026-07-06T09:00:00", "2026-07-06T10:00:00",
                     "2026-07-06T11:00:00", "2026-07-06T12:00:00",
                     "2026-07-06T14:00:00", "2026-07-06T15:00:00"]

def test_available_nonstring_busy_does_not_crash():
    days = [date(2026, 7, 6)]
    now = datetime(2026, 7, 6, 0, 0)
    slots = evox.available_slots(days, "1-4:09:00-16:00", [(12345, "")], set(), now)
    assert len(slots) == 7                          # garbage row ignored, full grid

def test_available_multiple_intervals_and_days_sorted():
    days = [date(2026, 7, 6), date(2026, 7, 7)]     # Mon, Tue
    now = datetime(2026, 7, 6, 0, 0)
    busy = [("2026-07-06T09:00:00", "2026-07-06T10:00:00"),
            ("2026-07-07T15:00:00", "2026-07-07T16:00:00")]
    slots = evox.available_slots(days, "1-4:09:00-16:00", busy, set(), now)
    assert slots == sorted(slots)                   # globally sorted across days
    assert "2026-07-06T09:00:00" not in slots
    assert "2026-07-07T15:00:00" not in slots
    assert "2026-07-06T10:00:00" in slots and "2026-07-07T09:00:00" in slots


def _cal_cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    evox.init_evox_tables(cx)
    cx.execute("""CREATE TABLE calendar_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        pushed_at TEXT, google_cal_id TEXT, google_event_id TEXT, calendar_name TEXT,
        summary TEXT, start TEXT, end TEXT, location TEXT, owner TEXT, status TEXT,
        cal_alert INTEGER, UNIQUE(google_cal_id, google_event_id))""")
    cx.commit(); return cx

def test_create_booking_writes_calendar_and_tags():
    cx = _cal_cx(); seen = {}
    def tag_fn(email, tags): seen[email] = tags
    b = evox.create_booking(cx, "c@x.com", "2026-07-06T11:00:00", tag_fn=tag_fn)
    assert b["end_ts"] == "2026-07-06T12:00:00"
    row = cx.execute("SELECT owner,status,google_cal_id,google_event_id FROM calendar_events").fetchone()
    assert (row["owner"], row["status"], row["google_cal_id"]) == ("rae", "visible", "delegated")
    assert row["google_event_id"] == f"evox-{b['id']}"
    assert seen["c@x.com"] == ["evox-client", "evox-ready"]
    assert "2026-07-06T11:00:00" in evox.booked_starts(cx)

def test_double_book_raises():
    cx = _cal_cx()
    evox.create_booking(cx, "c@x.com", "2026-07-06T11:00:00")
    import pytest
    with pytest.raises(evox.SlotTaken):
        evox.create_booking(cx, "d@x.com", "2026-07-06T11:00:00")

def test_booked_starts_excludes_cancelled():
    cx = _cal_cx()
    b = evox.create_booking(cx, "c@x.com", "2026-07-06T11:00:00")
    assert "2026-07-06T11:00:00" in evox.booked_starts(cx)
    cx.execute("UPDATE evox_bookings SET status='cancelled' WHERE id=?", (b["id"],)); cx.commit()
    assert "2026-07-06T11:00:00" not in evox.booked_starts(cx)

def test_rae_busy_intervals_reads_calendar():
    cx = _cal_cx()
    cx.execute("INSERT INTO calendar_events (pushed_at,google_cal_id,google_event_id,"
               "summary,start,end,owner,status) VALUES (?,?,?,?,?,?,?,?)",
               ("x", "rae@g", "e1", "Busy", "2026-07-06T13:00:00",
                "2026-07-06T14:00:00", "rae", "visible")); cx.commit()
    assert evox.rae_busy_intervals(cx, "2026-07-06", "2026-07-06") == \
        [("2026-07-06T13:00:00", "2026-07-06T14:00:00")]


import json
def test_build_ics_valid():
    ics = evox.build_ics(uid="u1@illtowell.com", start_ts="2026-07-06T11:00:00",
                         end_ts="2026-07-06T12:00:00", summary="EVOX Session",
                         description="Call Rae at 808-555-1212", location="Phone")
    t = ics.decode()
    assert t.startswith("BEGIN:VCALENDAR") and "METHOD:REQUEST" in t
    assert "BEGIN:VEVENT" in t and "UID:u1@illtowell.com" in t
    assert "DTSTART:20260706T110000" in t and "DTEND:20260706T120000" in t
    assert t.endswith("END:VCALENDAR\r\n") and "\r\n" in t

def test_session_credits():
    cx = _cx()
    assert evox.session_credit_balance(cx, "e@x.com") == 0
    assert evox.add_session_credits(cx, "e@x.com", 3) == 3
    assert evox.consume_session_credit(cx, "e@x.com") is True
    assert evox.session_credit_balance(cx, "e@x.com") == 2

def test_consume_credit_when_zero():
    cx = _cx()
    assert evox.consume_session_credit(cx, "z@x.com") is False

def test_has_cradle_purchase():
    cx = _cal_cx()
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, items_json TEXT)")
    cx.execute("INSERT INTO orders (email, items_json) VALUES (?,?)",
               ("buyer@x.com", json.dumps([{"slug": "hand-cradle", "qty": 1}]))); cx.commit()
    assert evox.has_cradle_purchase(cx, "BUYER@x.com") is True
    assert evox.has_cradle_purchase(cx, "nobody@x.com") is False
