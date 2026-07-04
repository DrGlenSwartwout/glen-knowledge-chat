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
