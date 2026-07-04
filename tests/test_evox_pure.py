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
