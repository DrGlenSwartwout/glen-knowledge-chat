import sqlite3
from datetime import datetime, timedelta
from dashboard import coaching


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    coaching.init_coaching_table(cx)
    return cx


def test_open_window_creates_and_active_window_returns_it():
    cx = _cx()
    res = coaching.open_window(cx, email="m@x.com", order_id=7,
                               days=coaching.WINDOW_DAYS, source="self_serve")
    assert res["created"] is True
    w = res["window"]
    assert w["email"] == "m@x.com" and w["order_id"] == 7 and w["source"] == "self_serve"
    aw = coaching.active_window(cx, "m@x.com")
    assert aw is not None and aw["id"] == w["id"]
    assert aw["days_remaining"] >= coaching.WINDOW_DAYS - 1


def test_no_stacking_second_open_is_noop():
    cx = _cx()
    first = coaching.open_window(cx, email="m@x.com", order_id=1,
                                 days=coaching.WINDOW_DAYS, source="self_serve")
    second = coaching.open_window(cx, email="m@x.com", order_id=2,
                                  days=coaching.WINDOW_DAYS, source="self_serve")
    assert second["created"] is False
    assert second["window"]["ends_at"] == first["window"]["ends_at"]
    assert cx.execute("SELECT COUNT(*) FROM coaching_windows").fetchone()[0] == 1


def test_one_window_per_order_after_lapse():
    cx = _cx()
    # Seed a LAPSED window for order 5 directly.
    past = (datetime.utcnow() - timedelta(days=40)).isoformat() + "Z"
    lapsed_end = (datetime.utcnow() - timedelta(days=10)).isoformat() + "Z"
    cx.execute("INSERT INTO coaching_windows (email, order_id, started_at, ends_at, source, created_at) "
               "VALUES (?,?,?,?,?,?)", ("m@x.com", 5, past, lapsed_end, "self_serve", past))
    cx.commit()
    assert coaching.active_window(cx, "m@x.com") is None       # lapsed, not active
    res = coaching.open_window(cx, email="m@x.com", order_id=5,
                               days=coaching.WINDOW_DAYS, source="self_serve")
    assert res["created"] is False                              # same order can't reopen
    assert coaching.window_for_order(cx, 5)["ends_at"] == lapsed_end


def test_list_windows_active_only():
    cx = _cx()
    coaching.open_window(cx, email="a@x.com", order_id=1, days=coaching.WINDOW_DAYS, source="self_serve")
    past = (datetime.utcnow() - timedelta(days=40)).isoformat() + "Z"
    cx.execute("INSERT INTO coaching_windows (email, order_id, started_at, ends_at, source, created_at) "
               "VALUES (?,?,?,?,?,?)", ("b@x.com", 2, past, past, "admin", past))
    cx.commit()
    assert len(coaching.list_windows(cx)) == 2
    active = coaching.list_windows(cx, active_only=True)
    assert len(active) == 1 and active[0]["email"] == "a@x.com"


def test_qualifying_sources_excludes_membership():
    assert "membership" not in coaching.QUALIFYING_SOURCES
    assert "wholesale" not in coaching.QUALIFYING_SOURCES
    assert {"biofield", "reorder", "portal-reorder"} <= coaching.QUALIFYING_SOURCES
