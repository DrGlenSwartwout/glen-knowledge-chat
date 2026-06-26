"""Console calendar: scope-aware multi-owner GET, date-range windows,
per-calendar->owner mapping, and accomplishments.

Mirrors the LOG_DB / CONSOLE_SECRET monkeypatch pattern used across the suite
(see test_notify_routes.py / test_access_token_guard.py).
"""

import calendar as _cal
import sqlite3
from datetime import date, timedelta

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod._init_auth_tables()
    appmod._init_workspace_schema()
    appmod._init_calendar_table()
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


ADMIN = {"X-Console-Key": "test-secret"}


def _seed_event(appmod, *, owner="glen", start="2026-06-22T09:00:00-10:00",
                summary="Consult", gcal="glen-cal", evid=None):
    evid = evid or f"ev-{owner}-{start}"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute(
            "INSERT OR REPLACE INTO calendar_events "
            "(pushed_at, google_cal_id, google_event_id, calendar_name, summary, "
            " start, end, location, owner, status, cal_alert) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-06-22T00:00:00Z", gcal, evid, gcal, summary,
             start, "", "", owner, "visible", 0),
        )
        cx.commit()


def _seed_scoped_token(appmod, token="shaira-tok", owner="shaira"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute(
            "INSERT INTO workspace_users (name, display_name, scope) VALUES (?,?,?)",
            (owner, owner.title(), f"workspace:{owner}"),
        )
        uid = cx.execute("SELECT id FROM workspace_users WHERE name=?", (owner,)).fetchone()[0]
        cx.execute(
            "INSERT INTO access_tokens (token, user_id, note) VALUES (?,?,?)",
            (token, uid, "test"),
        )
        cx.commit()
    return {"X-Console-Key": token}


# ── date-range window math (pure) ─────────────────────────────────────────────

def test_range_today_is_single_day():
    import app as appmod
    start, end = appmod._calendar_range_window("today", anchor="2026-06-22")
    assert start == "2026-06-22"
    assert end == "2026-06-22"


def test_range_2day_spans_tomorrow():
    import app as appmod
    start, end = appmod._calendar_range_window("2day", anchor="2026-06-22")
    assert start == "2026-06-22"
    assert end == "2026-06-23"


def test_range_week_is_monday_to_sunday():
    import app as appmod
    anchor = "2026-06-24"  # a Wednesday
    start, end = appmod._calendar_range_window("week", anchor=anchor)
    d = date.fromisoformat(anchor)
    mon = d - timedelta(days=d.weekday())
    sun = mon + timedelta(days=6)
    assert start == mon.isoformat()
    assert end == sun.isoformat()


def test_range_month_is_first_to_last():
    import app as appmod
    start, end = appmod._calendar_range_window("month", anchor="2026-06-15")
    last = _cal.monthrange(2026, 6)[1]
    assert start == "2026-06-01"
    assert end == f"2026-06-{last:02d}"


# ── per-calendar -> owner mapping (pure) ──────────────────────────────────────

def test_owner_mapping_resolves_by_cal_id():
    import app as appmod
    mp = {"rae-cal-xyz": "rae", "shaira-cal-abc": "shaira"}
    assert appmod._calendar_owner_for("rae-cal-xyz", "Rae", mp) == "rae"
    assert appmod._calendar_owner_for("shaira-cal-abc", "Shaira Work", mp) == "shaira"


def test_owner_mapping_unknown_defaults_glen():
    import app as appmod
    assert appmod._calendar_owner_for("some-random-cal", "Whatever", {}) == "glen"


# ── scope-aware GET ───────────────────────────────────────────────────────────

def test_get_requires_auth(client):
    c, appmod = client
    assert c.get("/api/calendar?range=today").status_code == 401


def test_admin_sees_all_owners(client):
    c, appmod = client
    _seed_event(appmod, owner="glen", summary="G")
    _seed_event(appmod, owner="rae", summary="R")
    _seed_event(appmod, owner="shaira", summary="S")
    r = c.get("/api/calendar?range=today&date=2026-06-22", headers=ADMIN)
    assert r.status_code == 200
    owners = {e["owner"] for e in r.get_json()["events"]}
    assert owners == {"glen", "rae", "shaira"}


def test_scoped_user_sees_only_own(client):
    c, appmod = client
    _seed_event(appmod, owner="glen", summary="G")
    _seed_event(appmod, owner="shaira", summary="S")
    hdr = _seed_scoped_token(appmod)
    r = c.get("/api/calendar?range=today&date=2026-06-22", headers=hdr)
    assert r.status_code == 200
    owners = {e["owner"] for e in r.get_json()["events"]}
    assert owners == {"shaira"}


def test_range_filters_out_of_window(client):
    c, appmod = client
    _seed_event(appmod, owner="glen", start="2026-06-22T09:00:00-10:00", summary="today", evid="a")
    _seed_event(appmod, owner="glen", start="2026-07-15T09:00:00-10:00", summary="next-month", evid="b")
    r = c.get("/api/calendar?range=today&date=2026-06-22", headers=ADMIN)
    summaries = {e["summary"] for e in r.get_json()["events"]}
    assert summaries == {"today"}


# ── delegate (move, not copy) ─────────────────────────────────────────────────

def _delegate(c, event_id, to):
    return c.patch(f"/api/calendar/{event_id}",
                   headers=ADMIN, json={"action": "delegate", "to": to})


def test_delegate_moves_event_off_delegator_lane(client):
    c, appmod = client
    _seed_event(appmod, owner="glen", summary="Hand to Rae", evid="d1")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        eid = cx.execute("SELECT id FROM calendar_events WHERE google_event_id='d1'").fetchone()[0]
    assert _delegate(c, eid, "rae").status_code == 200

    # Original no longer visible on Glen's lane (moved, not copied).
    g = c.get("/api/calendar?range=today&date=2026-06-22&owners=glen", headers=ADMIN)
    assert "Hand to Rae" not in {e["summary"] for e in g.get_json()["events"]}


def test_delegate_shows_event_on_delegatee_lane(client):
    c, appmod = client
    _seed_event(appmod, owner="glen", summary="Hand to Rae", evid="d2")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        eid = cx.execute("SELECT id FROM calendar_events WHERE google_event_id='d2'").fetchone()[0]
    assert _delegate(c, eid, "rae").status_code == 200

    r = c.get("/api/calendar?range=today&date=2026-06-22&owners=rae", headers=ADMIN)
    rae = [e for e in r.get_json()["events"] if e["summary"] == "Hand to Rae"]
    assert rae and rae[0]["owner"] == "rae"


# ── accomplishments ───────────────────────────────────────────────────────────

def test_accomplishment_add_and_list(client):
    c, appmod = client
    r = c.post("/api/calendar/accomplishment",
               headers=ADMIN,
               json={"owner": "rae", "title": "Closed the week", "at": "2026-06-22T14:00:00-10:00"})
    assert r.status_code == 200
    g = c.get("/api/calendar?range=today&date=2026-06-22&kind=accomplishments", headers=ADMIN)
    items = g.get_json()["events"]
    assert any(i["title"] == "Closed the week" and i["kind"] == "accomplishment" for i in items)


def test_scoped_user_cannot_write_other_owner(client):
    c, appmod = client
    hdr = _seed_scoped_token(appmod)
    r = c.post("/api/calendar/accomplishment",
               headers=hdr,
               json={"owner": "glen", "title": "nope", "at": "2026-06-22T14:00:00-10:00"})
    assert r.status_code == 403


def test_post_mapping_overrides_heuristic_owner(client):
    c, appmod = client
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("INSERT INTO calendar_owner_map (google_cal_id, owner) VALUES (?,?)",
                   ("shaira-cal", "shaira"))
        cx.commit()
    # The cron posts its heuristic owner=glen, but this calendar is mapped to shaira.
    r = c.post("/api/calendar", headers=ADMIN, json=[{
        "google_cal_id": "shaira-cal", "google_event_id": "e1", "calendar_name": "Shaira",
        "summary": "Work block", "start": "2026-06-22T09:00:00-10:00", "owner": "glen"}])
    assert r.status_code == 201
    g = c.get("/api/calendar?range=today&date=2026-06-22", headers=ADMIN)
    evs = [e for e in g.get_json()["events"] if e["summary"] == "Work block"]
    assert evs and evs[0]["owner"] == "shaira"


def test_post_unmapped_keeps_explicit_owner(client):
    c, appmod = client
    r = c.post("/api/calendar", headers=ADMIN, json=[{
        "google_cal_id": "glen-cal", "google_event_id": "e2", "calendar_name": "Glen",
        "summary": "Pay invoice", "start": "2026-06-22T10:00:00-10:00", "owner": "rae"}])
    assert r.status_code == 201
    g = c.get("/api/calendar?range=today&date=2026-06-22", headers=ADMIN)
    evs = [e for e in g.get_json()["events"] if e["summary"] == "Pay invoice"]
    assert evs and evs[0]["owner"] == "rae"


def test_events_default_kind_excludes_accomplishments(client):
    c, appmod = client
    _seed_event(appmod, owner="glen", summary="an-event")
    c.post("/api/calendar/accomplishment", headers=ADMIN,
           json={"owner": "glen", "title": "a-done-thing", "at": "2026-06-22T14:00:00-10:00"})
    r = c.get("/api/calendar?range=today&date=2026-06-22", headers=ADMIN)  # default kind=events
    titles = {e.get("summary") or e.get("title") for e in r.get_json()["events"]}
    assert "an-event" in titles
    assert "a-done-thing" not in titles
