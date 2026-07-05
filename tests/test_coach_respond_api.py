# tests/test_coach_respond_api.py
import sqlite3
from unittest import mock
import pytest
import app as appmod
from dashboard import coach_directory as _cd, coach_connect as _cc


@pytest.fixture(autouse=True)
def _fresh_coach_tables():
    """LOG_DB is one fixed path for the whole pytest run (DATA_DIR is set once
    by the test harness), so coach volunteers/applications/waitlist/interest
    rows from one test would otherwise leak into the next. Reset just the
    coaching-arc tables before each test (other app tables are untouched)."""
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _cd.init_coach_tables(cx)
        _cc.init_connect_tables(cx)
        cx.execute("DELETE FROM coach_volunteers")
        cx.execute("DELETE FROM coach_requests")
        cx.execute("DELETE FROM coach_waitlist")
        cx.execute("DELETE FROM coaching_interest")
        cx.commit()
    yield


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(coach_email="coach1@x.com", *, capacity=1):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx); _cc.init_connect_tables(cx)
        _cd.upsert_volunteer(cx, email=coach_email, name="Cora", focus="sleep",
                             intro_video_url="u", capacity=capacity, cert_ok=1)
        rid = _cc.create_request(cx, coach_email, "mem@x.com", "Mel", "sleep help")
        cx.commit()
    return rid


def test_requests_list_first_name_note_no_email():
    c = _client(); rid = _seed()
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="pid1"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="coach1@x.com"):
        d = c.get("/api/practitioner/coach-requests?token=t").get_json()
    assert d["pending"][0]["member_name"] == "Mel" and d["pending"][0]["note"] == "sleep help"
    assert "member_email" not in d["pending"][0] and "email" not in repr(d["pending"][0])
    assert d["capacity"] == 1 and d["slots_left"] == 1


def test_respond_accept_then_at_capacity():
    c = _client(); rid = _seed(capacity=1)
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="pid1"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="coach1@x.com"):
        r1 = c.post("/api/practitioner/coach-request/respond?token=t",
                    json={"request_id": rid, "accept": True})
        assert r1.get_json()["status"] == "accepted"
        # a second pending request now cannot be accepted (capacity 1 filled)
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _cc.init_connect_tables(cx)
            rid2 = _cc.create_request(cx, "coach1@x.com", "mem2@x.com", "Mo", "n"); cx.commit()
        r2 = c.post("/api/practitioner/coach-request/respond?token=t",
                    json={"request_id": rid2, "accept": True})
        assert r2.status_code == 409


def test_first_accept_wins_second_coach_member_taken():
    c = _client()
    # member applied to TWO coaches; coach1 accepts first, coach2 then gets member_taken
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cd.init_coach_tables(cx); _cc.init_connect_tables(cx)
        for e in ("coach1@x.com", "coach2@x.com"):
            _cd.upsert_volunteer(cx, email=e, name="C", focus="f", intro_video_url="u",
                                 capacity=1, cert_ok=1)
        rid1 = _cc.create_request(cx, "coach1@x.com", "mm@x.com", "Mel", "n")
        rid2 = _cc.create_request(cx, "coach2@x.com", "mm@x.com", "Mel", "n")
        cx.commit()
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="p1"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="coach1@x.com"):
        assert c.post("/api/practitioner/coach-request/respond?token=t",
                      json={"request_id": rid1, "accept": True}).get_json()["status"] == "accepted"
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="p2"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="coach2@x.com"):
        r2 = c.post("/api/practitioner/coach-request/respond?token=t",
                    json={"request_id": rid2, "accept": True})
    assert r2.status_code == 409 and r2.get_json()["error"] == "member_taken"


def test_respond_non_owner_404():
    c = _client(); rid = _seed()
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value="pid9"), \
         mock.patch("dashboard.practitioner_portal.practitioner_email_by_id", return_value="other@x.com"):
        r = c.post("/api/practitioner/coach-request/respond?token=t",
                   json={"request_id": rid, "accept": True})
    assert r.status_code == 404


def test_requests_requires_session():
    with mock.patch.object(appmod, "_practitioner_session_pid", return_value=None):
        assert _client().get("/api/practitioner/coach-requests?token=x").status_code == 401
