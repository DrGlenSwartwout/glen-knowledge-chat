"""Task 7: continuity_view.send_recommendation + the 3 gate-checked practitioner routes.

Patterns borrowed from sibling tests:
  - app-import + LOG_DB monkeypatch + _practitioner_session_pid override:
    tests/test_practitioner_pricing_routes.py
  - real subscriptions init + migrate chain (attribution + consent columns) to
    seed consented-continuity memberships: tests/test_continuity_authz.py /
    tests/test_continuity_roster.py

SECURITY FOCUS: the patient + recommend routes must gate-check BEFORE any read
or write. The critical assertions are the 403 cases: no patient data in the
response body, and no practitioner_recommendations row written.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import subscriptions as subs
from dashboard import practitioner_recommendations as pr

PID = "prac-42"
OTHER_PID = "prac-99"


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _init_migrate(cx):
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)


def _mk(cx, email, pid, consent=True, status="active"):
    sid = subs.create_membership(
        cx, email=email, stripe_customer_id="c", stripe_payment_method_id="p",
        amount_cents=9900, next_charge_date="2026-08-01", attributed_practitioner_id=pid,
    )
    if consent:
        cx.execute("UPDATE subscriptions SET practitioner_share_consent=1 WHERE id=?", (sid,))
    if status != "active":
        cx.execute("UPDATE subscriptions SET status=? WHERE id=?", (status, sid))
    cx.commit()
    return sid


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Wires up a real sqlite file at app.LOG_DB (so routes' own connections see
    seeded data), a signed-in session for PID, and pre-runs the migrate chain."""
    app_module = _app()
    db_path = tmp_path / "chat_log.db"
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        _init_migrate(cx)

    monkeypatch.setattr(app_module, "LOG_DB", db_path)
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: PID)
    return app_module, db_path


def _seed(db_path, email, pid, consent=True, status="active"):
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        _init_migrate(cx)
        _mk(cx, email, pid, consent=consent, status=status)


# ---------------------------------------------------------------------------
# roster
# ---------------------------------------------------------------------------
def test_roster_requires_auth(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: None)
    client = app_module.app.test_client()
    r = client.get("/api/practitioner/continuity/roster")
    assert r.status_code == 401
    assert not r.get_json()["ok"]


def test_roster_returns_only_signed_in_doctors_consented_patients(wired):
    app_module, db_path = wired
    _seed(db_path, "mine@x.com", PID, consent=True)
    _seed(db_path, "not-mine@x.com", OTHER_PID, consent=True)
    _seed(db_path, "unconsented@x.com", PID, consent=False)

    client = app_module.app.test_client()
    r = client.get("/api/practitioner/continuity/roster")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    emails = {row["email"] for row in body["roster"]}
    assert emails == {"mine@x.com"}


# ---------------------------------------------------------------------------
# patient view
# ---------------------------------------------------------------------------
def test_patient_route_requires_auth(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: None)
    client = app_module.app.test_client()
    r = client.get("/api/practitioner/continuity/patient/pat@x.com")
    assert r.status_code == 401


def test_patient_route_403_for_unauthorized_patient_with_no_data_in_body(wired):
    app_module, db_path = wired
    # pat@x.com belongs to a DIFFERENT doctor, not PID.
    _seed(db_path, "pat@x.com", OTHER_PID, consent=True)

    client = app_module.app.test_client()
    r = client.get("/api/practitioner/continuity/patient/pat@x.com")
    assert r.status_code == 403
    body = r.get_json()
    assert body["ok"] is False
    # no patient data leaked into the body
    assert "trajectory" not in body
    assert "narrative" not in body
    assert "suggested_step" not in body


def test_patient_route_200_with_view_for_authorized_patient(wired):
    app_module, db_path = wired
    _seed(db_path, "pat@x.com", PID, consent=True)

    client = app_module.app.test_client()
    r = client.get("/api/practitioner/continuity/patient/pat@x.com")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert "trajectory" in body
    assert "narrative" in body
    assert "suggested_step" in body


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------
def test_recommend_requires_auth(monkeypatch, wired):
    app_module, _db = wired
    monkeypatch.setattr(app_module, "_practitioner_session_pid", lambda: None)
    client = app_module.app.test_client()
    r = client.post("/api/practitioner/continuity/recommend",
                    json={"patient_email": "pat@x.com", "items": [], "note": ""})
    assert r.status_code == 401


def test_recommend_403_for_unauthorized_patient_and_no_row_written(wired):
    app_module, db_path = wired
    _seed(db_path, "pat@x.com", OTHER_PID, consent=True)

    client = app_module.app.test_client()
    r = client.post("/api/practitioner/continuity/recommend",
                    json={"patient_email": "pat@x.com",
                          "items": [{"slug": "neuro-magnesium", "qty": 1}],
                          "note": "keep going"})
    assert r.status_code == 403
    body = r.get_json()
    assert body["ok"] is False

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        pr.init_table(cx)
        assert pr.active_for_patient(cx, "pat@x.com") is None


def test_recommend_writes_row_and_attempts_notification_when_authorized(monkeypatch, wired):
    app_module, db_path = wired
    _seed(db_path, "pat@x.com", PID, consent=True)

    sent = []
    from dashboard import inbox as _inbox
    monkeypatch.setattr(_inbox, "send_email",
                        lambda *a, **k: sent.append((a, k)) or {"id": "x"})

    client = app_module.app.test_client()
    r = client.post("/api/practitioner/continuity/recommend",
                    json={"patient_email": "pat@x.com",
                          "items": [{"slug": "neuro-magnesium", "qty": 1}],
                          "note": "keep going"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["id"]

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        pr.init_table(cx)
        rec = pr.active_for_patient(cx, "pat@x.com")
        assert rec is not None
        assert rec["items"] == [{"slug": "neuro-magnesium", "qty": 1}]
        assert rec["note"] == "keep going"
        assert rec["practitioner_id"] == PID

    # best-effort notification was attempted
    assert len(sent) == 1


def test_recommend_notification_failure_does_not_fail_the_recommend(monkeypatch, wired):
    app_module, db_path = wired
    _seed(db_path, "pat@x.com", PID, consent=True)

    from dashboard import inbox as _inbox
    def _boom(*a, **k):
        raise RuntimeError("smtp down")
    monkeypatch.setattr(_inbox, "send_email", _boom)

    client = app_module.app.test_client()
    r = client.post("/api/practitioner/continuity/recommend",
                    json={"patient_email": "pat@x.com", "items": [], "note": ""})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        pr.init_table(cx)
        assert pr.active_for_patient(cx, "pat@x.com") is not None
