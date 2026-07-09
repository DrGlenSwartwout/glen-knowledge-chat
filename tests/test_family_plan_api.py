"""Console endpoints to enroll, inspect and cancel a caregiver's Family Plan.

v1 enrollment mirrors the $300 tier's "just reply to arrange it": Glen enrols the
caregiver from the console after taking payment. Self-serve Stripe checkout comes
later; the entitlement is identical either way.
"""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import family_plan as fp
from dashboard import household as hh


CAREGIVER = "caregiver@example.com"
PET = "pet@example.com"
HDRS = {"X-Console-Key": "testkey"}


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


@pytest.fixture()
def client(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    with sqlite3.connect(tmp_db) as cx:
        hh.init_household_tables(cx)
        fp.init_family_plan_table(cx)
        hh.add_member(cx, CAREGIVER, PET, relationship="pet")
    return app.app.test_client()


def test_enroll_requires_the_console_key(client):
    r = client.post("/api/console/family-plan", json={"email": CAREGIVER})
    assert r.status_code == 401


def test_enroll_activates_the_plan(client, tmp_db):
    r = client.post("/api/console/family-plan", headers=HDRS,
                    json={"email": CAREGIVER, "next_charge_at": "2026-08-09"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        assert fp.is_active(cx, CAREGIVER) is True


def test_enrolling_a_caregiver_covers_their_member(client, tmp_db):
    client.post("/api/console/family-plan", headers=HDRS,
                json={"email": CAREGIVER, "next_charge_at": "2026-08-09"})
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        assert fp.covers(cx, PET) is True


def test_comped_enrollment_needs_no_charge_date(client, tmp_db):
    r = client.post("/api/console/family-plan", headers=HDRS,
                    json={"email": CAREGIVER, "source": "comp"})
    assert r.status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        row = fp.get(cx, CAREGIVER)
    assert row["source"] == "comp"
    assert row["next_charge_at"] is None


def test_enroll_rejects_a_missing_email(client):
    r = client.post("/api/console/family-plan", headers=HDRS, json={})
    assert r.status_code == 400


def test_enroll_is_idempotent(client, tmp_db):
    for _ in range(2):
        assert client.post("/api/console/family-plan", headers=HDRS,
                           json={"email": CAREGIVER, "source": "comp"}).status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM family_subscriptions").fetchone()[0]
    assert n == 1


def test_get_reports_status_and_covered_members(client):
    client.post("/api/console/family-plan", headers=HDRS,
                json={"email": CAREGIVER, "source": "comp"})
    r = client.get(f"/api/console/family-plan?email={CAREGIVER}", headers=HDRS)
    body = r.get_json()
    assert body["active"] is True
    assert PET in body["covered_members"]


def test_get_on_an_unenrolled_caregiver_is_inactive_not_404(client):
    r = client.get(f"/api/console/family-plan?email={CAREGIVER}", headers=HDRS)
    assert r.status_code == 200
    assert r.get_json()["active"] is False


def test_cancel_stops_covering_the_member(client, tmp_db):
    client.post("/api/console/family-plan", headers=HDRS,
                json={"email": CAREGIVER, "source": "comp"})
    r = client.post("/api/console/family-plan/cancel", headers=HDRS,
                    json={"email": CAREGIVER})
    assert r.status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        assert fp.covers(cx, PET) is False


def test_cancel_requires_the_console_key(client):
    r = client.post("/api/console/family-plan/cancel", json={"email": CAREGIVER})
    assert r.status_code == 401
