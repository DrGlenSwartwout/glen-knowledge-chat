"""Tests for the console practitioner-admin logic layer (dashboard/practitioner_admin.py).

The Supabase-touching functions (list/create/update) are exercised at the route
level with monkeypatched cursors; here we test the pure pieces: input validation,
SQLite activity aggregation, and the row-merge that feeds the roster UI.
"""
import sqlite3

import pytest

from dashboard import practitioner_admin as pa
from dashboard import practitioner_portal as pp


# ── validate_new_practitioner ──────────────────────────────────────────────────

def test_validate_requires_valid_email():
    clean, err = pa.validate_new_practitioner({"email": "nope", "name": "A", "role": "coach"})
    assert clean is None
    assert "email" in err.lower()


def test_validate_requires_name():
    clean, err = pa.validate_new_practitioner({"email": "a@b.com", "name": "", "role": "coach"})
    assert clean is None
    assert "name" in err.lower()


def test_validate_requires_known_role():
    clean, err = pa.validate_new_practitioner({"email": "a@b.com", "name": "A", "role": "wizard"})
    assert clean is None
    assert "role" in err.lower()


def test_validate_clamps_level_to_0_12():
    clean, err = pa.validate_new_practitioner(
        {"email": "a@b.com", "name": "A", "role": "coach", "level": 99})
    assert err is None
    assert clean["level"] == 12
    clean2, _ = pa.validate_new_practitioner(
        {"email": "a@b.com", "name": "A", "role": "coach", "level": -3})
    assert clean2["level"] == 0


def test_validate_rejects_non_numeric_level():
    clean, err = pa.validate_new_practitioner(
        {"email": "a@b.com", "name": "A", "role": "coach", "level": "lots"})
    assert clean is None
    assert "level" in err.lower()


def test_validate_normalizes_and_passes_flags():
    clean, err = pa.validate_new_practitioner({
        "email": "  AKing@Yahoo.com ", "name": "  Ashley King ", "role": "coach",
        "credentials": "Health Coach", "wholesale_access": True, "level": 0,
        "list_in_finder": True, "city": "Austin", "state": "tx", "send_invite": True})
    assert err is None
    assert clean["email"] == "aking@yahoo.com"
    assert clean["name"] == "Ashley King"
    assert clean["portal_role"] == "coach"
    assert clean["credentials"] == "Health Coach"
    assert clean["wholesale_access"] is True
    assert clean["level"] == 0
    assert clean["list_in_finder"] is True
    assert clean["city"] == "Austin"
    assert clean["state"] == "TX"
    assert clean["send_invite"] is True


# ── aggregate_activity ──────────────────────────────────────────────────────────

def _seed_orders(db_path):
    with sqlite3.connect(db_path) as cx:
        pp._ensure_orders_table(cx)
        pp._ensure_dispensary_table(cx)
        cx.executemany(
            "INSERT INTO wholesale_orders "
            "(invoice_id, practitioner_id, doc_number, total_cents, credit_cents, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [("i1", "p1", "1001", 7000, 0, "2026-06-01T00:00:00"),
             ("i2", "p1", "1002", 3500, 0, "2026-06-10T00:00:00"),
             ("i3", "p2", "1003", 9000, 0, "2026-06-05T00:00:00")])
        cx.executemany(
            "INSERT INTO dispensary_orders "
            "(invoice_id, practitioner_id, customer_email, bottles, credit_earned_cents, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [("d1", "p1", "c@x.com", 2, 1400, "2026-06-02T00:00:00"),
             ("d2", "p1", "c2@x.com", 1, 700, "2026-06-03T00:00:00")])
        cx.commit()


def test_aggregate_activity_sums_orders_per_practitioner(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed_orders(db)
    agg = pa.aggregate_activity(db)
    assert agg["p1"]["orders"] == 2
    assert agg["p1"]["spent_cents"] == 10500
    assert agg["p1"]["last_order"] == "2026-06-10T00:00:00"
    assert agg["p2"]["orders"] == 1
    assert agg["p2"]["spent_cents"] == 9000


def test_aggregate_activity_sums_dispensary(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed_orders(db)
    agg = pa.aggregate_activity(db)
    assert agg["p1"]["disp_count"] == 2
    assert agg["p1"]["disp_credit_cents"] == 2100
    assert agg["p1"]["disp_bottles"] == 3
    assert "p2" not in agg or agg["p2"].get("disp_count", 0) == 0


def test_aggregate_activity_empty_db_is_empty(tmp_path):
    db = str(tmp_path / "chat_log.db")
    agg = pa.aggregate_activity(db)
    assert agg == {}


# ── build_rows ──────────────────────────────────────────────────────────────────

def test_build_rows_merges_activity_and_defaults_zeros():
    practitioners = [
        {"id": "p1", "name": "Ashley King", "email": "a@b.com", "portal_role": "coach",
         "credentials": "Health Coach", "modules_completed": 0, "wallet_balance_cents": 0,
         "wholesale_unlocked_at": "2026-06-23T00:00:00", "application_status": None,
         "show_contact": True, "city": "Austin", "state": "TX"},
        {"id": "p2", "name": "Dr Who", "email": "w@b.com", "portal_role": "licensed",
         "credentials": "OD", "modules_completed": 5, "wallet_balance_cents": 1500,
         "wholesale_unlocked_at": None, "application_status": "pending",
         "show_contact": False, "city": None, "state": None},
    ]
    activity = {"p1": {"orders": 2, "spent_cents": 10500, "last_order": "2026-06-10T00:00:00",
                       "disp_count": 2, "disp_credit_cents": 2100, "disp_bottles": 3}}
    rows = pa.build_rows(practitioners, activity)
    by_id = {r["id"]: r for r in rows}
    # p1 gets its activity + derived booleans + section
    assert by_id["p1"]["wholesale_access"] is True
    assert by_id["p1"]["finder_listed"] is True
    assert by_id["p1"]["section"] == "coach"
    assert by_id["p1"]["orders"] == 2
    assert by_id["p1"]["spent_cents"] == 10500
    # p2 has no activity → zeros, locked, hidden, practitioner section
    assert by_id["p2"]["wholesale_access"] is False
    assert by_id["p2"]["finder_listed"] is False
    assert by_id["p2"]["section"] == "practitioner"
    assert by_id["p2"]["orders"] == 0
    assert by_id["p2"]["spent_cents"] == 0
    assert by_id["p2"]["disp_count"] == 0
