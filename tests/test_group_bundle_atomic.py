"""Tests for `_grant_group_bundle` claim-first atomicity (Task 1).

`_grant_group_bundle` now claims its `group_bundle_grants` marker via an
atomic `INSERT ... ON CONFLICT DO NOTHING` BEFORE doing any membership work
(create/extend + welcome email). This prevents a concurrent double-settle
(redirect + Stripe webhook both firing for the same invoice) from granting
two memberships and sending two welcome emails.

Scenarios:
  1. First grant  -> exactly one membership row + one welcome call; marker claimed.
  2. Second call, SAME invoice -> no-op: still one membership row, welcome not
     called again (rowcount==0 on the second claim attempt).
  3. Marker pre-claimed (simulating a concurrent winner) -> bails immediately:
     no membership created, welcome never called.
"""

import sqlite3

import app as appmod
from dashboard import stripe_pay as _stripe_pay_mod
from dashboard import subscriptions as subs


def _wire(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("GROUP_BUNDLE_ENABLED", "1")

    monkeypatch.setattr(
        _stripe_pay_mod, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1"})

    calls = []
    monkeypatch.setattr(appmod, "_member_join_welcome",
                        lambda cx, email, source=None: calls.append((email, source)))
    return db, calls


def _active(db, email="a@b.com"):
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        subs.init_subscriptions_table(cx)
        subs.migrate_add_membership_columns(cx)
        subs.migrate_add_term_cap_column(cx)
        subs.migrate_add_attribution_column(cx)
        subs.migrate_add_consent_column(cx)
        return subs.active_memberships_by_email(cx, email)


def _md():
    return {"grant_group_months": "1", "email": "a@b.com", "invoice_id": "tok1"}


# ── Test 1: first grant creates exactly one membership + one welcome ────────

def test_first_grant_creates_membership_and_welcome(monkeypatch, tmp_path):
    db, calls = _wire(monkeypatch, tmp_path)

    appmod._grant_group_bundle(_md(), "pi_1")

    rows = _active(db)
    assert len(rows) == 1
    assert len(calls) == 1
    assert calls[0][0] == "a@b.com"

    with sqlite3.connect(db) as cx:
        marker = cx.execute(
            "SELECT 1 FROM group_bundle_grants WHERE invoice_id=?",
            ("tok1",)).fetchone()
    assert marker is not None


# ── Test 2: a second call for the SAME invoice is a no-op ──────────────────

def test_second_call_same_invoice_is_noop(monkeypatch, tmp_path):
    db, calls = _wire(monkeypatch, tmp_path)

    appmod._grant_group_bundle(_md(), "pi_1")
    assert len(_active(db)) == 1
    assert len(calls) == 1

    # Same invoice again (e.g. redirect AND webhook both settle it).
    appmod._grant_group_bundle(_md(), "pi_1")

    rows = _active(db)
    assert len(rows) == 1, "must not create a second membership"
    assert len(calls) == 1, "welcome must not be sent twice"


# ── Test 3: marker pre-claimed by a concurrent run -> bail before any side effect ─

def test_preclaimed_marker_bails_before_side_effects(monkeypatch, tmp_path):
    db, calls = _wire(monkeypatch, tmp_path)

    # Simulate a concurrent winner: create the schema + claim the marker for
    # tok1 directly, as if another run (redirect vs webhook) already won the race.
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        subs.init_subscriptions_table(cx)
        subs.migrate_add_membership_columns(cx)
        subs.migrate_add_term_cap_column(cx)
        subs.migrate_add_attribution_column(cx)
        subs.migrate_add_consent_column(cx)
        cx.execute(
            "CREATE TABLE IF NOT EXISTS group_bundle_grants "
            "(invoice_id TEXT PRIMARY KEY, created_at TEXT)")
        cx.execute(
            "INSERT INTO group_bundle_grants (invoice_id, created_at) VALUES (?,?)",
            ("tok1", "2026-01-01T00:00:00+00:00"))
        cx.commit()

    appmod._grant_group_bundle(_md(), "pi_1")

    rows = _active(db)
    assert rows == [], "no membership should be created when marker is pre-claimed"
    assert calls == [], "welcome must never fire when marker is pre-claimed"


# ── Test 4: marker is claimed BEFORE create_membership is invoked ──────────
#
# Tests 1-3 above pass on both the fixed (claim-first) code and the old
# ordering (membership created, then marker inserted), because they only
# ever call `_grant_group_bundle` once or twice sequentially -- they never
# observe the *moment* create_membership runs. This test does: it spies on
# `subscriptions.create_membership` and checks, from inside the spy, whether
# the `group_bundle_grants` marker row already exists at call time. Only the
# claim-first ordering makes that true.

def test_marker_claimed_before_create_membership(monkeypatch, tmp_path):
    db, calls = _wire(monkeypatch, tmp_path)
    observed = {}

    real_create_membership = subs.create_membership

    def spy_create_membership(cx, **kwargs):
        marker = cx.execute(
            "SELECT 1 FROM group_bundle_grants WHERE invoice_id=?",
            ("tok1",)).fetchone()
        observed["marker_present_at_call"] = marker is not None
        # Perform the real insert so the flow completes normally (membership
        # row + welcome still fire as expected).
        return real_create_membership(cx, **kwargs)

    monkeypatch.setattr(subs, "create_membership", spy_create_membership)

    appmod._grant_group_bundle(_md(), "pi_1")

    assert observed.get("marker_present_at_call") is True, (
        "group_bundle_grants marker must already be committed BEFORE "
        "create_membership is called (claim-first ordering)")

    # Sanity: the flow still completed normally through the spy.
    assert len(_active(db)) == 1
    assert len(calls) == 1
