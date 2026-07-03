"""Tests for the one-shot admin trigger POST /api/console/repertoire-reseed.

Retroactively populates dashboard.repertoire for every CURRENTLY-PAID member
(active memberships row OR active kind='membership' subscription, gated
through the same _is_paid_member check the discount system uses) from their
purchase_history (365-day window). Console-gated only (require_console_key +
ok/fail — see /api/console/fmp-history-rebuild, tests/test_fmp_history_rebuild_route.py),
NOT feature-flag gated: this is a manual one-shot admin action, harmless
because the discount system already reads repertoire under REPERTOIRE_ENABLED.
"""
import sqlite3
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    import dashboard as _dashboard
    from dashboard import subscriptions as _subs

    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(_dashboard, "CONSOLE_SECRET", "test-secret")

    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.init_membership_tables(cx)
        _subs.init_subscriptions_table(cx)
        _subs.migrate_add_membership_columns(cx)
        cx.commit()

    appmod.app.config["TESTING"] = True
    return appmod


@pytest.fixture
def client(appmod):
    return appmod.app.test_client()


def _future_iso(days=30):
    return (datetime.utcnow() + timedelta(days=days)).isoformat() + "Z"


def _seed_active_membership(appmod, email, *, source="founding"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute(
            "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source) "
            "VALUES (?,?,?,?,?,?)",
            (f"mem_{email}", email, datetime.utcnow().isoformat() + "Z", _future_iso(),
             "test", source))
        cx.commit()


def _seed_purchase_history(appmod, email, slugs, *, days_ago=10):
    from dashboard import purchase_history as _ph
    purchased_at = (datetime.utcnow() - timedelta(days=days_ago)).isoformat() + "Z"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _ph.init_purchase_history_table(cx)
        for i, slug in enumerate(slugs):
            cx.execute(
                "INSERT OR IGNORE INTO purchase_history (email, slug, purchased_at, source, source_ref) "
                "VALUES (?,?,?,?,?)",
                (email, slug, purchased_at, "fmp", f"{email}-{i}"))
        cx.commit()


def _repertoire_slugs(appmod, email):
    from dashboard import repertoire as _rep
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _rep.init_repertoire_table(cx)
        return _rep.repertoire_slugs(cx, email)


def test_reseed_requires_console_key(client):
    r = client.post("/api/console/repertoire-reseed")
    assert r.status_code == 401


def test_reseed_populates_repertoire_for_active_members_only(appmod, client):
    # Two ACTIVE paid members with purchase history in the 365-day window.
    _seed_active_membership(appmod, "alice@x.com")
    _seed_purchase_history(appmod, "alice@x.com", ["neuro-magnesium", "terrain-restore"])

    _seed_active_membership(appmod, "bob@x.com")
    _seed_purchase_history(appmod, "bob@x.com", ["terrain-restore"])

    # A NON-member with purchase history must NOT be reseeded.
    _seed_purchase_history(appmod, "carol@x.com", ["neuro-magnesium"])

    r = client.post("/api/console/repertoire-reseed", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    data = body["data"]
    assert data["members_seen"] == 2
    assert data["members_reseeded"] == 2
    assert data["slugs_added"] == 3

    assert _repertoire_slugs(appmod, "alice@x.com") == {"neuro-magnesium", "terrain-restore"}
    assert _repertoire_slugs(appmod, "bob@x.com") == {"terrain-restore"}
    assert _repertoire_slugs(appmod, "carol@x.com") == set()


def test_reseed_is_idempotent_on_rerun(appmod, client):
    _seed_active_membership(appmod, "dana@x.com")
    _seed_purchase_history(appmod, "dana@x.com", ["neuro-magnesium"])

    r1 = client.post("/api/console/repertoire-reseed", headers={"X-Console-Key": "test-secret"})
    assert r1.get_json()["data"]["slugs_added"] == 1

    r2 = client.post("/api/console/repertoire-reseed", headers={"X-Console-Key": "test-secret"})
    body2 = r2.get_json()["data"]
    assert body2["members_seen"] == 1
    assert body2["slugs_added"] == 0
    assert body2["members_reseeded"] == 0


def test_reseed_candidate_from_subscription_and_grant_is_not_double_counted(appmod, client):
    """A member who shows up in BOTH candidate sources (an active kind='membership'
    subscription row AND an active memberships grant row for the same email) must
    be counted once, not twice, by the DISTINCT union. (_is_paid_member itself is
    gated on the memberships-grant table — see reseed-report.md "concerns" — so a
    subscription-only member with no grant row would not pass the paid-member
    filter; this fixture gives 'erin' both so the union's dedup is exercised
    without depending on that separate gate quirk.)"""
    from dashboard import subscriptions as _subs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute(
            "INSERT INTO subscriptions (email, stripe_customer_id, stripe_payment_method_id, "
            "kind, status, cadence_months, order_count, next_charge_date, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("erin@x.com", "cus_1", "pm_1", "membership", "active", 1, 1,
             _future_iso(), _subs._now_iso(), _subs._now_iso()))
        cx.commit()
    _seed_active_membership(appmod, "erin@x.com")
    _seed_purchase_history(appmod, "erin@x.com", ["terrain-restore"])

    r = client.post("/api/console/repertoire-reseed", headers={"X-Console-Key": "test-secret"})
    body = r.get_json()["data"]
    assert body["members_seen"] == 1
    assert body["members_reseeded"] == 1
    assert body["slugs_added"] == 1
    assert _repertoire_slugs(appmod, "erin@x.com") == {"terrain-restore"}
