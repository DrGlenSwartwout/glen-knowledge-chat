# tests/test_membership_reconcile_alerts.py
"""Ops alert for the duplicate-member orphaned-charge case in
_fulfill_membership_product: when a year_monthly (recurring_capped) checkout
lands on an email that already has an active membership, no new subscription
is created for the month-1 charge just taken -- it needs manual reconcile.
This durably records that in membership_reconcile_alerts (idempotent on
session_id) and exposes it via an owner-only console endpoint.

Fixture style mirrors tests/test_membership_products_fulfill.py (tmp-file
sqlite via monkeypatched LOG_DB, Stripe faked by monkeypatching
dashboard.stripe_pay, QBO faked so the one-time-tier booking never hits the
network).
"""
import importlib, sys, os, sqlite3
import pytest


def _load_app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app import failed: {e}")


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    app = _load_app()
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(app, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app, "PUBLIC_BASE_URL", "https://illtowell.com", raising=False)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "sekret", raising=False)
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sekret", raising=False)
    from dashboard import subscriptions as subs
    cx = sqlite3.connect(db)
    subs.init_subscriptions_table(cx)
    for name in dir(subs):
        if name.startswith("migrate_add"):
            try: getattr(subs, name)(cx)
            except Exception: pass
    app.init_membership_tables(cx)
    cx.close()
    for fn in ("_ingest_order", "_member_join_welcome"):
        if hasattr(app, fn): monkeypatch.setattr(app, fn, lambda *a, **k: None, raising=False)
    import dashboard.qbo_billing as qb
    monkeypatch.setattr(qb, "find_or_create_customer", lambda *a, **k: {"Id": "1"})
    monkeypatch.setattr(qb, "create_invoice", lambda *a, **k: {"Id": "inv1"})
    monkeypatch.setattr(qb, "record_payment", lambda *a, **k: {"Id": "pay1"})
    return app


def _mock_stripe(appmod, monkeypatch, sid_metadata_email="a@x.com"):
    def _get_session(sid):
        return {
            "id": sid, "payment_status": "paid",
            "metadata": {"kind": "membership_product", "tier": "year_monthly",
                         "email": sid_metadata_email},
            "payment_intent": "pi_" + sid, "customer": "cus_1",
        }
    monkeypatch.setattr(appmod.stripe_pay, "get_session", _get_session)
    monkeypatch.setattr(appmod.stripe_pay, "get_payment_intent", lambda pi: {
        "status": "succeeded", "customer": "cus_1", "payment_method": "pm_1"})


def _drive_to_duplicate_member(appmod, monkeypatch):
    """First fulfillment creates the active membership; second (different
    session, same email) hits the duplicate_member branch."""
    _mock_stripe(appmod, monkeypatch)
    assert appmod._fulfill_membership_product("cs_first") == "ok"
    result = appmod._fulfill_membership_product("cs_dup")
    assert result == "duplicate_member"
    return "cs_dup"


def test_duplicate_member_writes_reconcile_alert(appmod, monkeypatch):
    sid = _drive_to_duplicate_member(appmod, monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT * FROM membership_reconcile_alerts WHERE session_id=?", (sid,)).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["email"] == "a@x.com"
    assert row["tier"] == "year_monthly"
    assert row["amount_cents"] == 9900
    assert row["status"] == "open"


def test_duplicate_member_alert_insert_idempotent(appmod, monkeypatch):
    sid = _drive_to_duplicate_member(appmod, monkeypatch)
    # Replay the exact same session -- claim-then-create means the fulfiller
    # itself would treat this as "already", but drive the alert insert helper
    # again directly to prove the INSERT OR IGNORE keying holds.
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    from dashboard import membership_products as mp
    appmod._record_membership_reconcile_alert(cx, sid, "a@x.com", mp.get_tier("year_monthly"))
    cx.close()
    cx = sqlite3.connect(appmod.LOG_DB)
    n = cx.execute(
        "SELECT COUNT(*) FROM membership_reconcile_alerts WHERE session_id=?", (sid,)).fetchone()[0]
    assert n == 1


def test_reconcile_alerts_endpoint_owner(appmod, monkeypatch):
    sid = _drive_to_duplicate_member(appmod, monkeypatch)
    r = appmod.app.test_client().get("/api/console/membership/reconcile-alerts?key=sekret")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert len(body["alerts"]) == 1
    assert body["alerts"][0]["session_id"] == sid
    assert body["alerts"][0]["email"] == "a@x.com"


def test_reconcile_alerts_endpoint_rejects_bad_key(appmod, monkeypatch):
    _drive_to_duplicate_member(appmod, monkeypatch)
    r = appmod.app.test_client().get("/api/console/membership/reconcile-alerts?key=wrong")
    assert r.status_code == 401


def _seed_va_token(appmod, token):
    """Same seeding pattern as tests/test_membership_enroll_endpoint.py: a
    workspace_users row scoped 'workspace:shaira' + an access_tokens row
    resolves to a real, non-None VA-role actor (not the actor-is-None branch)."""
    appmod._init_workspace_schema()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("INSERT INTO workspace_users (name, display_name, scope) VALUES (?,?,?)",
                   ("shaira", "Shaira", "workspace:shaira"))
        uid = cx.execute("SELECT id FROM workspace_users WHERE name=?", ("shaira",)).fetchone()[0]
        cx.execute("INSERT INTO access_tokens (token, user_id, note) VALUES (?,?,?)",
                   (token, uid, "t"))
        cx.commit()


def test_reconcile_alerts_endpoint_rejects_va_actor(appmod, monkeypatch):
    _drive_to_duplicate_member(appmod, monkeypatch)
    _seed_va_token(appmod, "sha-tok")
    assert appmod._role_for_token("sha-tok") == appmod._bos_rbac.VA
    r = appmod.app.test_client().get("/api/console/membership/reconcile-alerts?key=sha-tok")
    assert r.status_code == 401
