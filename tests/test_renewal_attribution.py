import sqlite3, importlib, sys
from pathlib import Path

def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    return importlib.import_module("app")

def _seed(path):
    cx = sqlite3.connect(path)
    cx.execute("CREATE TABLE prepay_term_grants (session_id TEXT PRIMARY KEY, email TEXT, tier_key TEXT, granted_at TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INTEGER DEFAULT 0, term_end TEXT)")
    cx.execute("CREATE TABLE subscriptions (email TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INTEGER, kind TEXT, created_at TEXT)")
    cx.commit(); cx.close()

def test_none_when_no_history(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    assert app._last_attributed_practitioner("pat@x.com", db_path=p) is None

def test_prepay_grant_returned(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    cx = sqlite3.connect(p); cx.execute("INSERT INTO prepay_term_grants (session_id,email,granted_at,attributed_practitioner_id,practitioner_share_consent) VALUES ('s1','pat@x.com','2026-01-01T00:00:00Z','prac-42',1)"); cx.commit(); cx.close()
    assert app._last_attributed_practitioner("PAT@x.com", db_path=p) == {"pid": "prac-42", "consent": 1}

def test_most_recent_across_sources(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    cx = sqlite3.connect(p)
    cx.execute("INSERT INTO prepay_term_grants (session_id,email,granted_at,attributed_practitioner_id,practitioner_share_consent) VALUES ('s1','pat@x.com','2026-01-01T00:00:00Z','prac-A',1)")
    cx.execute("INSERT INTO subscriptions (email,attributed_practitioner_id,practitioner_share_consent,kind,created_at) VALUES ('pat@x.com','prac-B',0,'membership','2026-06-01T00:00:00Z')")
    cx.commit(); cx.close()
    assert app._last_attributed_practitioner("pat@x.com", db_path=p) == {"pid": "prac-B", "consent": 0}  # later wins

def test_ignores_unattributed_rows(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    cx = sqlite3.connect(p); cx.execute("INSERT INTO prepay_term_grants (session_id,email,granted_at,attributed_practitioner_id) VALUES ('s1','pat@x.com','2026-01-01T00:00:00Z',NULL)"); cx.commit(); cx.close()
    assert app._last_attributed_practitioner("pat@x.com", db_path=p) is None


# ---------------------------------------------------------------------------
# Task 2: _fulfill_prepay_term inherits the patient's prior attribution when
# the NEW session carries no explicit dispensary_pid (sticky renewal). Reuses
# the Stripe-stub pattern from tests/test_care_share_prepay.py.
# ---------------------------------------------------------------------------

def _fresh_app(tmp_path, monkeypatch):
    app = _app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    monkeypatch.setattr(app, "REPERTOIRE_ENABLED", False, raising=False)
    monkeypatch.setattr(app, "PUBLIC_BASE_URL", "", raising=False)
    monkeypatch.setattr(app, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(app, "_send_inquiry_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app, "_notify_first_cc_signup", lambda *a, **k: None, raising=False)
    with sqlite3.connect(db) as cx:
        app.init_membership_tables(cx)
        cx.commit()
    return app, db


def _mock_prepay_session(app, monkeypatch, *, email, tier_key,
                          dispensary_pid=None, share_consent=None):
    from dashboard import stripe_pay
    md = {"kind": "prepay_term", "email": email, "tier_key": tier_key}
    if dispensary_pid is not None:
        md["dispensary_pid"] = dispensary_pid
    if share_consent is not None:
        md["share_consent"] = share_consent
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": dict(md), "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})


def _patch_care_share(monkeypatch, *, modules=12):
    from dashboard import care_share, wallet
    monkeypatch.setattr(care_share, "modules_for_practitioner", lambda pid: modules)
    rec = []
    monkeypatch.setattr(wallet, "earn_care_share",
        lambda pid, cents, *, event_ref: rec.append((pid, cents, event_ref)) or cents)
    return rec


def _grant_row(db, session_id):
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        return cx.execute(
            "SELECT * FROM prepay_term_grants WHERE session_id=?", (session_id,)).fetchone()


def _seed_prior_grant(app, db, *, email, pid, consent,
                       granted_at="2026-01-01T00:00:00Z", session_id="cs_prior"):
    """Seed a PRIOR attributed prepay grant for `email` — renewal history that
    _last_attributed_practitioner should pick up (a separate, earlier session
    from the one being fulfilled in the test)."""
    with sqlite3.connect(db) as cx:
        cx.execute(
            "CREATE TABLE IF NOT EXISTS prepay_term_grants "
            "(session_id TEXT PRIMARY KEY, email TEXT, tier_key TEXT, granted_at TEXT)")
        app._ensure_prepay_grant_columns(cx)
        cx.execute(
            "INSERT INTO prepay_term_grants "
            "(session_id, email, tier_key, granted_at, attributed_practitioner_id, "
            "practitioner_share_consent) VALUES (?,?,?,?,?,?)",
            (session_id, email, "12mo", granted_at, pid, consent))
        cx.commit()


def test_prepay_renewal_inherits_prior_attribution(monkeypatch, tmp_path):
    app, db = _fresh_app(tmp_path, monkeypatch)
    rec = _patch_care_share(monkeypatch, modules=12)
    _seed_prior_grant(app, db, email="pat@x.com", pid="prac-42", consent=1)

    _mock_prepay_session(app, monkeypatch, email="pat@x.com", tier_key="12mo")  # no dispensary_pid
    res = app._fulfill_prepay_term("cs_renew")
    assert res.get("ok") is True

    row = _grant_row(db, "cs_renew")
    assert row["attributed_practitioner_id"] == "prac-42"
    assert row["practitioner_share_consent"] == 1

    from dashboard import care_share
    expected = care_share.share_cents(99000, 12)
    assert rec == [("prac-42", expected, "care_share:prepay:cs_renew")]


def test_prepay_explicit_dispensary_pid_wins(monkeypatch, tmp_path):
    app, db = _fresh_app(tmp_path, monkeypatch)
    rec = _patch_care_share(monkeypatch, modules=12)
    _seed_prior_grant(app, db, email="pat2@x.com", pid="prac-42", consent=1)

    _mock_prepay_session(app, monkeypatch, email="pat2@x.com", tier_key="12mo",
                          dispensary_pid="prac-99", share_consent="1")
    res = app._fulfill_prepay_term("cs_explicit")
    assert res.get("ok") is True

    row = _grant_row(db, "cs_explicit")
    assert row["attributed_practitioner_id"] == "prac-99"
    assert row["practitioner_share_consent"] == 1
    assert rec and rec[0][0] == "prac-99"


def test_prepay_no_prior_no_explicit_unattributed(monkeypatch, tmp_path):
    app, db = _fresh_app(tmp_path, monkeypatch)
    rec = _patch_care_share(monkeypatch, modules=12)

    _mock_prepay_session(app, monkeypatch, email="fresh@x.com", tier_key="12mo")
    res = app._fulfill_prepay_term("cs_fresh")
    assert res.get("ok") is True

    row = _grant_row(db, "cs_fresh")
    assert row["attributed_practitioner_id"] in (None, "")
    assert rec == []


# ---------------------------------------------------------------------------
# Task 3: _fulfill_continuous_care_monthly inherits the patient's prior
# attribution when the NEW session carries no explicit dispensary_pid (sticky
# renewal). Reuses the Stripe-stub pattern from tests/test_care_share_enroll.py.
# ---------------------------------------------------------------------------

def _fresh_monthly_app(tmp_path, monkeypatch):
    app = _app()
    db = str(tmp_path / "chat_log_monthly.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        subscriptions.migrate_add_term_cap_column(cx)
        subscriptions.migrate_add_attribution_column(cx)
        subscriptions.migrate_add_consent_column(cx)
        app.init_membership_tables(cx)
        cx.commit()
    monkeypatch.setattr(app, "REPERTOIRE_ENABLED", False, raising=False)
    monkeypatch.setattr(app, "PUBLIC_BASE_URL", "", raising=False)
    monkeypatch.setattr(app, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(app, "_send_subscription_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app, "_send_inquiry_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app, "_notify_first_cc_signup", lambda *a, **k: None, raising=False)
    return app, db


def _mock_monthly_session(app, monkeypatch, *, email, term_months=12,
                           dispensary_pid=None, share_consent=None):
    from dashboard import stripe_pay
    md = {"kind": "continuous_care_monthly", "email": email,
          "term_months": str(term_months)}
    if dispensary_pid is not None:
        md["dispensary_pid"] = dispensary_pid
    if share_consent is not None:
        md["share_consent"] = share_consent
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": dict(md), "payment_intent": "pi_m1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_m1", "payment_method": "pm_m1", "status": "succeeded"})


def _seed_prior_membership(db, *, email, pid, consent,
                            created_at="2026-01-01T00:00:00Z"):
    """Seed a PRIOR attributed (now-cancelled) membership for `email` — renewal
    history that _last_attributed_practitioner should pick up. Cancelled so it
    doesn't trip the active-membership duplicate-guard in the fulfilment path
    under test."""
    with sqlite3.connect(db) as cx:
        cx.execute(
            "INSERT INTO subscriptions (email, stripe_customer_id, stripe_payment_method_id, "
            "items_json, cadence_months, status, order_count, next_charge_date, "
            "ship_address_json, skip_next, created_at, updated_at, kind, amount_cents, "
            "attributed_practitioner_id, practitioner_share_consent) "
            "VALUES (?,?,?,'[]',1,'cancelled',0,?, '{}',0,?,?,'membership',9900,?,?)",
            (email, "cus_prior", "pm_prior", created_at, created_at, created_at, pid, consent))
        cx.commit()


def test_monthly_reenroll_inherits_prior_attribution(monkeypatch, tmp_path):
    app, db = _fresh_monthly_app(tmp_path, monkeypatch)
    _seed_prior_membership(db, email="pat3@x.com", pid="prac-42", consent=1)

    from dashboard import care_share, subscriptions as subs
    rec = []
    monkeypatch.setattr(care_share, "credit_for_charge",
        lambda sub, *, charge_cents, **kw: rec.append(
            (sub.get("attributed_practitioner_id"), charge_cents)))

    _mock_monthly_session(app, monkeypatch, email="pat3@x.com")  # no dispensary_pid
    res = app._fulfill_continuous_care_monthly("cs_monthly_renew")
    assert res.get("ok") is True

    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "pat3@x.com")
    assert len(rows) == 1
    row = rows[0]
    assert row["attributed_practitioner_id"] == "prac-42"
    assert row["practitioner_share_consent"] == 1

    assert rec == [("prac-42", 9900)]


def test_monthly_explicit_dispensary_pid_wins(monkeypatch, tmp_path):
    app, db = _fresh_monthly_app(tmp_path, monkeypatch)
    _seed_prior_membership(db, email="pat4@x.com", pid="prac-42", consent=1)

    from dashboard import care_share, subscriptions as subs
    monkeypatch.setattr(care_share, "credit_for_charge", lambda sub, **kw: None)

    _mock_monthly_session(app, monkeypatch, email="pat4@x.com",
                          dispensary_pid="prac-99", share_consent="1")
    res = app._fulfill_continuous_care_monthly("cs_monthly_explicit")
    assert res.get("ok") is True

    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "pat4@x.com")
    assert len(rows) == 1
    assert rows[0]["attributed_practitioner_id"] == "prac-99"
