# tests/test_cc_signup_alert.py
"""Email Glen ONCE on the first Continuous Care signup (monthly or up-front).
Covers the once-guard directly and through the monthly fulfiller."""
import sqlite3
import app as appmod


def _capture_glen(monkeypatch):
    sent = []
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda to, name, subject, body: sent.append(
                            {"to": to, "subject": subject, "body": body}) or ("smtp", None))
    return sent


def test_first_cc_signup_emails_glen_once(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "log.db"))
    monkeypatch.setenv("GLEN_EMAIL", "glen@example.com")
    sent = _capture_glen(monkeypatch)
    appmod._notify_first_cc_signup("client@x.com", "Continuous Care monthly (6mo term)", 9900)
    assert len(sent) == 1
    assert sent[0]["to"] == "glen@example.com"
    assert "Continuous Care" in sent[0]["subject"]
    assert "client@x.com" in sent[0]["body"]
    assert "6mo term" in sent[0]["body"]
    assert "$99.00" in sent[0]["body"]
    # Second signup must NOT re-alert (once-guard).
    appmod._notify_first_cc_signup("client2@x.com", "Continuous Care up front (12 months)", 99000)
    assert len(sent) == 1


def test_alert_never_raises_on_db_error(monkeypatch):
    # Point LOG_DB at an unusable path — the helper must swallow and not raise.
    monkeypatch.setattr(appmod, "LOG_DB", "/nonexistent-dir/nope.db")
    appmod._notify_first_cc_signup("x@x.com", "Continuous Care monthly (6mo term)", 9900)  # no exception


# -- integration through the monthly fulfiller --------------------------------

def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    monkeypatch.setattr(app_module, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        subscriptions.migrate_add_term_cap_column(cx)
        app_module.init_membership_tables(cx)
        cx.commit()
    monkeypatch.setattr(app_module, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(app_module, "_send_subscription_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app_module, "_mint_membership_cancel_url", lambda *a, **k: "", raising=False)
    return db


def _mock_stripe_ok(app_module, monkeypatch, email, term="6"):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda sid: {"metadata": {"email": email, "kind": "continuous_care_monthly",
                                  "term_months": term}, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi_id: {"status": "succeeded", "id": "pi_1", "customer": "cus_1",
                       "payment_method": "pm_1"})


def test_monthly_fulfiller_alerts_glen_on_first_only(monkeypatch, tmp_path):
    _fresh(appmod, monkeypatch, tmp_path)
    sent = _capture_glen(monkeypatch)
    _mock_stripe_ok(appmod, monkeypatch, "first@x.com")
    appmod._fulfill_continuous_care_monthly("sess_a")
    assert len(sent) == 1 and "first@x.com" in sent[0]["body"]
    # A different signup (distinct session + email) must NOT re-alert.
    _mock_stripe_ok(appmod, monkeypatch, "second@x.com")
    appmod._fulfill_continuous_care_monthly("sess_b")
    assert len(sent) == 1
