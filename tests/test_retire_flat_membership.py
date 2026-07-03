# tests/test_retire_flat_membership.py
"""Sub-project 6: retire OLD flat-membership framing.
Covers the two logic changes: (1) the receipt email names the right product
(Continuous Care vs group coaching), (2) the pause/cadence page is cancel-only
for a Continuous Care fixed-term member."""
import sqlite3
from datetime import datetime, timezone, timedelta
import app as appmod
from dashboard import subscriptions as subs


def _capture_email(monkeypatch):
    cap = {}
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda to, _x, subject, body: cap.update(
                            to=to, subject=subject, body=body) or ("smtp", None))
    return cap


def test_receipt_continuous_care_copy(monkeypatch):
    cap = _capture_email(monkeypatch)
    appmod._send_subscription_email(
        "x@y.com", "receipt",
        {"kind": "membership", "product": "continuous_care", "total_cents": 9900,
         "next_charge_date": "2026-08-01", "invoice_id": "pi_1",
         "cancel_url": "https://test.example/membership/cancel/TOK"})
    assert "Continuous Care" in cap["subject"]
    assert "Continuous Care" in cap["body"]
    assert "group coaching" not in cap["body"].lower()   # the bug: no more group-coaching copy
    assert "https://test.example/membership/cancel/TOK" in cap["body"]  # cancel link folded in


def test_receipt_group_coaching_copy_unchanged(monkeypatch):
    cap = _capture_email(monkeypatch)
    appmod._send_subscription_email(
        "x@y.com", "receipt",
        {"kind": "membership", "product": "group_coaching", "total_cents": 9900,
         "next_charge_date": "2026-08-01"})
    assert "group coaching" in cap["body"].lower()        # legacy bundle copy preserved
    assert "Continuous Care" not in cap["subject"]


def test_receipt_defaults_to_group_coaching_without_product(monkeypatch):
    cap = _capture_email(monkeypatch)
    appmod._send_subscription_email(
        "x@y.com", "receipt", {"kind": "membership", "total_cents": 9900})
    assert "group coaching" in cap["body"].lower()        # back-compat: no product key


# -- pause route: cancel-only for Continuous Care ---------------------------

def _seed_token_and_sub(db, email, *, term_cap):
    with sqlite3.connect(db) as cx:
        subs.init_subscriptions_table(cx)
        subs.migrate_add_membership_columns(cx)
        subs.migrate_add_term_cap_column(cx)
        subs.create_membership(cx, email=email, stripe_customer_id="cus",
                               stripe_payment_method_id="pm", amount_cents=9900,
                               next_charge_date="2026-08-01", cadence_months=1,
                               term_charges_total=term_cap, initial_order_count=1)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens "
                   "(token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT)")
        exp = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
                   "VALUES (?,?,?,?,?)",
                   (appmod._hash_token("TOK"), email, "membership_cancel",
                    datetime.now(timezone.utc).isoformat(), exp))
        cx.commit()


def test_pause_page_cancel_only_for_continuous_care(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "PUBLIC_BASE_URL", "https://test.example")
    _seed_token_and_sub(db, "cc@x.com", term_cap=6)   # capped term = Continuous Care
    r = appmod.app.test_client().get("/membership/pause/TOK")
    assert r.status_code == 200
    assert b'"continuous_care": true' in r.data
    assert b'/membership/cancel/TOK' in r.data
    # No cadence/pause controls surfaced for a committed term.
    assert b'"paused_charge_date"' not in r.data


def test_pause_post_does_not_pause_continuous_care(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "PUBLIC_BASE_URL", "https://test.example")
    _seed_token_and_sub(db, "cc2@x.com", term_cap=12)
    # A POST that would pause/switch cadence must be refused for a CC term.
    r = appmod.app.test_client().post("/membership/pause/TOK", data={"mode": "once"})
    assert r.status_code == 200
    assert b'"continuous_care": true' in r.data
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        row = subs.active_memberships_by_email(cx, "cc2@x.com")[0]
    assert not row["skip_next"]   # never paused


def test_pause_page_still_works_for_uncapped_membership(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "PUBLIC_BASE_URL", "https://test.example")
    _seed_token_and_sub(db, "gc@x.com", term_cap=None)   # uncapped = legacy group coaching
    r = appmod.app.test_client().get("/membership/pause/TOK")
    assert r.status_code == 200
    assert b'"continuous_care": true' not in r.data
    assert b'"paused_charge_date"' in r.data   # normal pause preview still offered
