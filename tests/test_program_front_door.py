# tests/test_program_front_door.py
"""Task 2 of Program -> deposit front door: a paid program (biofield) purchase grants a
30-day Continuous Care taster window behind PROGRAM_CARE_TASTER_ENABLED. Source
"care_taster" (NOT "biofield_trial") so it reads as a paid grant (member pricing kept).

Mirrors the fixture pattern from tests/test_prepay_checkout.py and drives the biofield
return the way tests/test_biofield_checkout.py does.
"""
import importlib, sqlite3, sys
from datetime import datetime
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    monkeypatch.setattr(app_module, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        app_module.init_membership_tables(cx)
        cx.commit()
    # Keep the return path's best-effort side-effects out of the test.
    monkeypatch.setattr(app_module, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: None, raising=False)
    return db


def _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com",
                                 invoice_id="INVP", customer_id="C1"):
    from dashboard import stripe_pay as _sp
    monkeypatch.setattr(_sp, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid", "amount_total": 30000,
        "payment_intent": "pi_1",
        "metadata": {"kind": "biofield", "email": email, "tier": "scalable",
                     "invoice_id": invoice_id, "customer_id": customer_id,
                     "points_redeemed_cents": "0"},
    })
    monkeypatch.setattr(app_module.stripe_pay, "get_session", _sp.get_session, raising=False)
    monkeypatch.setattr(_sp, "get_payment_intent",
        lambda pi: {"customer": customer_id, "payment_method": "pm_1", "status": "succeeded"})
    monkeypatch.setattr(app_module.stripe_pay, "get_payment_intent", _sp.get_payment_intent,
                        raising=False)

    # Neutralize the other best-effort side-effects on this same branch (QBO, points,
    # biofield_store) so the test is scoped to the care-taster grant.
    from dashboard import qbo_billing as _qb
    monkeypatch.setattr(_qb, "record_payment", lambda *a, **k: None)
    monkeypatch.setattr(app_module._bos_orders, "find_order_by_external_ref",
                        lambda cx, ref: None)


def test_program_return_grants_one_care_taster_window(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", True, raising=False)
    _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com")
    r = app_module.app.test_client().get("/begin/checkout-return?session_id=cs_1")
    assert r.status_code == 302

    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        grants = cx.execute(
            "SELECT granted_at, expires_at, source FROM memberships "
            "WHERE email='buyer@x.com' AND source='care_taster'").fetchall()
    assert len(grants) == 1
    granted = datetime.fromisoformat(grants[0]["granted_at"].rstrip("Z"))
    expires = datetime.fromisoformat(grants[0]["expires_at"].rstrip("Z"))
    days = (expires - granted).days
    assert 29 <= days <= 31, f"expected ~30 day taster window, got {days} days"
    assert app_module._is_paid_member("buyer@x.com") is True


def test_program_return_care_taster_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", True, raising=False)
    _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com")
    c = app_module.app.test_client()
    c.get("/begin/checkout-return?session_id=cs_1")
    c.get("/begin/checkout-return?session_id=cs_1")

    with sqlite3.connect(db) as cx:
        n = cx.execute(
            "SELECT COUNT(*) FROM memberships WHERE email='buyer@x.com' "
            "AND source='care_taster'").fetchone()[0]
    assert n == 1, "replay must not double-grant the care taster window"


def test_program_return_flag_off_grants_nothing(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", False, raising=False)
    _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com")
    app_module.app.test_client().get("/begin/checkout-return?session_id=cs_1")

    with sqlite3.connect(db) as cx:
        n = cx.execute(
            "SELECT COUNT(*) FROM memberships WHERE email='buyer@x.com' "
            "AND source='care_taster'").fetchone()[0]
    assert n == 0


def _grant(cx, email, source, days=90):
    from datetime import timedelta
    import uuid
    now = datetime.utcnow()
    cx.execute(
        "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source, truly_vip_ref, notes) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), email, now.isoformat() + "Z", (now + timedelta(days=days)).isoformat() + "Z",
         source, source, "", ""))
    cx.commit()


def test_care_taster_outranks_lingering_trial_grant(monkeypatch, tmp_path):
    """Regression (sticky-trial, same shape as the prepay-term test): a biofield_trial
    grant alone reads as 'trial' (discount withheld). Once the SAME email also holds a
    care_taster grant (a real paid program purchase), it must no longer read as trial and
    must be treated as a paid member."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    with sqlite3.connect(db) as cx:
        _grant(cx, "d@x.com", "biofield_trial")
        _grant(cx, "d@x.com", "care_taster", days=30)
    assert app_module.membership_category("d@x.com") != "trial"
    assert app_module._is_paid_member("d@x.com") is True
