# tests/test_history_seed.py
"""Task 4: union purchase_history (FMP/GK backfill) into the repertoire seeder
alongside orders, at all 3 existing seed_from_history call sites (prepay
won-claim, continuous-care won-claim, continuous-care duplicate-member).
Mirrors the sqlite harness + fulfiller mocks in tests/test_repertoire_wiring.py.
"""
import importlib, sqlite3, sys
from datetime import datetime, timedelta, timezone
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
    with sqlite3.connect(db) as cx:
        app_module.init_membership_tables(cx)
        app_module.repertoire.init_repertoire_table(cx)
        cx.commit()
    return db


def _seed_purchase_history(db, *, email, slug, source="fmp", source_ref="ref1",
                            days_ago=1):
    """Insert a purchase_history row directly so history-seed tests control
    purchased_at precisely (a proven remedy known only from the FMP/GK
    backfill, not the live orders board)."""
    from dashboard import purchase_history as _ph
    purchased = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    with sqlite3.connect(db) as cx:
        _ph.init_purchase_history_table(cx)
        cx.execute(
            "INSERT OR IGNORE INTO purchase_history "
            "(email, slug, purchased_at, source, source_ref) VALUES (?,?,?,?,?)",
            (email.strip().lower(), slug.strip().lower(), purchased, source, source_ref))
        cx.commit()


def _mock_paid_prepay_session(app_module, monkeypatch, email="a@b.com", tier_key="6mo"):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "prepay_term", "email": email, "tier_key": tier_key},
                   "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})


def _mock_paid_continuous_care_session(app_module, monkeypatch, email="a@b.com",
                                        term_months=6):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "continuous_care_monthly", "email": email,
                                 "term_months": str(term_months)},
                   "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"id": "pi_1", "customer": "cus_1", "payment_method": "pm_1",
                    "status": "succeeded"})


def test_prepay_term_seeds_repertoire_from_purchase_history(monkeypatch, tmp_path):
    """A 6-month prepay term (180d window) for an email whose ONLY record of
    buying SKU x is a purchase_history row (source=fmp) dated within the window
    gets x in their repertoire; a row older than the window does not seed."""
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    monkeypatch.setattr(A, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(A, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(A, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    monkeypatch.setattr(A, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(A, "_send_inquiry_email", lambda *a, **k: None, raising=False)

    email = "histmember@x.com"
    _seed_purchase_history(db, email=email, slug="x", source_ref="fmp1", days_ago=30)
    # Outside the 180d window — must be excluded.
    _seed_purchase_history(db, email=email, slug="y", source_ref="fmp2", days_ago=400)

    _mock_paid_prepay_session(A, monkeypatch, email=email, tier_key="6mo")
    res = A._fulfill_prepay_term("cs_1")
    assert res.get("ok") is True

    with sqlite3.connect(db) as cx:
        slugs = A.repertoire.repertoire_slugs(cx, email)
    assert slugs == {"x"}


def test_continuous_care_won_claim_seeds_repertoire_from_purchase_history(monkeypatch, tmp_path):
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    monkeypatch.setattr(A, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(A, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    monkeypatch.setattr(A, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(A, "_send_subscription_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(A, "_mint_membership_cancel_url", lambda *a, **k: None, raising=False)

    email = "cchistmember@x.com"
    _seed_purchase_history(db, email=email, slug="x", source_ref="fmp1", days_ago=30)
    _seed_purchase_history(db, email=email, slug="y", source_ref="fmp2", days_ago=400)

    _mock_paid_continuous_care_session(A, monkeypatch, email=email, term_months=6)
    res = A._fulfill_continuous_care_monthly("cs_1")
    assert res.get("ok") is True

    with sqlite3.connect(db) as cx:
        slugs = A.repertoire.repertoire_slugs(cx, email)
    assert slugs == {"x"}


def test_continuous_care_duplicate_member_seeds_repertoire_from_purchase_history(monkeypatch, tmp_path):
    """The duplicate_member early-return path (existing active membership from
    a non-seeding origin) must ALSO union purchase_history, same as the
    won-claim branch."""
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    monkeypatch.setattr(A, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(A, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    monkeypatch.setattr(A, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(A, "_send_subscription_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(A, "_mint_membership_cancel_url", lambda *a, **k: None, raising=False)

    email = "dupchistmember@x.com"

    from dashboard import subscriptions as _subs
    with sqlite3.connect(db) as cx:
        _subs.init_subscriptions_table(cx)
        _subs.migrate_add_membership_columns(cx)
        _subs.migrate_add_term_cap_column(cx)
        _subs.migrate_add_attribution_column(cx)
        _subs.create_membership(
            cx, email=email, stripe_customer_id="cus_other",
            stripe_payment_method_id="pm_other", amount_cents=9900,
            next_charge_date="2026-08-01", cadence_months=1,
            term_charges_total=None, initial_order_count=1)

    _seed_purchase_history(db, email=email, slug="x", source_ref="fmp1", days_ago=30)
    _seed_purchase_history(db, email=email, slug="y", source_ref="fmp2", days_ago=400)

    _mock_paid_continuous_care_session(A, monkeypatch, email=email, term_months=6)
    res = A._fulfill_continuous_care_monthly("cs_1")
    assert res.get("ok") is True
    assert res.get("duplicate_member") is True

    with sqlite3.connect(db) as cx:
        slugs = A.repertoire.repertoire_slugs(cx, email)
    assert slugs == {"x"}


def test_prepay_term_seed_flag_off_no_purchase_history_write(monkeypatch, tmp_path):
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", False)
    monkeypatch.setattr(A, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(A, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(A, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    monkeypatch.setattr(A, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(A, "_send_inquiry_email", lambda *a, **k: None, raising=False)

    email = "flagoffhist@x.com"
    _seed_purchase_history(db, email=email, slug="x", source_ref="fmp1", days_ago=30)

    _mock_paid_prepay_session(A, monkeypatch, email=email, tier_key="6mo")
    res = A._fulfill_prepay_term("cs_1")
    assert res.get("ok") is True

    with sqlite3.connect(db) as cx:
        slugs = A.repertoire.repertoire_slugs(cx, email)
    assert slugs == set()
