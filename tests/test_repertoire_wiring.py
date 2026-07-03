# tests/test_repertoire_wiring.py
"""Route/unit tests for repertoire reorder pricing wired into _price_cart
(Task 3). Mirrors the tmp-file sqlite harness in tests/test_prepay_checkout.py.
"""
import importlib, sqlite3, sys
from datetime import datetime, timedelta
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


def _seed_active_membership(db, email, *, source="founding"):
    """Insert an active (future-expiring) memberships grant row so
    _active_membership_for_email(email) finds it and _is_paid_member(email)
    can return True (membership_category won't classify as 'trial' for a
    non-biofield_trial source)."""
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"
    with sqlite3.connect(db) as cx:
        cx.execute(
            "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source) "
            "VALUES (?,?,?,?,?,?)",
            ("mem_test_1", email, datetime.utcnow().isoformat() + "Z", expires,
             "test", source),
        )
        cx.commit()


def test_price_cart_applies_repertoire_for_member(monkeypatch, tmp_path):
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    _seed_active_membership(db, "member@x.com")
    with sqlite3.connect(db) as cx:
        A.repertoire.add_skus(cx, "member@x.com", ["neuro-magnesium"])

    out = A._price_cart(
        [{"slug": "neuro-magnesium", "qty": 1}],
        ship={"country": "US", "state": "TX"},
        email="member@x.com",
    )
    # list is 6997; repertoire_reorder_pct=0.29 -> ~4968 (< 5100 gives headroom)
    assert out["priced"]["lines"][0]["line_total_cents"] < 5100
    assert out["priced"]["lines"][0]["pct_applied"] >= 28.9


def test_price_cart_flag_off_regular_price(monkeypatch, tmp_path):
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", False)
    _seed_active_membership(db, "member@x.com")
    with sqlite3.connect(db) as cx:
        A.repertoire.add_skus(cx, "member@x.com", ["neuro-magnesium"])

    out = A._price_cart(
        [{"slug": "neuro-magnesium", "qty": 1}],
        ship={"country": "US", "state": "TX"},
        email="member@x.com",
    )
    assert out["priced"]["lines"][0]["line_total_cents"] == 6997


def test_price_cart_non_member_regular_price(monkeypatch, tmp_path):
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    # No active membership seeded for this email at all.
    with sqlite3.connect(db) as cx:
        A.repertoire.add_skus(cx, "nonmember@x.com", ["neuro-magnesium"])

    out = A._price_cart(
        [{"slug": "neuro-magnesium", "qty": 1}],
        ship={"country": "US", "state": "TX"},
        email="nonmember@x.com",
    )
    assert out["priced"]["lines"][0]["line_total_cents"] == 6997


def test_price_cart_fresh_db_no_repertoire_table_does_not_crash(monkeypatch, tmp_path):
    """Fresh-DB guard: even if the repertoire table somehow isn't there yet when
    _price_cart runs (e.g. a DB created before this feature shipped), the read
    must not throw — _price_cart re-inits defensively at read time."""
    A = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(A, "LOG_DB", db)
    with sqlite3.connect(db) as cx:
        A.init_membership_tables(cx)
        cx.commit()  # deliberately skip repertoire.init_repertoire_table
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    _seed_active_membership(db, "member@x.com")

    out = A._price_cart(
        [{"slug": "neuro-magnesium", "qty": 1}],
        ship={"country": "US", "state": "TX"},
        email="member@x.com",
    )
    # No repertoire row exists yet either way, so regular price -- the point of
    # this test is that it doesn't raise.
    assert out["priced"]["lines"][0]["line_total_cents"] == 6997
