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


####################################################################
# Task 4: seed repertoire on membership conversion + append on member order
####################################################################

def _seed_order(db, *, source, external_ref, email, slugs, status="done",
                 days_ago=1):
    """Insert an orders row directly (bypassing _ingest_order) so history-seed
    tests control created_at precisely."""
    import json
    from datetime import datetime, timedelta, timezone
    from dashboard.orders import init_orders_table
    created = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    items = [{"slug": s, "qty": 1} for s in slugs]
    with sqlite3.connect(db) as cx:
        init_orders_table(cx)
        cx.execute(
            "INSERT INTO orders (created_at, source, external_ref, channel, email, "
            "items_json, total_cents, status) VALUES (?,?,?,?,?,?,?,?)",
            (created, source, external_ref, "retail", email,
             json.dumps(items), 1000, status))
        cx.commit()


def _mock_paid_prepay_session(app_module, monkeypatch, email="a@b.com", tier_key="6mo"):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "prepay_term", "email": email, "tier_key": tier_key},
                   "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})


def test_prepay_term_won_claim_seeds_repertoire_from_windowed_history(monkeypatch, tmp_path):
    """Fulfilling a 6-month prepay term (180d window) for an email with 2
    non-cancelled orders (slugs a,b) in the last 180d seeds {a,b}; a cancelled
    order's slug is excluded."""
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    monkeypatch.setattr(A, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(A, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(A, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    # Keep ledger + email side-effects out of this test — only seeding is under test.
    monkeypatch.setattr(A, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(A, "_send_inquiry_email", lambda *a, **k: None, raising=False)

    email = "member@x.com"
    _seed_order(db, source="funnel", external_ref="o1", email=email, slugs=["a"], days_ago=10)
    _seed_order(db, source="funnel", external_ref="o2", email=email, slugs=["b"], days_ago=20)
    # Cancelled order — its slug must be excluded from seeding.
    _seed_order(db, source="funnel", external_ref="o3", email=email, slugs=["c"],
                status="cancelled", days_ago=5)
    # Outside the 180d window — must be excluded too.
    _seed_order(db, source="funnel", external_ref="o4", email=email, slugs=["d"], days_ago=200)

    _mock_paid_prepay_session(A, monkeypatch, email=email, tier_key="6mo")
    res = A._fulfill_prepay_term("cs_1")
    assert res.get("ok") is True

    with sqlite3.connect(db) as cx:
        slugs = A.repertoire.repertoire_slugs(cx, email)
    assert slugs == {"a", "b"}


def test_prepay_term_seed_flag_off_no_write(monkeypatch, tmp_path):
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", False)
    monkeypatch.setattr(A, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(A, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(A, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    monkeypatch.setattr(A, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(A, "_send_inquiry_email", lambda *a, **k: None, raising=False)

    email = "member2@x.com"
    _seed_order(db, source="funnel", external_ref="o1", email=email, slugs=["a"], days_ago=10)

    _mock_paid_prepay_session(A, monkeypatch, email=email, tier_key="6mo")
    res = A._fulfill_prepay_term("cs_1")
    assert res.get("ok") is True

    with sqlite3.connect(db) as cx:
        slugs = A.repertoire.repertoire_slugs(cx, email)
    assert slugs == set()


def test_ingest_order_appends_slugs_for_paid_member(monkeypatch, tmp_path):
    """A member's completed purchase adds its item slugs to their repertoire so
    the NEXT reorder is discounted."""
    A = _load_app()
    db = str(tmp_path / "chat_log.db")
    from dashboard.orders import init_orders_table
    from dashboard.client_portal import init_client_portal_table
    from dashboard.email_suppression import init_table as init_suppression
    with sqlite3.connect(db) as cx:
        init_orders_table(cx)
        init_client_portal_table(cx)
        init_suppression(cx)
        A.init_membership_tables(cx)
        A.repertoire.init_repertoire_table(cx)
        cx.commit()
    monkeypatch.setattr(A, "LOG_DB", db)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    monkeypatch.setattr(A, "_send_full_report_email", lambda *a, **k: ("mock", None))

    email = "member3@x.com"
    _seed_active_membership(db, email)

    A._ingest_order(source="test", external_ref="o1", email=email, name="N",
                     items=[{"slug": "neuro-magnesium", "qty": 1},
                            {"slug": "terrain-restore", "qty": 2}],
                     total_cents=5000, status="done")

    with sqlite3.connect(db) as cx:
        slugs = A.repertoire.repertoire_slugs(cx, email)
    assert slugs == {"neuro-magnesium", "terrain-restore"}


def test_ingest_order_does_not_append_for_non_member(monkeypatch, tmp_path):
    A = _load_app()
    db = str(tmp_path / "chat_log.db")
    from dashboard.orders import init_orders_table
    from dashboard.client_portal import init_client_portal_table
    from dashboard.email_suppression import init_table as init_suppression
    with sqlite3.connect(db) as cx:
        init_orders_table(cx)
        init_client_portal_table(cx)
        init_suppression(cx)
        A.init_membership_tables(cx)
        A.repertoire.init_repertoire_table(cx)
        cx.commit()
    monkeypatch.setattr(A, "LOG_DB", db)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", True)
    monkeypatch.setattr(A, "_send_full_report_email", lambda *a, **k: ("mock", None))

    email = "nonmember2@x.com"  # no active membership seeded

    A._ingest_order(source="test", external_ref="o1", email=email, name="N",
                     items=[{"slug": "neuro-magnesium", "qty": 1}],
                     total_cents=5000, status="done")

    with sqlite3.connect(db) as cx:
        slugs = A.repertoire.repertoire_slugs(cx, email)
    assert slugs == set()


def test_ingest_order_flag_off_no_append(monkeypatch, tmp_path):
    A = _load_app()
    db = str(tmp_path / "chat_log.db")
    from dashboard.orders import init_orders_table
    from dashboard.client_portal import init_client_portal_table
    from dashboard.email_suppression import init_table as init_suppression
    with sqlite3.connect(db) as cx:
        init_orders_table(cx)
        init_client_portal_table(cx)
        init_suppression(cx)
        A.init_membership_tables(cx)
        A.repertoire.init_repertoire_table(cx)
        cx.commit()
    monkeypatch.setattr(A, "LOG_DB", db)
    monkeypatch.setattr(A, "REPERTOIRE_ENABLED", False)
    monkeypatch.setattr(A, "_send_full_report_email", lambda *a, **k: ("mock", None))

    email = "member4@x.com"
    _seed_active_membership(db, email)

    A._ingest_order(source="test", external_ref="o1", email=email, name="N",
                     items=[{"slug": "neuro-magnesium", "qty": 1}],
                     total_cents=5000, status="done")

    with sqlite3.connect(db) as cx:
        slugs = A.repertoire.repertoire_slugs(cx, email)
    assert slugs == set()


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
