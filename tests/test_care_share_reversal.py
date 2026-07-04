# tests/test_care_share_reversal.py
"""Owner console care-share reversal action (Task 7).

There is no Stripe refund webhook for Continuous Care memberships, so when a
membership charge is manually refunded the owner console must explicitly
reverse the previously-posted care-share credit. POST
/api/console/care-share/reverse {sub_id, order_count} reads the subscription
(for its attributed_practitioner_id) and calls
wallet.reverse_care_share(pid, event_ref="care_share:<sub_id>:<order_count>") —
the exact event_ref format the earn side used. The reversal amount is NOT
recomputed at the doctor's current cert rate (modules_for_practitioner /
share_cents) — reverse_care_share reads the ACTUAL posted credit for that
event_ref from the wallet ledger, so a manual refund reverses exactly what was
credited even if the doctor's modules_completed changed since the charge.

Console-gated the same way every other /api/console/* route is: _console_key_ok()
(X-Console-Key header / ?key= / CONSOLE_SECRET), mirroring tests/test_reorder_velocity_api.py.
subscriptions table + care_share/wallet are patched the way test_care_share_enroll.py
patches them, so no real Stripe/Supabase call is made.
"""
import importlib, sqlite3, sys
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


@pytest.fixture
def client(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "test-secret")
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        subscriptions.migrate_add_term_cap_column(cx)
        subscriptions.migrate_add_attribution_column(cx)
        subscriptions.migrate_add_consent_column(cx)
        cx.commit()
    app_module.app.config["TESTING"] = True
    app_module._db = db  # stash for tests
    return app_module.app.test_client()


def _seed_sub(app_module_db, *, attributed_pid="prac-42", amount_cents=9900):
    """Insert a membership subscription row directly and return its id."""
    from dashboard import subscriptions as subs
    with sqlite3.connect(app_module_db) as cx:
        sub_id = subs.create_membership(
            cx, email="pat@x.com", stripe_customer_id="cus_1",
            stripe_payment_method_id="pm_1", amount_cents=amount_cents,
            next_charge_date="2026-08-01",
            attributed_practitioner_id=attributed_pid,
        )
        cx.commit()
    return sub_id


def _app_db():
    import app as appmod
    return appmod._db


def test_valid_reversal_calls_wallet_with_expected_args(monkeypatch, client):
    import app as appmod
    db = _app_db()
    sub_id = _seed_sub(db, attributed_pid="prac-42", amount_cents=9900)

    from dashboard import care_share, wallet
    # The endpoint must NOT consult modules_for_practitioner/share_cents at
    # all any more — poison it so the test fails loudly if the recompute
    # ever creeps back in.
    def _poison(pid):
        raise AssertionError("endpoint must not recompute the current cert rate")
    monkeypatch.setattr(care_share, "modules_for_practitioner", _poison)

    rec = {}

    def fake_reverse(pid, *, event_ref):
        rec["pid"] = pid
        rec["event_ref"] = event_ref
        return 4123  # the amount actually reversed, per the ledger — arbitrary here

    monkeypatch.setattr(wallet, "reverse_care_share", fake_reverse)

    r = client.post(
        "/api/console/care-share/reverse",
        json={"sub_id": sub_id, "order_count": 2},
        headers={"X-Console-Key": "test-secret"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["reversed_cents"] == 4123  # exactly what reverse_care_share reported

    assert rec["pid"] == "prac-42"
    assert rec["event_ref"] == f"care_share:{sub_id}:2"


def test_missing_console_key_unauthorized(client):
    r = client.post(
        "/api/console/care-share/reverse",
        json={"sub_id": 1, "order_count": 1},
    )
    assert r.status_code in (401, 403)


def test_invalid_console_key_unauthorized(client):
    r = client.post(
        "/api/console/care-share/reverse",
        json={"sub_id": 1, "order_count": 1},
        headers={"X-Console-Key": "wrong-secret"},
    )
    assert r.status_code in (401, 403)


def test_unattributed_sub_returns_404_no_reversal(monkeypatch, client):
    db = _app_db()
    sub_id = _seed_sub(db, attributed_pid=None, amount_cents=9900)

    from dashboard import wallet
    rec = []
    monkeypatch.setattr(wallet, "reverse_care_share",
                         lambda *a, **k: rec.append((a, k)))

    r = client.post(
        "/api/console/care-share/reverse",
        json={"sub_id": sub_id, "order_count": 1},
        headers={"X-Console-Key": "test-secret"},
    )
    assert r.status_code == 404
    body = r.get_json()
    assert body["ok"] is False
    assert "error" in body
    assert rec == []


def test_missing_sub_returns_404_no_reversal(monkeypatch, client):
    from dashboard import wallet
    rec = []
    monkeypatch.setattr(wallet, "reverse_care_share",
                         lambda *a, **k: rec.append((a, k)))

    r = client.post(
        "/api/console/care-share/reverse",
        json={"sub_id": 999999, "order_count": 1},
        headers={"X-Console-Key": "test-secret"},
    )
    assert r.status_code == 404
    assert rec == []
