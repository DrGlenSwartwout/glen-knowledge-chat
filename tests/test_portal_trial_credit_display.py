"""Portal trial-credit display (PR4): /api/portal/<token> exposes
membership_category + trial_credit_cents so the portal can show a trial buyer the
upgrade credit they've accrued so far.
"""
import json
import sqlite3
import pytest

EXPECTED = 0  # trial-credit machinery retired; always 0


@pytest.fixture
def env(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    appmod._init_membership_tables()
    appmod.app.config["TESTING"] = True
    return appmod


def _seed_portal(appmod, email, name="Client"):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, {"greeting": "Hi", "reorder_items": []})
    cx.close()
    return token


def _seed_trial_with_orders(appmod, email):
    from dashboard import subscriptions as subs
    from dashboard import orders as orders_mod
    from datetime import date
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx); subs.migrate_add_membership_columns(cx)
    orders_mod.init_orders_table(cx)
    subs.create_membership(cx, email=email, stripe_customer_id="c",
                           stripe_payment_method_id="p", amount_cents=9900,
                           next_charge_date="2026-07-01")  # order_count 0 -> trial
    today = date.today().isoformat()
    cx.execute("INSERT INTO orders (created_at, source, external_ref, email, items_json, "
               "total_cents, status) VALUES (?, 'biofield_trial', 'bt1', ?, ?, 100, 'new')",
               (today + "T00:00:00+00:00", email, json.dumps([{"name": "Biofield Analysis", "qty": 1}])))
    cx.execute("INSERT INTO orders (created_at, source, external_ref, email, items_json, "
               "total_cents, status) VALUES (?, 'reorder', 're1', ?, ?, 0, 'new')",
               (today + "T01:00:00+00:00", email, json.dumps([{"name": "Bone Builder", "qty": 3}])))
    cx.commit(); cx.close()


def test_trial_buyer_sees_category_and_credit(env):
    email = "trial-portal@example.com"
    _seed_trial_with_orders(env, email)
    token = _seed_portal(env, email)
    r = env.app.test_client().get(f"/api/portal/{token}")
    assert r.status_code == 200, r.data
    d = r.get_json()
    assert d["membership_category"] == "trial"
    assert d["trial_credit_cents"] == EXPECTED


def test_non_member_portal_has_no_credit(env):
    email = "plain@example.com"
    token = _seed_portal(env, email)
    r = env.app.test_client().get(f"/api/portal/{token}")
    assert r.status_code == 200, r.data
    d = r.get_json()
    assert d["membership_category"] == "none"
    assert d["trial_credit_cents"] == 0
