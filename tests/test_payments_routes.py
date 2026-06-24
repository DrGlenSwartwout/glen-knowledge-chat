"""Route tests for the console payments page + API (/console/payments,
/api/payments). The list test seeds a REAL orders DB and goes through the route
end-to-end (no mock of the ledger) so a wrong query shape would fail here."""
import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _key(appmod):
    return appmod.CONSOLE_SECRET or ""


def _seed(appmod):
    from dashboard import orders as O, stripe_alerts as SA
    cx = sqlite3.connect(appmod.LOG_DB)
    O.init_orders_table(cx)
    SA.init_stripe_alerts_table(cx)
    # one-time card checkout (PI in the column)
    O.upsert_order(cx, source="funnel", external_ref="cs_1", email="a@x.com",
                   name="A", items=[{"name": "X", "qty": 1, "desc": "X"}],
                   total_cents=7000, address={}, channel="retail")
    oid = cx.execute("SELECT id FROM orders WHERE external_ref='cs_1'").fetchone()[0]
    O.set_order_stripe_pi(cx, oid, "pi_one")
    O.set_order_payment(cx, oid, method="card", amount_cents=7000)
    # subscription renewal (PI only in external_ref)
    O.upsert_order(cx, source="subscription", external_ref="pi_sub", email="b@x.com",
                   name="B", items=[{"name": "Y", "qty": 1, "desc": "Y"}],
                   total_cents=9900, address={}, channel="retail")
    SA.record_failure(cx, "subscription charge", "card declined",
                      now="2026-06-24T00:00:00+00:00", notify=False)
    cx.commit()
    cx.close()


def test_page_served(client):
    c, _ = client
    r = c.get("/console/payments")
    assert r.status_code == 200


def test_api_gated_without_key(client):
    c, appmod = client
    r = c.get("/api/payments")
    if appmod.CONSOLE_SECRET:
        assert r.status_code == 401
    else:
        assert r.status_code == 200


def test_api_lists_payments_failures_and_summary(client):
    c, appmod = client
    _seed(appmod)
    r = c.get("/api/payments?key=" + _key(appmod))
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    refs = {row["external_ref"] for row in body["data"]}
    assert refs == {"cs_1", "pi_sub"}          # both Stripe captures present
    sub = next(x for x in body["data"] if x["external_ref"] == "pi_sub")
    assert sub["stripe_payment_intent"] == "pi_sub"   # normalized from external_ref
    assert sub["amount_cents"] == 9900                # fell back to total_cents
    assert body["summary"]["count"] == 2
    assert body["summary"]["total_cents"] == 16900
    assert body["failures"][0]["context"] == "subscription charge"


def test_api_source_filter(client):
    c, appmod = client
    _seed(appmod)
    r = c.get("/api/payments?source=subscription&key=" + _key(appmod))
    body = r.get_json()
    assert [x["external_ref"] for x in body["data"]] == ["pi_sub"]
