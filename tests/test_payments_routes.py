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
    # /console/payments now redirects to the consolidated Money console.
    r = c.get("/console/payments")
    assert r.status_code == 302
    assert "/console/money" in r.headers.get("Location", "")


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


# --- backfill trigger endpoint ----------------------------------------------

def _seed_grants(appmod, sessions, monkeypatch):
    """Create biofield_trial_grants + orders table, and mock Stripe get_session."""
    from dashboard import orders as O, stripe_pay
    cx = sqlite3.connect(appmod.LOG_DB)
    O.init_orders_table(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_trial_grants "
               "(session_id TEXT PRIMARY KEY, email TEXT, granted_at TEXT)")
    for sid in sessions:
        cx.execute("INSERT OR IGNORE INTO biofield_trial_grants VALUES (?,?,?)",
                   (sid, sid + "@x.com", "2026-06-01T00:00:00Z"))
    cx.commit(); cx.close()
    monkeypatch.setattr(stripe_pay, "get_session", lambda s: sessions[s])


def test_backfill_endpoint_gated(client):
    c, appmod = client
    r = c.post("/api/console/backfill-trial-orders")
    if appmod.CONSOLE_SECRET:
        assert r.status_code == 401


def test_backfill_endpoint_dry_run_writes_nothing(client, monkeypatch):
    c, appmod = client
    _seed_grants(appmod, {"cs_a": {"payment_intent": "pi_a", "amount_total": 100}}, monkeypatch)
    r = c.post("/api/console/backfill-trial-orders?dry_run=1&key=" + _key(appmod))
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["dry_run"] is True
    assert body["result"]["created"] == 1
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    from dashboard import payments as P
    assert P.list_payments(cx) == []  # dry run wrote nothing
    cx.close()


def test_backfill_endpoint_apply_creates_orders(client, monkeypatch):
    c, appmod = client
    _seed_grants(appmod, {"cs_a": {"payment_intent": "pi_a", "amount_total": 100}}, monkeypatch)
    r = c.post("/api/console/backfill-trial-orders?key=" + _key(appmod))
    body = r.get_json()
    assert body["ok"] is True and body["dry_run"] is False
    assert body["result"]["created"] == 1
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    from dashboard import payments as P
    assert {p["external_ref"] for p in P.list_payments(cx)} == {"pi_a"}
    cx.close()


def test_backfill_marks_trial_order_paid_and_keeps_done(client, monkeypatch):
    """A backfilled $1-trial order is a captured charge -> pay_status 'paid', and it
    stays 'done' (digital, nothing to ship) rather than being pulled into 'new'."""
    c, appmod = client
    _seed_grants(appmod, {"cs_a": {"payment_intent": "pi_a", "amount_total": 100}}, monkeypatch)
    r = c.post("/api/console/backfill-trial-orders?key=" + _key(appmod))
    assert r.get_json()["result"]["created"] == 1
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    row = cx.execute("SELECT status, pay_status, paid_cents FROM orders "
                     "WHERE external_ref='pi_a'").fetchone()
    cx.close()
    assert row["status"] == "done"
    assert row["pay_status"] == "paid"
    assert row["paid_cents"] == 100


def test_backfill_reconciles_preexisting_unpaid_trial_order(client, monkeypatch):
    """The Done-board 'Unpaid' bug: a trial order recorded before it was marked paid
    is flipped to paid on a (re-)run, without leaving 'done'."""
    c, appmod = client
    from dashboard import orders as O
    cx = sqlite3.connect(appmod.LOG_DB)
    O.init_orders_table(cx)
    # a stale unpaid trial order (how the 6 prod cards looked)
    O.upsert_order(cx, source="biofield_trial", external_ref="pi_old", email="o@x.com",
                   items=[], total_cents=100, address={}, channel="retail", status="done")
    cx.commit(); cx.close()
    _seed_grants(appmod, {}, monkeypatch)   # no new grants; reconcile pass only
    r = c.post("/api/console/backfill-trial-orders?key=" + _key(appmod))
    assert r.get_json()["result"]["reconciled"] == 1
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    row = cx.execute("SELECT status, pay_status FROM orders WHERE external_ref='pi_old'").fetchone()
    cx.close()
    assert row["status"] == "done" and row["pay_status"] == "paid"
