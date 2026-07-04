"""Console ops for dispensary attribution: pay-mix (read-only ratio proxy) and the
in-web backfill trigger (disk is mounted only on the web container)."""
import importlib
import sqlite3

import dashboard.practitioner_portal as pp_mod
from dashboard import referrals as rf

KEY = "test-console-secret"


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    monkeypatch.setenv("CONSOLE_SECRET", KEY)
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config["TESTING"] = True
    return appmod


def _seed_orders(appmod):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        from dashboard import orders as o
        o.init_orders_table(cx)
        # 3 dispensary card orders (have a stripe_payment_intent), 2 alt-pay (NULL)
        rows = [("dispensary", "D1", 7000, "pi_1"), ("dispensary", "D2", 5000, "pi_2"),
                ("dispensary", "D3", 6000, "pi_3"), ("dispensary", "D4", 4000, None),
                ("dispensary", "D5", 3000, None),
                ("wholesale", "W1", 9999, "pi_x")]  # non-dispensary → excluded
        for src, ref, cents, pi in rows:
            cx.execute("INSERT INTO orders(created_at,source,external_ref,total_cents,"
                       "stripe_payment_intent) VALUES('t',?,?,?,?)", (src, ref, cents, pi))
        cx.commit()


def test_pay_mix_splits_card_vs_altpay(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    _seed_orders(appmod)
    c = appmod.app.test_client()
    r = c.get("/api/console/dispensary-pay-mix", headers={"X-Console-Key": KEY})
    assert r.status_code == 200
    d = r.get_json()
    assert d["card"]["orders"] == 3 and d["card"]["total_cents"] == 18000
    assert d["alt_pay"]["orders"] == 2 and d["alt_pay"]["total_cents"] == 7000
    assert d["total_orders"] == 5           # wholesale row excluded
    assert d["alt_pay_order_share"] == 0.4  # 2/5


def test_pay_mix_requires_console_key(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    assert c.get("/api/console/dispensary-pay-mix").status_code == 401


def test_backfill_endpoint_dry_run_writes_nothing(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("CREATE TABLE dispensary_orders(invoice_id TEXT PRIMARY KEY, "
                   "practitioner_id TEXT, customer_email TEXT)")
        cx.execute("INSERT INTO dispensary_orders VALUES('D1','p1','a@x.com')")
        cx.execute("INSERT INTO dispensary_orders VALUES('D2','p1','b@x.com')")
        cx.commit()
    c = appmod.app.test_client()
    r = c.post("/api/console/backfill-dispensary-referrals?dry_run=1",
               headers={"X-Console-Key": KEY})
    assert r.status_code == 200
    d = r.get_json()
    assert d["dry_run"] is True and d["written"] == 2
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rf.redemption_by_order_ref(cx, "D1") is None   # nothing written


def test_backfill_endpoint_real_run_writes_rows(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("CREATE TABLE dispensary_orders(invoice_id TEXT PRIMARY KEY, "
                   "practitioner_id TEXT, customer_email TEXT)")
        cx.execute("INSERT INTO dispensary_orders VALUES('D1','p1','a@x.com')")
        cx.commit()
    c = appmod.app.test_client()
    r = c.post("/api/console/backfill-dispensary-referrals?dry_run=0",
               headers={"X-Console-Key": KEY})
    assert r.status_code == 200 and r.get_json()["written"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rf.owner_of_referee(cx, "a@x.com") == "doc@x.com"
