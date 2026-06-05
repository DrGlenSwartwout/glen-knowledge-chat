import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_is_configured_reads_env(monkeypatch):
    from dashboard import easypost as EP
    monkeypatch.delenv("EASYPOST_API_KEY", raising=False)
    assert EP.is_configured() is False
    monkeypatch.setenv("EASYPOST_API_KEY", "ezk_test")
    assert EP.is_configured() is True


def test_build_shipment_shape():
    from dashboard import easypost as EP
    order = {"name": "Ann Buyer", "address": {"street": "1 Main St", "city": "Hilo",
             "state": "HI", "zip": "96720"}, "items": [{"qty": 2}]}
    s = EP.build_shipment(order, from_address={"name": "Remedy Match", "street": "x",
                          "city": "Hilo", "state": "HI", "zip": "96720"})
    assert s["to_address"]["name"] == "Ann Buyer"
    assert s["to_address"]["zip"] == "96720"
    assert s["from_address"]["zip"] == "96720"
    assert s["parcel"]["weight"] > 0  # ounces, derived from item count


def test_clicknship_url_constant():
    from dashboard import easypost as EP
    assert EP.CLICKNSHIP_URL.startswith("https://")


def test_create_label_handoff_when_unconfigured(monkeypatch):
    import sqlite3
    monkeypatch.delenv("EASYPOST_API_KEY", raising=False)
    from dashboard import orders as O, dispatch as D, events as E, rbac as R, actions as A
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx); O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="LBL-1", name="Ann")
    assert A.get_action("orders.create_label") is not None
    res = D.dispatch_action(cx, "orders.create_label", {"order_id": oid}, R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    msg = (res["result"] or {}).get("message", "")
    assert "click-n-ship" in msg.lower() or "cns.usps" in (res["result"] or {}).get("handoff", "").lower()


def test_send_tracking_records_shipment(monkeypatch):
    import sqlite3
    from dashboard import orders as O, dispatch as D, events as E, rbac as R, actions as A
    from dashboard import tracking as T
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx); O.init_orders_table(cx); T.init_tracking_schema(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="TR-1", name="Ann",
                         email="ann@x.com")
    O.set_order_tracking(cx, oid, "9400111899")
    # stub the gmail send so the test never hits the network
    import dashboard.orders as OM
    monkeypatch.setattr(OM, "_gmail_send_tracking", lambda to, subj, html: True, raising=False)
    res = D.dispatch_action(cx, "orders.send_tracking", {"order_id": oid}, R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    sh = cx.execute("SELECT status, resolved_email FROM shipments WHERE tracking_number='9400111899'").fetchone()
    assert sh is not None and sh["resolved_email"] == "ann@x.com"


def test_send_tracking_requires_tracking_number():
    import sqlite3
    from dashboard import orders as O, dispatch as D, events as E, rbac as R
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx); O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="TR-2", email="b@x.com")
    res = D.dispatch_action(cx, "orders.send_tracking", {"order_id": oid}, R.Actor(role=R.OWNER))
    assert res["status"] == "failed"


def test_send_tracking_does_not_double_email(monkeypatch):
    import sqlite3
    from dashboard import orders as O, dispatch as D, events as E, rbac as R
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx); O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="TR-3", name="Ann", email="a@x.com")
    O.set_order_tracking(cx, oid, "9400111777")
    calls = {"n": 0}
    monkeypatch.setattr(O, "_gmail_send_tracking",
                        lambda to, s, h: (calls.__setitem__("n", calls["n"] + 1), True)[1])
    r1 = D.dispatch_action(cx, "orders.send_tracking", {"order_id": oid}, R.Actor(role=R.OWNER))
    r2 = D.dispatch_action(cx, "orders.send_tracking", {"order_id": oid}, R.Actor(role=R.OWNER))
    assert r1["status"] == "done" and r2["status"] == "done"
    assert calls["n"] == 1  # the customer is emailed only once
    assert "already sent" in (r2["result"] or {}).get("message", "").lower()
