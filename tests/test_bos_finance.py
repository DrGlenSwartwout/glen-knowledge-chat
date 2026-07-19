import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

NOW = datetime(2026, 6, 5, tzinfo=timezone.utc)


def test_aging_filters_zero_balance_and_computes_overdue():
    from dashboard import finance as F
    invs = [
        {"Id": "1", "DocNumber": "1001", "Balance": "70.00", "TotalAmt": "70.00",
         "DueDate": "2026-06-01", "CustomerRef": {"name": "Ann"}},   # 4 days overdue
        {"Id": "2", "DocNumber": "1002", "Balance": "0", "TotalAmt": "50.00",
         "DueDate": "2026-05-01", "CustomerRef": {"name": "Paid"}},  # zero balance -> excluded
        {"Id": "3", "DocNumber": "1003", "Balance": "40.00", "TotalAmt": "40.00",
         "DueDate": "2026-06-20", "CustomerRef": {"name": "Future"}},  # not due yet
    ]
    aged = F.aging(invs, now=NOW)
    assert [a["id"] for a in aged] == ["1", "3"]  # zero-balance dropped, sorted most-overdue first
    assert aged[0]["days_overdue"] == 4
    assert aged[0]["customer"] == "Ann"
    assert aged[1]["days_overdue"] < 0  # future


def test_aging_tags_source_qbo():
    from dashboard import finance as F
    aged = F.aging([{"Id": "1", "DocNumber": "1", "Balance": "10", "TotalAmt": "10",
                     "DueDate": "2026-06-01", "CustomerRef": {"name": "A"}}], now=NOW)
    assert aged[0]["source"] == "qbo"


def test_inhouse_aging_shapes_unpaid_orders_and_dedupes_qbo():
    from dashboard import finance as F
    orders = [
        # unpaid, 20 days old (net-14 -> 6 days overdue), in-house only
        {"id": 11, "created_at": "2026-05-16", "external_ref": "manual-11",
         "email": "a@x.com", "name": "Al", "total_cents": 9900, "paid_cents": 0,
         "pay_status": "unpaid", "status": "confirmed"},
        # already paid -> excluded
        {"id": 12, "created_at": "2026-05-01", "external_ref": "manual-12",
         "total_cents": 5000, "pay_status": "paid", "status": "shipped"},
        # cancelled -> excluded
        {"id": 13, "created_at": "2026-05-01", "external_ref": "manual-13",
         "total_cents": 5000, "pay_status": "unpaid", "status": "cancelled"},
        # fully covered by paid_cents (zero balance) -> excluded
        {"id": 14, "created_at": "2026-05-01", "external_ref": "manual-14",
         "total_cents": 5000, "paid_cents": 5000, "pay_status": "unpaid", "status": "new"},
        # external_ref IS a QBO invoice already on the QBO list -> deduped out
        {"id": 15, "created_at": "2026-05-01", "external_ref": "QBO-9",
         "total_cents": 4000, "pay_status": "unpaid", "status": "new"},
        # recent unpaid, still within net terms -> included but not overdue
        {"id": 16, "created_at": "2026-06-03", "external_ref": "manual-16",
         "total_cents": 3000, "pay_status": "unpaid", "status": "new"},
    ]
    rows = F.inhouse_aging(orders, qbo_ids={"QBO-9"}, now=NOW)
    assert [r["id"] for r in rows] == [11, 16]  # paid/cancelled/zero/deduped dropped, most-overdue first
    assert rows[0]["source"] == "inhouse"
    assert rows[0]["order_id"] == 11
    assert rows[0]["balance"] == 99.0
    assert rows[0]["days_overdue"] == 6      # 20 days old, net-14 terms
    assert rows[0]["customer"] == "Al"
    assert rows[1]["days_overdue"] < 0       # id 16 still within terms


def test_summarize_totals():
    from dashboard import finance as F
    aged = [{"balance": 70.0, "days_overdue": 4}, {"balance": 40.0, "days_overdue": -5}]
    s = F.summarize(aged, cash_total=1234.5)
    assert s["open_count"] == 2
    assert s["open_total"] == 110.0
    assert s["overdue_count"] == 1
    assert s["overdue_total"] == 70.0
    assert s["cash_total"] == 1234.5


def test_money_signal_from_levels():
    from dashboard import finance as F
    from dashboard import signals as S
    assert F.money_signal_from({"open_count": 0, "overdue_count": 0})["level"] == S.GREEN
    assert F.money_signal_from({"open_count": 3, "overdue_count": 0, "open_total": 200})["level"] == S.AMBER
    red = F.money_signal_from({"open_count": 3, "overdue_count": 2, "overdue_total": 150})
    assert red["level"] == S.RED and red["count"] == 2
    # cash floor breach also goes red
    low = F.money_signal_from({"open_count": 0, "overdue_count": 0, "cash_total": 50}, cash_floor=500)
    assert low["level"] == S.RED


def test_money_signal_registered_and_defensive(monkeypatch):
    import sqlite3
    from dashboard import finance as F, signals as S
    # force the QBO-backed summary to blow up -> signal must return GRAY, not raise
    monkeypatch.setattr(F, "finance_summary", lambda: (_ for _ in ()).throw(RuntimeError("qbo down")))
    cx = sqlite3.connect(":memory:")
    cell = F.money_signal(cx, None)
    assert cell["level"] == S.GRAY
    assert S.SIGNAL_REGISTRY.get("money") is not None


def test_void_invoice_action_registered():
    from dashboard import finance as F, actions as A
    a = A.get_action("finance.void_invoice")
    assert a is not None
    assert a.module == "money"
    assert a.permission == ("owner", "ops")  # not va
    assert a.risk_tier == A.IRREVERSIBLE  # a void is permanent
    assert a.confirm_summary is not None  # the confirm dialog is not the generic fallback


def test_void_confirm_summary_rich_and_fallback():
    from dashboard import finance as F, actions as A
    fn = A.get_action("finance.void_invoice").confirm_summary
    # Rich: the panel passes _doc/_who/_amount for a readable prompt.
    rich = fn({"invoice_id": 24434, "_doc": "1016",
               "_who": "backdoc.molina@gmail.com", "_amount": 225})
    assert "invoice 1016" in rich
    assert "backdoc.molina@gmail.com" in rich
    assert "$225.00" in rich
    assert "cannot be undone" in rich
    # Fallback: a bare call (e.g. Justus) still yields a sane, id-based prompt.
    bare = fn({"invoice_id": 24434})
    assert "invoice 24434" in bare
    assert "cannot be undone" in bare


def test_refund_action_registered():
    from dashboard import finance as F, actions as A  # noqa: F401
    a = A.get_action("finance.refund_order")
    assert a is not None
    assert a.module == "money"
    assert a.risk_tier == A.MONEY_SEND
    # owner+ops confirm; va queues (per RBAC policy); all three are in permission
    assert "owner" in a.permission and "ops" in a.permission and "va" in a.permission
    assert a.confirm_summary is not None


def test_refund_owner_needs_confirmation_no_qbo_call(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB
    called = {"refund": 0}
    monkeypatch.setattr(QB, "create_refund_receipt",
                        lambda *a, **k: called.__setitem__("refund", called["refund"] + 1) or {"Id": "1"})
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 80},
                            R.Actor(role=R.OWNER))
    assert res["status"] == "needs_confirmation"
    assert "80" in res["summary"]
    assert called["refund"] == 0  # nothing happened without confirmation


def test_refund_va_queues_no_qbo_call(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB
    called = {"refund": 0}
    monkeypatch.setattr(QB, "create_refund_receipt",
                        lambda *a, **k: called.__setitem__("refund", called["refund"] + 1) or {"Id": "1"})
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 80}, R.Actor(role=R.VA))
    assert res["status"] == "queued"
    assert called["refund"] == 0
    assert E.get_event(cx, res["event_id"])["status"] == "pending_approval"


def test_refund_executes_when_confirmed(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB
    monkeypatch.setattr(QB, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "C7"}, "DocNumber": "1009"})
    captured = {}
    def _fake_refund(customer_id, amount, **k):
        captured.update({"customer_id": customer_id, "amount": amount})
        return {"Id": "RR1", "DocNumber": "RR-1"}
    monkeypatch.setattr(QB, "create_refund_receipt", _fake_refund)
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 80, "reason": "duplicate"},
                            R.Actor(role=R.OWNER), confirmed=True)
    assert res["status"] == "done"
    assert captured == {"customer_id": "C7", "amount": 80.0}
    assert "80" in res["result"]["message"]


def test_refund_issues_stripe_card_refund_first(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB, stripe_pay as SP
    order = {}
    monkeypatch.setattr(QB, "get_invoice", lambda iid: {"CustomerRef": {"value": "C1"}, "DocNumber": "1"})
    calls = []
    monkeypatch.setattr(SP, "refund", lambda pi, amount_cents=None: calls.append(("stripe", pi, amount_cents)) or {"id": "re_1", "status": "succeeded"})
    monkeypatch.setattr(QB, "create_refund_receipt", lambda cid, amt, **k: calls.append(("qbo", cid, amt)) or {"Id": "RR1", "DocNumber": "RR-1"})
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 40, "stripe_payment_intent": "pi_9"},
                            R.Actor(role=R.OWNER), confirmed=True)
    assert res["status"] == "done"
    # stripe refund runs BEFORE the qbo record, with cents
    assert calls[0] == ("stripe", "pi_9", 4000)
    assert calls[1][0] == "qbo"
    assert "card" in res["result"]["message"].lower()


def test_refund_qbo_only_without_payment_intent(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB, stripe_pay as SP
    monkeypatch.setattr(QB, "get_invoice", lambda iid: {"CustomerRef": {"value": "C1"}})
    monkeypatch.setattr(QB, "create_refund_receipt", lambda cid, amt, **k: {"Id": "RR2", "DocNumber": "RR-2"})
    def _no_stripe(*a, **k): raise AssertionError("stripe.refund must not be called")
    monkeypatch.setattr(SP, "refund", _no_stripe)
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    from dashboard import orders as O; O.init_orders_table(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV-NONE", "amount": 40},
                            R.Actor(role=R.OWNER), confirmed=True)
    assert res["status"] == "done"
    assert "quickbooks" in res["result"]["message"].lower()


def test_refund_card_failure_blocks_qbo(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB, stripe_pay as SP
    monkeypatch.setattr(QB, "get_invoice", lambda iid: {"CustomerRef": {"value": "C1"}})
    monkeypatch.setattr(SP, "refund", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("card declined")))
    qbo_called = {"n": 0}
    monkeypatch.setattr(QB, "create_refund_receipt", lambda *a, **k: qbo_called.__setitem__("n", 1) or {"Id": "RR"})
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 40, "stripe_payment_intent": "pi_x"},
                            R.Actor(role=R.OWNER), confirmed=True)
    assert res["status"] == "failed"
    assert qbo_called["n"] == 0  # card failed -> nothing booked
