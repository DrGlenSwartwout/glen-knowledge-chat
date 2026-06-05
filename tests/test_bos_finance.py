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
