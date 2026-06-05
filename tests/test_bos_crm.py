import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE household_candidates (id INTEGER PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE pending_merges (id INTEGER PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE inbound_leads (id INTEGER PRIMARY KEY, status TEXT, "
               "last_outbound_at TEXT, email TEXT)")
    cx.commit()
    return cx


def test_crm_signal_green_when_empty():
    from dashboard import crm as C, signals as S
    assert C.crm_signal(_db(), None)["level"] == S.GREEN


def test_crm_signal_amber_on_candidates_only():
    from dashboard import crm as C, signals as S
    cx = _db()
    cx.execute("INSERT INTO household_candidates (status) VALUES ('pending')")
    cx.execute("INSERT INTO household_candidates (status) VALUES ('confirmed')")  # not counted
    cx.commit()
    sig = C.crm_signal(cx, None)
    assert sig["level"] == S.AMBER
    assert sig["count"] == 1
    assert "household" in sig["summary"].lower()


def test_crm_signal_red_on_leads_or_merges():
    from dashboard import crm as C, signals as S
    cx = _db()
    cx.execute("INSERT INTO inbound_leads (status, last_outbound_at, email) "
               "VALUES ('pending', '', 'a@b.com')")
    cx.commit()
    assert C.crm_signal(cx, None)["level"] == S.RED  # unreplied new lead is time-sensitive
    cx2 = _db()
    cx2.execute("INSERT INTO pending_merges (status) VALUES ('pending')")
    cx2.commit()
    assert C.crm_signal(cx2, None)["level"] == S.RED


def test_crm_signal_gray_when_tables_missing():
    from dashboard import crm as C, signals as S
    cx = sqlite3.connect(":memory:")  # no CRM tables
    assert C.crm_signal(cx, None)["level"] == S.GRAY


def test_crm_signal_registered():
    from dashboard import crm as C, signals as S
    assert S.SIGNAL_REGISTRY.get("crm") is not None
