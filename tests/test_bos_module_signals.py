import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_marketing_signal(monkeypatch):
    from dashboard import module_signals as M, signals as S
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE inbound_leads (id INTEGER PRIMARY KEY, source TEXT, status TEXT, "
               "last_outbound_at TEXT, email TEXT)")
    cx.commit()
    assert M.marketing_signal(cx, None)["level"] == S.GREEN
    cx.execute("INSERT INTO inbound_leads (source, status, last_outbound_at, email) "
               "VALUES ('scoreapp', 'pending', '', 'a@b.com')")
    cx.execute("INSERT INTO inbound_leads (source, status, last_outbound_at, email) "
               "VALUES ('groovekart', 'pending', '', 'c@d.com')")  # not scoreapp -> excluded
    cx.commit()
    sig = M.marketing_signal(cx, None)
    assert sig["level"] == S.AMBER and sig["count"] == 1


def test_products_signal(monkeypatch, tmp_path):
    from dashboard import module_signals as M, signals as S
    (tmp_path / "products.json").write_text(json.dumps(
        {"products": {"a": {"name": "A"}, "b": {"name": "B", "info_only": True}}}))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    sig = M.products_signal(None, None)
    assert sig["level"] == S.GREEN and sig["count"] == 1  # info_only excluded


def test_content_signal(monkeypatch, tmp_path):
    from dashboard import module_signals as M, signals as S
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "atlas-pending.json").write_text(json.dumps({"concepts": [{"id": "1"}, {"id": "2"}]}))
    sig = M.content_signal(None, None)
    assert sig["level"] == S.AMBER and sig["count"] == 2
    (tmp_path / "atlas-pending.json").write_text(json.dumps({"concepts": []}))
    assert M.content_signal(None, None)["level"] == S.GREEN


def test_comms_signal():
    from dashboard import module_signals as M, signals as S
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE calendar_events (id INTEGER PRIMARY KEY, status TEXT, start TEXT)")
    cx.commit()
    assert M.comms_signal(cx, None)["level"] == S.GREEN
    soon = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    cx.execute("INSERT INTO calendar_events (status, start) VALUES ('visible', ?)", (soon,))
    cx.execute("INSERT INTO calendar_events (status, start) VALUES ('visible', ?)", (far,))
    cx.commit()
    sig = M.comms_signal(cx, None)
    assert sig["level"] == S.AMBER and sig["count"] == 1  # only the soon one


def test_b2b_signal():
    from dashboard import module_signals as M, signals as S
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, source TEXT, status TEXT)")
    cx.commit()
    assert M.b2b_signal(cx, None)["level"] == S.GREEN and M.b2b_signal(cx, None)["count"] == 0
    cx.execute("INSERT INTO orders (source, status) VALUES ('wholesale', 'new')")
    cx.execute("INSERT INTO orders (source, status) VALUES ('funnel', 'new')")  # not b2b
    cx.commit()
    sig = M.b2b_signal(cx, None)
    assert sig["level"] == S.GREEN and sig["count"] == 1


def test_all_defensive_gray_on_missing():
    from dashboard import module_signals as M, signals as S
    cx = sqlite3.connect(":memory:")  # no tables
    assert M.marketing_signal(cx, None)["level"] == S.GRAY
    assert M.comms_signal(cx, None)["level"] == S.GRAY
    assert M.b2b_signal(cx, None)["level"] == S.GRAY


def test_all_registered():
    from dashboard import module_signals as M  # noqa: F401
    from dashboard import signals as S
    for m in ("marketing", "products", "content", "comms", "b2b"):
        assert S.SIGNAL_REGISTRY.get(m) is not None, m
