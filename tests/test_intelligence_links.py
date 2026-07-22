import json
from pathlib import Path

from dashboard import intelligence as intel
from dashboard import db


def test_write_then_read_links_roundtrips(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    reg = {"r1": {"type": "person", "display": "Jane",
                  "url": "/console/crm?email=jane%40x.com"}}
    intel.write_links("money-cash", reg, db_path=dbp)
    assert intel.read_links("money-cash", db_path=dbp) == reg


def test_read_links_missing_returns_empty(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    assert intel.read_links("money-cash", db_path=dbp) == {}


def test_read_briefing_includes_links(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    intel.write_briefing("money-cash", "# Finance\n\n[Jane](ref:r1)", db_path=dbp)
    intel.write_links("money-cash", {"r1": {"type": "person", "display": "Jane",
                                            "url": "/console/crm?email=jane%40x.com"}},
                       db_path=dbp)
    data = intel.read_briefing("money-cash", db_path=dbp)
    assert data["links"]["r1"]["url"] == "/console/crm?email=jane%40x.com"


def test_read_briefing_empty_has_no_links(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    data = intel.read_briefing("money-cash", db_path=dbp)
    assert data["empty"] is True
    assert "links" not in data


def test_read_links_corrupt_returns_empty(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    with db.connect(dbp, timeout=10) as cx:
        intel._ensure_table(cx)
        cx.execute(
            "INSERT INTO intelligence_briefings (slug, links_json, updated_at) "
            "VALUES (?,?,?)",
            ("money-cash", "not valid json{{", "2026-01-01T00:00:00+00:00"),
        )
        cx.commit()
    assert intel.read_links("money-cash", db_path=dbp) == {}
