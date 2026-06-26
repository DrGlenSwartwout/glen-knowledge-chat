import json
from pathlib import Path

from dashboard import intelligence as intel


def _use_tmp(monkeypatch, tmp_path):
    d = tmp_path / "intelligence"
    d.mkdir()
    monkeypatch.setattr(intel, "DATA_DIR", d)
    return d


def test_write_then_read_links_roundtrips(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    reg = {"r1": {"type": "person", "display": "Jane",
                  "url": "/console/crm?email=jane%40x.com"}}
    intel.write_links("money-cash", reg)
    assert intel.read_links("money-cash") == reg


def test_read_links_missing_returns_empty(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    assert intel.read_links("money-cash") == {}


def test_read_briefing_includes_links(monkeypatch, tmp_path):
    d = _use_tmp(monkeypatch, tmp_path)
    (d / "money-cash.md").write_text("# Finance\n\n[Jane](ref:r1)")
    intel.write_links("money-cash", {"r1": {"type": "person", "display": "Jane",
                                            "url": "/console/crm?email=jane%40x.com"}})
    data = intel.read_briefing("money-cash")
    assert data["links"]["r1"]["url"] == "/console/crm?email=jane%40x.com"


def test_read_briefing_empty_has_no_links(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    data = intel.read_briefing("money-cash")
    assert data["empty"] is True
    assert "links" not in data
