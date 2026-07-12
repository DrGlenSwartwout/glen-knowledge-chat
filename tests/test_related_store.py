import json, os
from dashboard import related_store as rs

def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    rs.save_manual("iop-syntropy", ["immune-modulation", "wholomega"])
    assert rs.load_manual("iop-syntropy") == ["immune-modulation", "wholomega"]
    assert rs.load_manual()["iop-syntropy"] == ["immune-modulation", "wholomega"]

def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert rs.load_manual("nope") == []
    assert rs.load_manual() == {}
