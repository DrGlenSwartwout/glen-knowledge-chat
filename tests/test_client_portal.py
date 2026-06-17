# tests/test_client_portal.py
import sqlite3
from dashboard import client_portal as cp


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); cp.init_client_portal_table(cx); return cx


def test_status_defaults_confirmed_for_legacy(tmp_path):
    cx = _cx(tmp_path)
    cp.upsert_portal(cx, "a@x.com", "A", {"layers": [{"n": 1, "title": "t", "remedy": "R"}]})
    assert cp.get_biofield_status(cx, "a@x.com") == "confirmed"   # no field -> confirmed


def test_set_and_get_status(tmp_path):
    cx = _cx(tmp_path)
    cp.upsert_portal(cx, "a@x.com", "A", {"biofield_status": "ai_draft", "layers": []})
    assert cp.get_biofield_status(cx, "a@x.com") == "ai_draft"
    assert cp.set_biofield_status(cx, "a@x.com", "interested") is True
    assert cp.get_biofield_status(cx, "a@x.com") == "interested"
    assert cp.set_biofield_status(cx, "nobody@x.com", "interested") is False
