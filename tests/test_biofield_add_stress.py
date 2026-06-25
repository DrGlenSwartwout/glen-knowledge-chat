import sqlite3
from dashboard.biofield_stress import add_stress, add_voice_stress, init_stress_tables, seed_from_scan


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    return cx


def test_add_stress_tag_source(tmp_path):
    cx = _cx(tmp_path)
    assert add_stress(cx, "a5", "Adrenal Fatigue", source="tag") is True
    row = cx.execute("SELECT source, balance, code FROM biofield_auth_stress WHERE test_id=5").fetchone()
    assert row[0] == "tag" and row[1] == "required" and row[2] == "adrenal fatigue"


def test_add_stress_merges_cross_source(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Liver Congestion")              # voice
    assert add_stress(cx, "a5", "  liver congestion ", source="tag") is False   # merge
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == 1


def test_add_voice_stress_still_voice(tmp_path):
    cx = _cx(tmp_path)
    assert add_voice_stress(cx, "a5", "Brain Fog") is True
    assert cx.execute("SELECT source FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == "voice"
