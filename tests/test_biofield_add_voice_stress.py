import sqlite3
from dashboard.biofield_stress import add_voice_stress, init_stress_tables, seed_from_scan


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    return cx


def test_inserts_voice_stress_required(tmp_path):
    cx = _cx(tmp_path)
    assert add_voice_stress(cx, "a5", "Liver Congestion") is True
    row = cx.execute("SELECT source, balance, code, label FROM biofield_auth_stress "
                     "WHERE test_id=5").fetchone()
    assert row[0] == "voice" and row[1] == "required" and row[2] == "liver congestion"
    assert row[3] == "Liver Congestion"


def test_normalized_duplicate_merges(tmp_path):
    cx = _cx(tmp_path)
    assert add_voice_stress(cx, "a5", "Liver Congestion") is True
    assert add_voice_stress(cx, "a5", "  liver   congestion!! ") is False   # normalized dup
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == 1


def test_two_distinct_voice_stresses_both_insert(tmp_path):
    cx = _cx(tmp_path)
    assert add_voice_stress(cx, "a5", "Liver Congestion") is True
    assert add_voice_stress(cx, "a5", "Adrenal Fatigue") is True   # both code='' would collide pre-fix
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == 2


def test_merges_against_existing_scan_stress(tmp_path):
    cx = _cx(tmp_path)
    seed_from_scan(cx, "a5", [{"code": "ED1", "name": "Liver congestion"}], {})
    assert add_voice_stress(cx, "a5", "liver  congestion") is False   # matches scan label
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == 1
