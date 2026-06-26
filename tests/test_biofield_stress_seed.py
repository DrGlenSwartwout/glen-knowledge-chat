import sqlite3
from dashboard.biofield_stress import init_stress_tables, seed_from_scan

_FIND = [{"code": "ED1", "name": "Membrane"}, {"code": "ES3", "name": "Lymph"},
         {"code": "MR2", "name": "Calm Mind"}]
_COV = {"neuro magnesium": {"ED1", "ES3"}}   # MR2 not covered -> optional


def test_seed_assigns_required_optional_and_coverage(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    res = seed_from_scan(cx, "a5", _FIND, _COV)
    assert res["stresses"] == 3 and res["required"] == 2
    rows = {r[0]: r[1] for r in cx.execute(
        "SELECT code, balance FROM biofield_auth_stress WHERE test_id=5").fetchall()}
    assert rows == {"ED1": "required", "ES3": "required", "MR2": "optional"}
    cov = cx.execute("SELECT remedy, code FROM biofield_auth_remedy_coverage WHERE test_id=5 ORDER BY code").fetchall()
    assert ("neuro magnesium", "ED1") in cov and ("neuro magnesium", "ES3") in cov


def test_reseed_preserves_manual_balanced_and_rebuilds_coverage(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    seed_from_scan(cx, "a5", _FIND, _COV)
    cx.execute("UPDATE biofield_auth_stress SET manual_balanced=1 WHERE code='MR2' AND test_id=5")
    cx.commit()
    seed_from_scan(cx, "a5", _FIND, {"neuro magnesium": {"ED1"}})   # ES3 now optional, coverage shrank
    rows = {r[0]: (r[1], r[2]) for r in cx.execute(
        "SELECT code, balance, manual_balanced FROM biofield_auth_stress WHERE test_id=5").fetchall()}
    assert rows["MR2"][1] == 1                      # manual flag preserved
    assert rows["ES3"][0] == "optional"            # reclassified on re-seed
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_remedy_coverage WHERE test_id=5").fetchone()[0] == 1
