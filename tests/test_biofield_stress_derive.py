import sqlite3
from dashboard.biofield_stress import (
    init_stress_tables, list_stresses, seed_from_scan, set_manual_balanced)

_FIND = [{"code": "ED1", "name": "Membrane"}, {"code": "ES3", "name": "Lymph"},
         {"code": "MR2", "name": "Calm Mind"}]
_COV = {"neuro magnesium": {"ED1", "ES3"}}


def _seeded(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    seed_from_scan(cx, "a5", _FIND, _COV)
    return cx


def test_chain_remedy_balances_its_codes(tmp_path):
    cx = _seeded(tmp_path)
    res = list_stresses(cx, "a5", ["Neuro Magnesium"])    # case-insensitive
    bal = {s["code"]: s["balanced_by"] for s in res["balanced"]}
    assert set(bal) == {"ED1", "ES3"} and bal["ED1"] == "neuro magnesium"
    assert {s["code"] for s in res["active"]} == {"MR2"}


def test_off_scan_remedy_clears_nothing(tmp_path):
    cx = _seeded(tmp_path)
    res = list_stresses(cx, "a5", ["Some Tincture"])
    assert res["balanced"] == [] and len(res["active"]) == 3


def test_manual_overrides_regardless_of_chain(tmp_path):
    cx = _seeded(tmp_path)
    sid = cx.execute("SELECT id FROM biofield_auth_stress WHERE code='MR2' AND test_id=5").fetchone()[0]
    set_manual_balanced(cx, "a5", sid, True)
    res = list_stresses(cx, "a5", [])
    bal = {s["code"]: s["balanced_by"] for s in res["balanced"]}
    assert bal == {"MR2": "manual"}
    set_manual_balanced(cx, "a5", sid, False)
    assert list_stresses(cx, "a5", [])["balanced"] == []
