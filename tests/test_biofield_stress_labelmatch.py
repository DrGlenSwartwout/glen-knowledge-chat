import sqlite3
from dashboard.biofield_stress import add_voice_stress, init_stress_tables, list_stresses


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    add_voice_stress(cx, "a5", "Liver Congestion")
    return cx


def test_chain_row_head_balances_voice_stress(tmp_path):
    cx = _cx(tmp_path)
    rows = [{"head": "liver  congestion", "remedy": "Hepato Tonic"}]
    res = list_stresses(cx, "a5", rows)
    assert [s["label"] for s in res["balanced"]] == ["Liver Congestion"]
    assert res["balanced"][0]["balanced_by"] == "Hepato Tonic"
    assert res["active"] == []


def test_row_without_remedy_does_not_balance(tmp_path):
    cx = _cx(tmp_path)
    res = list_stresses(cx, "a5", [{"head": "liver congestion", "remedy": ""}])
    assert [s["label"] for s in res["active"]] == ["Liver Congestion"]
    assert res["balanced"] == []


def test_removing_row_reactivates(tmp_path):
    cx = _cx(tmp_path)
    assert list_stresses(cx, "a5", [{"head": "liver congestion", "remedy": "X"}])["active"] == []
    assert [s["label"] for s in list_stresses(cx, "a5", [])["active"]] == ["Liver Congestion"]


def test_string_list_still_works_backcompat(tmp_path):
    cx = _cx(tmp_path)
    # plain remedy-name strings: no head -> no label match -> voice stress stays active
    res = list_stresses(cx, "a5", ["Some Remedy"])
    assert [s["label"] for s in res["active"]] == ["Liver Congestion"]
