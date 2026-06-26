import sqlite3
from dashboard.biofield_stress import (
    add_voice_stress, init_stress_tables, seed_from_scan, suggest_minimal_remedies)


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    cx.executescript("""
        CREATE TABLE fmp_snap_client_active_main_stress(id_pk INTEGER, main_stress TEXT);
        CREATE TABLE fmp_snap_client_causal_chain(id_pk INTEGER, id_fk_active_stress INTEGER);
        CREATE TABLE fmp_snap_client_remedy(id_fk_causal_chain INTEGER, remedy TEXT);
    """)
    cx.execute("INSERT INTO fmp_snap_client_active_main_stress VALUES(1,'Adrenal Fatigue')")
    cx.execute("INSERT INTO fmp_snap_client_causal_chain VALUES(10,1)")
    cx.execute("INSERT INTO fmp_snap_client_remedy VALUES(10,'Adaptogen Blend')")
    cx.commit()
    return cx


def test_nonscan_stress_in_setcover(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Adrenal Fatigue")
    res = suggest_minimal_remedies(cx, "a5", [])
    assert {"remedy": "adaptogen blend", "covers": ["Adrenal Fatigue"]} in res["picks"]
    assert res["uncovered"] == []


def test_nonscan_no_history_uncovered(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Mystery Stress")          # no FMP history
    res = suggest_minimal_remedies(cx, "a5", [])
    assert res["picks"] == [] and res["uncovered"] == ["Mystery Stress"]


def test_scan_and_nonscan_covered_together(tmp_path):
    cx = _cx(tmp_path)
    seed_from_scan(cx, "a5", [{"code": "ED1", "name": "Membrane"}],
                   {"neuro magnesium": {"ED1"}})           # scan stress, required
    add_voice_stress(cx, "a5", "Adrenal Fatigue")          # non-scan, historical
    res = suggest_minimal_remedies(cx, "a5", [])
    by = {p["remedy"]: p["covers"] for p in res["picks"]}
    assert by.get("neuro magnesium") == ["Membrane"]
    assert by.get("adaptogen blend") == ["Adrenal Fatigue"]
    assert res["uncovered"] == []
