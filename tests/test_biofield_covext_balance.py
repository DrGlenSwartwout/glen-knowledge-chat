import sqlite3
from dashboard.biofield_stress import (
    add_stress, add_voice_stress, historical_remedies, init_stress_tables, list_stresses)


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    # FMP snapshot tables that stress_suggestions joins on
    cx.executescript("""
        CREATE TABLE fmp_snap_client_active_main_stress(id_pk INTEGER, main_stress TEXT);
        CREATE TABLE fmp_snap_client_causal_chain(id_pk INTEGER, id_fk_active_stress INTEGER);
        CREATE TABLE fmp_snap_client_remedy(id_fk_causal_chain INTEGER, remedy TEXT);
    """)
    # history: "Adrenal Fatigue" was balanced with "Adaptogen Blend"
    cx.execute("INSERT INTO fmp_snap_client_active_main_stress VALUES(1,'Adrenal Fatigue')")
    cx.execute("INSERT INTO fmp_snap_client_causal_chain VALUES(10,1)")
    cx.execute("INSERT INTO fmp_snap_client_remedy VALUES(10,'Adaptogen Blend')")
    cx.commit()
    return cx


def test_historical_remedies_lowercased(tmp_path):
    cx = _cx(tmp_path)
    assert historical_remedies(cx, "Adrenal Fatigue") == {"adaptogen blend"}
    assert historical_remedies(cx, "Unknown Thing") == set()


def test_nonscan_stress_balanced_by_history(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Adrenal Fatigue")                       # source=voice, no E4L code
    # a chain row whose remedy is historically used for the stress (head unrelated, so NOT a B2 label-match)
    res = list_stresses(cx, "a5", [{"head": "unrelated layer", "remedy": "Adaptogen Blend"}])
    bal = {s["label"]: s["balanced_by"] for s in res["balanced"]}
    assert bal == {"Adrenal Fatigue": "adaptogen blend"}
    assert res["active"] == []
    # removing the remedy reactivates it
    assert [s["label"] for s in list_stresses(cx, "a5", [])["active"]] == ["Adrenal Fatigue"]


def test_unrelated_remedy_does_not_balance(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Adrenal Fatigue")
    res = list_stresses(cx, "a5", [{"head": "x", "remedy": "Something Else"}])
    assert [s["label"] for s in res["active"]] == ["Adrenal Fatigue"]
