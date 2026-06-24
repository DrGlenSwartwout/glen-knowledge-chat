"""Causal Chain Report from the FMP snapshot: layer-ordered, remedies joined from
client_remedy, layer taken from the linked active-stress factor."""
import sqlite3
import pytest
from dashboard.biofield_report import causal_chain_report, list_tests


def _seed(db):
    cx = sqlite3.connect(db)
    cx.executescript("""
      CREATE TABLE fmp_snap_clients (id_pk TEXT, email TEXT, name_first TEXT, name_last TEXT);
      CREATE TABLE fmp_snap_client_biofield_test (id_pk TEXT, id_fk_client TEXT, date_test TEXT);
      CREATE TABLE fmp_snap_client_active_main_stress (id_pk TEXT, layer TEXT);
      CREATE TABLE fmp_snap_client_causal_chain
        (id_pk TEXT, id_fk_test TEXT, id_fk_active_stress TEXT, head_chain TEXT, most_affected TEXT);
      CREATE TABLE fmp_snap_client_remedy
        (id_fk_causal_chain TEXT, remedy TEXT, dosage TEXT, frequency TEXT, timing TEXT);
    """)
    cx.execute("INSERT INTO fmp_snap_clients VALUES ('5','lz@x.com','Lewis','Zardo')")
    cx.execute("INSERT INTO fmp_snap_client_biofield_test VALUES ('10','5','2026-06-01')")
    cx.executemany("INSERT INTO fmp_snap_client_active_main_stress VALUES (?,?)",
                   [("100", "2"), ("101", "1")])
    cx.executemany("INSERT INTO fmp_snap_client_causal_chain VALUES (?,?,?,?,?)",
                   [("200", "10", "100", "Acid", "Liver"),
                    ("201", "10", "101", "Night", "Night")])
    cx.executemany("INSERT INTO fmp_snap_client_remedy VALUES (?,?,?,?,?)",
                   [("200", "Sterol Max", "3 caps", "daily", "with food"),
                    ("201", "TMG", "1 scoop", "daily", "at night")])
    cx.commit()
    return cx


def test_report_orders_by_stress_layer_and_joins_remedy(tmp_path):
    db = str(tmp_path / "chat_log.db")
    cx = _seed(db)
    rep = causal_chain_report(cx, "10")
    assert rep["client"]["name"] == "Lewis Zardo"
    assert rep["client"]["email"] == "lz@x.com"
    # layer 1 (Night) before layer 2 (Acid)
    assert [(l["layer"], l["head"], l["remedy"]) for l in rep["layers"]] == [
        (1, "Night", "TMG"),
        (2, "Acid", "Sterol Max"),
    ]


def test_report_includes_schedule(tmp_path):
    db = str(tmp_path / "chat_log.db")
    cx = _seed(db)
    rep = causal_chain_report(cx, "10")
    slots = {e["name"]: e["slots"] for e in rep["schedule"]["entries"]}
    assert slots["TMG"] == ["Bedtime"]
    assert slots["Sterol Max"] == ["Breakfast"]


def test_list_tests_returns_client_and_count(tmp_path):
    db = str(tmp_path / "chat_log.db")
    cx = _seed(db)
    tests = list_tests(cx)
    assert len(tests) == 1
    assert tests[0]["test_id"] == "10"
    assert tests[0]["name"] == "Lewis Zardo"
    assert tests[0]["layer_count"] == 2
