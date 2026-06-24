"""Authoring helpers over the FMP snapshot: remedy catalog (+phase/system), dosing
autofill, stress vocabulary, and 'what you've used before' history suggestions."""
import sqlite3
from dashboard.biofield_authoring import (
    remedy_catalog, remedy_dosing, stress_vocab, stress_suggestions)


def _seed(db):
    cx = sqlite3.connect(db)
    cx.executescript("""
      CREATE TABLE fmp_snap_products(id_pk TEXT,product_name TEXT,dosage TEXT,dosage_freq TEXT,dosage_timing TEXT);
      CREATE TABLE fmp_snap_products_phases(id_fk_product TEXT,text TEXT);
      CREATE TABLE fmp_snap_products_systems(id_fk_product TEXT,text TEXT);
      CREATE TABLE fmp_snap_client_active_main_stress(id_pk TEXT,main_stress TEXT,layer TEXT);
      CREATE TABLE fmp_snap_client_causal_chain(id_pk TEXT,id_fk_active_stress TEXT);
      CREATE TABLE fmp_snap_client_remedy(id_fk_causal_chain TEXT,remedy TEXT);
    """)
    cx.execute("INSERT INTO fmp_snap_products VALUES('7','Sterol Max','3 caps','daily','with food')")
    cx.execute("INSERT INTO fmp_snap_products VALUES('8','Stone Solvent','1 capsule','daily','with food')")
    cx.execute("INSERT INTO fmp_snap_products_phases VALUES('7','Cleanse')")
    cx.execute("INSERT INTO fmp_snap_products_systems VALUES('7','Flow')")
    cx.executemany("INSERT INTO fmp_snap_client_active_main_stress VALUES(?,?,?)",
                   [("100", "Acid", "2"), ("101", "Acid", "1")])
    cx.executemany("INSERT INTO fmp_snap_client_causal_chain VALUES(?,?)",
                   [("200", "100"), ("201", "101")])
    cx.executemany("INSERT INTO fmp_snap_client_remedy VALUES(?,?)",
                   [("200", "Sterol Max"), ("201", "Sterol Max")])
    cx.commit()
    return cx


def test_catalog_search_with_phase_system(tmp_path):
    cx = _seed(str(tmp_path / "chat_log.db"))
    res = remedy_catalog(cx, "sterol")
    assert len(res) == 1 and res[0]["name"] == "Sterol Max"
    assert res[0]["phase"] == "Cleanse" and res[0]["system"] == "Flow"
    assert res[0]["timing"] == "with food"


def test_dosing_autofill(tmp_path):
    cx = _seed(str(tmp_path / "chat_log.db"))
    assert remedy_dosing(cx, "Sterol Max") == {
        "dosage": "3 caps", "frequency": "daily", "timing": "with food"}


def test_stress_vocab(tmp_path):
    cx = _seed(str(tmp_path / "chat_log.db"))
    assert stress_vocab(cx, "ac") == ["Acid"]


def test_stress_suggestions_counts_history(tmp_path):
    cx = _seed(str(tmp_path / "chat_log.db"))
    s = stress_suggestions(cx, "Acid")
    assert s[0]["remedy"] == "Sterol Max" and s[0]["count"] == 2


def test_helpers_graceful_without_snapshot(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    assert remedy_catalog(cx, "x") == [] and stress_vocab(cx) == []
    assert stress_suggestions(cx, "x") == []
    assert remedy_dosing(cx, "x") == {"dosage": "", "frequency": "", "timing": ""}
