import sqlite3, json
from dashboard import condition_programs as cp

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cp.init_table(cx); return cx

def test_upsert_and_get_roundtrips_modifiers():
    cx = _cx()
    mods = [{"when": "drusen", "action": "add", "source": "diagnosis-implied",
             "client_default": True, "items": [{"slug": "lipid-zyme", "name": "Lipid Zyme"}]}]
    cp.upsert(cx, "dry-amd", "Dry AMD", False,
              [{"slug": "wholomega", "name": "WholOmega"}], mods)
    got = cp.get(cx, "dry-amd")
    assert got["modifiers"] == mods
    assert got["items"] == [{"slug": "wholomega", "name": "WholOmega"}]

def test_get_defaults_modifiers_to_empty_list_when_absent():
    cx = _cx()
    cp.upsert(cx, "dry-eye", "Dry Eye", False, [{"slug": "moisturize", "name": "Moisturize"}])
    assert cp.get(cx, "dry-eye")["modifiers"] == []

def test_migration_adds_column_to_preexisting_table():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    # simulate an OLD table with no modifiers_json column
    cx.execute("""CREATE TABLE condition_programs (condition_key TEXT PRIMARY KEY,
        label TEXT, consult_recommended INTEGER NOT NULL DEFAULT 0,
        items_json TEXT NOT NULL DEFAULT '[]', updated_at TEXT)""")
    cx.execute("INSERT INTO condition_programs VALUES ('x','X',0,'[]','t')")
    cp.init_table(cx)  # must ALTER-add modifiers_json without error
    assert cp.get(cx, "x")["modifiers"] == []


def test_resolver_applies_diagnosis_implied_add():
    prog = {"items": [{"slug": "wholomega", "name": "WholOmega"}],
            "modifiers": [{"when": "drusen", "action": "add", "source": "diagnosis-implied",
                "client_default": True, "items": [{"slug": "lipid-zyme", "name": "Lipid Zyme"}]}]}
    out = cp.resolve_program_items(prog)
    assert [i["slug"] for i in out] == ["wholomega", "lipid-zyme"]

def test_resolver_client_reported_remove_only_when_fact_true():
    prog = {"items": [{"slug": "macular-wellness-lutein", "name": "L"},
                      {"slug": "macular-wellness-crocin", "name": "C"}],
            "modifiers": [{"when": "on_areds2", "action": "remove", "source": "client-reported",
                "client_default": False, "items": [{"slug": "macular-wellness-lutein"}]}]}
    assert [i["slug"] for i in cp.resolve_program_items(prog)] == \
        ["macular-wellness-lutein", "macular-wellness-crocin"]
    assert [i["slug"] for i in cp.resolve_program_items(prog, client_facts={"on_areds2": True})] == \
        ["macular-wellness-crocin"]

def test_resolver_suppresses_clinician_measured():
    prog = {"items": [{"slug": "glucose-tolerance", "name": "G"}],
            "modifiers": [{"when": "proliferative", "action": "add", "source": "clinician-measured",
                "client_default": False, "items": [{"slug": "angiogenx", "name": "AngiogenX"}]}]}
    assert [i["slug"] for i in cp.resolve_program_items(prog)] == ["glucose-tolerance"]

def test_resolver_dedupes_adds_against_base():
    prog = {"items": [{"slug": "lipid-zyme", "name": "Lipid Zyme"}],
            "modifiers": [{"when": "drusen", "action": "add", "source": "diagnosis-implied",
                "client_default": True, "items": [{"slug": "lipid-zyme", "name": "Lipid Zyme"}]}]}
    assert [i["slug"] for i in cp.resolve_program_items(prog)] == ["lipid-zyme"]
