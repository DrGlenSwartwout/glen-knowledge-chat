"""A partially-spoken dose must not let the model invent the rest.

From Debra Herndon's real session (2026-07-09): Glen said only "Fiber Cleanse, one a
day". The catalog-dose fill was all-or-nothing —

    if not (dosage or frequency or timing):   # any one field set -> no fill at all
        dosage, frequency, timing = remedy_dosing(...)

— so a spoken frequency suppressed the catalog lookup entirely and the LLM's own
guesses for the other fields were stored. Fiber Cleanse got timing "with food"; the
catalog says "with extra water, away from beneficial oils" — the opposite instruction
for a fiber product.

Two guards:
  1. merge_dosing() fills each EMPTY field independently from the catalog.
  2. the interpret prompt forbids inferring a dose field that was not spoken.
"""
import json
import sqlite3

from biofield_local_app import create_app
from dashboard.biofield_authoring import merge_dosing
from dashboard.biofield_interpret import build_interpret_prompt


CATALOG = {"dosage": "1 capsule", "frequency": "daily",
           "timing": "with extra water, away from beneficial oils"}


# --- guard 1: per-field fill ---------------------------------------------------

def test_merge_dosing_fills_only_the_empty_fields():
    got = merge_dosing("", "one a day", "", CATALOG)
    assert got == {"dosage": "1 capsule",              # from catalog
                   "frequency": "one a day",           # spoken wins
                   "timing": "with extra water, away from beneficial oils"}


def test_merge_dosing_keeps_everything_spoken():
    spoken = {"dosage": "2 caps", "frequency": "twice a day", "timing": "at bedtime"}
    assert merge_dosing(spoken["dosage"], spoken["frequency"], spoken["timing"],
                        CATALOG) == spoken


def test_merge_dosing_all_empty_takes_the_catalog():
    assert merge_dosing("", "", "", CATALOG) == CATALOG


def test_merge_dosing_tolerates_missing_catalog():
    assert merge_dosing("", "one a day", "", {}) == {
        "dosage": "", "frequency": "one a day", "timing": ""}


# --- guard 2: the prompt forbids inventing a dose ------------------------------

def test_prompt_forbids_inferring_an_unspoken_dose():
    sys_prompt = build_interpret_prompt("t")["system"]
    low = sys_prompt.lower()
    assert "never infer" in low or "do not infer" in low
    assert "not spoken" in low


# --- end to end through the interpret route ------------------------------------

def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    cx.executescript("""
      CREATE TABLE fmp_snap_products
        (product_name TEXT, dosage TEXT, dosage_freq TEXT, dosage_timing TEXT);
    """)
    cx.execute("INSERT INTO fmp_snap_products VALUES (?,?,?,?)",
               ("Fiber Cleanse", CATALOG["dosage"], CATALOG["frequency"], CATALOG["timing"]))
    cx.commit()
    from dashboard.biofield_authoring import init_auth_tables, create_test
    init_auth_tables(cx)
    tid = create_test(cx, "Debra Herndon", "chakamom1@gmail.com", "2026-07-09")
    cx.close()
    return db, tid


def test_spoken_frequency_does_not_suppress_catalog_dosage_and_timing(tmp_path, monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    db, tid = _db(tmp_path)

    # The model returns ONLY what was spoken: a frequency. (Post-fix prompt behavior.)
    def interpret_complete(system, user):
        return json.dumps({"header": "", "layers": [
            {"layer": 1, "head": "Lung Meridian", "most_affected": "Lung Meridian",
             "remedy": "Fiber Cleanse", "dosage": "", "frequency": "one a day",
             "timing": ""}]})

    app = create_app(db, interpret_complete=interpret_complete)
    c = app.test_client()
    c.post(f"/author/{tid}/session", json={"transcript": "Fiber Cleanse, one a day."})
    r = c.post(f"/author/{tid}/interpret", json={}).get_json()
    assert r.get("added") == 1, r

    cx = sqlite3.connect(db)
    row = cx.execute("SELECT dosage, frequency, timing FROM biofield_auth_chain "
                     "WHERE test_id=? ", (str(tid).lstrip('a'),)).fetchone()
    if row is None:   # id may be stored with the 'a' prefix
        row = cx.execute("SELECT dosage, frequency, timing FROM biofield_auth_chain").fetchone()
    dosage, frequency, timing = row
    assert frequency == "one a day"                 # spoken survives
    assert dosage == "1 capsule"                    # filled from catalog, not invented
    assert timing == "with extra water, away from beneficial oils"
    assert timing != "with food"                    # the bug this test exists for
