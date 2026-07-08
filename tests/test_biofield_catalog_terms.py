"""Curated catalog vocabulary for Biofield Intake transcription (keyterm + glossary)."""
import sqlite3

from dashboard.biofield_catalog_terms import (
    CLINICAL_VOCAB, build_terms, keyterm_query, glossary_text, _strip_terrain,
    _is_equipment,
)


def _db():
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE biofield_auth_chain (remedy TEXT)")
    cx.execute("CREATE TABLE fmp_snap_products (product_name TEXT)")
    return cx


def test_clinical_vocab_always_present_and_first():
    cx = _db()
    terms = build_terms(cx)
    for v in CLINICAL_VOCAB:
        assert v in terms
    # vocab leads the list (highest priority)
    assert terms[: len(CLINICAL_VOCAB)] == CLINICAL_VOCAB


def test_prescribed_remedies_included_and_terrain_stripped():
    cx = _db()
    cx.executemany("INSERT INTO biofield_auth_chain(remedy) VALUES (?)",
                   [("Nous Energy",), ("Nous Energy",),
                    ("Foxglove Flower Essence in Terrain Restore",)])
    cx.commit()
    terms = build_terms(cx)
    assert "Nous Energy" in terms
    # the " in Terrain Restore" delivery suffix is stripped to the core name
    assert "Foxglove Flower Essence" in terms
    assert "Foxglove Flower Essence in Terrain Restore" not in terms


def test_equipment_and_packaging_filtered_out():
    cx = _db()
    cx.executemany("INSERT INTO fmp_snap_products(product_name) VALUES (?)",
                   [("100 mL wide mouth bottle for 30 caps",),
                    ("172 Hz Tuning Fork with hammer",),
                    ("23 Piece Cookware Set",),
                    ("Sulfur Syntropy",)])
    cx.commit()
    terms = build_terms(cx)
    assert "Sulfur Syntropy" in terms
    assert not any("Tuning Fork" in t for t in terms)
    assert not any("Cookware" in t for t in terms)
    assert not any(t.strip()[0].isdigit() for t in terms)


def test_dedup_case_insensitive():
    cx = _db()
    cx.executemany("INSERT INTO biofield_auth_chain(remedy) VALUES (?)",
                   [("Fungifuge",), ("fungifuge",), ("FUNGIFUGE",)])
    cx.commit()
    terms = build_terms(cx)
    assert sum(1 for t in terms if t.lower() == "fungifuge") == 1


def test_cap_respected_and_priority_survives():
    cx = _db()
    cx.executemany("INSERT INTO fmp_snap_products(product_name) VALUES (?)",
                   [(f"Formula Number {i}",) for i in range(500)])
    cx.commit()
    terms = build_terms(cx, cap=30)
    assert len(terms) == 30
    # clinical vocab is priority 1, so it is never crowded out by products
    assert "Terrain Restore" in terms


def test_keyterm_query_encoding():
    q = keyterm_query(["Nous Energy", "Terrain Restore"])
    assert q == "&keyterm=Nous%20Energy&keyterm=Terrain%20Restore"
    assert keyterm_query([]) == ""


def test_glossary_text():
    assert glossary_text(["A", "B", "C"]) == "A, B, C"


def test_helpers():
    assert _strip_terrain("X in Terrain Restore") == "X"
    assert _strip_terrain("Plain Name") == "Plain Name"
    assert _is_equipment("172 Hz Tuning Fork")
    assert _is_equipment("100 capsule machine")
    assert not _is_equipment("Nous Energy")


def test_missing_tables_returns_just_vocab():
    cx = sqlite3.connect(":memory:")  # no catalog tables at all
    terms = build_terms(cx)
    assert terms == CLINICAL_VOCAB


# --- Lever 2: catalog glossary reaches the interpret prompt -------------------

def test_interpret_prompt_includes_glossary_when_given():
    from dashboard.biofield_interpret import build_interpret_prompt
    p = build_interpret_prompt("t", glossary="Nous Energy, Fungifuge")
    assert "Nous Energy, Fungifuge" in p["system"]
    assert "keep it EXACTLY" in p["system"]  # conservative instruction present


def test_interpret_prompt_unchanged_without_glossary():
    from dashboard.biofield_interpret import build_interpret_prompt, _SYSTEM
    assert build_interpret_prompt("t")["system"] == _SYSTEM
    assert build_interpret_prompt("t", glossary="")["system"] == _SYSTEM


def test_interpret_transcript_threads_glossary_through():
    from dashboard.biofield_interpret import interpret_transcript
    seen = {}

    def fake_complete(system, user):
        seen["system"] = system
        return '{"header":"", "layers":[]}'

    interpret_transcript("balanced by nous energy", fake_complete,
                         glossary="Nous Energy")
    assert "Nous Energy" in seen["system"]
