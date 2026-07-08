"""Curated catalog vocabulary for Biofield Intake transcription (keyterm + glossary)."""
import sqlite3

from dashboard.biofield_catalog_terms import (
    CLINICAL_VOCAB, KEYTERM_TOKEN_BUDGET, build_terms, build_keyterms,
    keyterm_query, glossary_text, estimate_tokens, _strip_terrain, _is_equipment,
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


def test_apparel_and_devices_filtered_out():
    """Boost budget must not be spent on merch. Regression: 'Men's T Shirt' and
    'Metglas EMF Shielding Ribbon' were reaching Deepgram."""
    cx = _db()
    cx.executemany("INSERT INTO fmp_snap_products(product_name) VALUES (?)",
                   [("Men’s T Shirt",), ("Men’s Boxers",),
                    ("Women’s briefs",), ("Men’s Long Sleeve Shirt",),
                    ("Metglas EMF Shielding Ribbon",), ("Multiwave Oscillator",),
                    ("Oilcloth 48” x 36’ bonded vinyl",),
                    ("Nous Energy",)])
    cx.commit()
    terms = build_terms(cx)
    assert "Nous Energy" in terms
    for junk in ("Shirt", "Boxers", "briefs", "Sleeve", "Ribbon", "Oscillator",
                 "Oilcloth"):
        assert not any(junk.lower() in t.lower() for t in terms), junk


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


# --- Deepgram's REAL limit: 500 tokens across all keyterms -------------------
# Shipping a count cap of 100 sent ~650 tokens and hard-failed the socket with a
# bare 400. These lock the budget in.

def test_estimate_tokens_is_conservative():
    # never fewer tokens than words, and roughly 3 chars/token for rare words
    assert estimate_tokens("Perelandra") >= 3
    assert estimate_tokens("a b c d") >= 4          # word floor
    assert estimate_tokens("") == 0
    assert estimate_tokens("gemmotherapy") >= 4


def test_build_keyterms_stays_under_deepgram_token_budget():
    cx = _db()
    cx.executemany("INSERT INTO fmp_snap_products(product_name) VALUES (?)",
                   [(f"Botanical Formulation Number {i}",) for i in range(400)])
    cx.commit()
    kt = build_keyterms(cx)
    total = sum(estimate_tokens(t) for t in kt)
    assert total <= KEYTERM_TOKEN_BUDGET
    assert KEYTERM_TOKEN_BUDGET < 500          # headroom under the hard limit
    assert len(kt) <= 100


def test_token_budget_keeps_highest_priority_terms():
    cx = _db()
    cx.executemany("INSERT INTO fmp_snap_products(product_name) VALUES (?)",
                   [(f"Filler Formulation {i}",) for i in range(400)])
    cx.commit()
    kt = build_keyterms(cx)
    # clinical vocab is priority 1 -- it must never be crowded out by products
    assert "Terrain Restore" in kt
    assert "infoceutical" in kt


def test_token_budget_skips_overflowing_term_but_keeps_later_cheap_ones():
    cx = _db()
    cx.execute("INSERT INTO fmp_snap_products(product_name) VALUES (?)",
               ("Z" * 3000,))          # absurdly expensive single term
    cx.execute("INSERT INTO fmp_snap_products(product_name) VALUES (?)", ("Qi",))
    cx.commit()
    terms = build_terms(cx, token_budget=200)
    assert not any(len(t) > 100 for t in terms)   # the hog is skipped
    assert sum(estimate_tokens(t) for t in terms) <= 200


def test_build_terms_unbudgeted_by_default():
    """The LLM glossary has no token limit, so the default builder must not apply
    the Deepgram token budget -- it yields strictly more terms than build_keyterms."""
    cx = _db()
    cx.executemany("INSERT INTO fmp_snap_products(product_name) VALUES (?)",
                   [(f"Botanical Formulation Number {i}",) for i in range(400)])
    cx.commit()
    glossary = build_terms(cx, cap=300)          # no token_budget
    keyterms = build_keyterms(cx)                # count cap + token budget
    assert len(glossary) > len(keyterms)
    assert sum(estimate_tokens(t) for t in glossary) > KEYTERM_TOKEN_BUDGET


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
