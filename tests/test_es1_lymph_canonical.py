"""ES1 is Lymph, not Immune — and fixing that needs the matcher, not just the catalog.

`resolve_remedy_name` fuzzy-matches spoken remedy names against **FMP's** product
names. FMP has exactly one ES1 row and it is misnamed "Immune". So saying the true
name auto-corrected to the false one, which is how "ES1 Immune Energetic Star
Infoceutical" reached a client's published report while its reorder line pointed at
the OTHER twin's slug (`es1-lymph`, the record with the real storefront URL).

Retiring the misnamed twin alone makes it WORSE: it strips the only ES1 from the
candidate pool, and the matcher reaches for a neighbour —

    "ES1 Immune Energetic Star Infoceutical" -> ES5 Auto-Immune ...
    "ES1 Lymph  Energetic Star Infoceutical" -> ES8 Chill ...

Different remedies, silently, on a causal chain. Three changes, tested here:

  1. the candidate pool also carries ACTIVE CATALOG names, so a survivor that lives
     only in the catalog (no `fmp_id`) is matchable at all;
  2. an exact spoken match on a RETIRED name redirects to its survivor's canonical
     name rather than fuzzy-matching onto a stranger;
  3. `es1-lymph` becomes the canonical, long-form ES1 record.
"""
import json
import sqlite3
from pathlib import Path

import pytest

import dashboard.biofield_authoring as ba
from dashboard.biofield_authoring import resolve_remedy_name, _norm_name

ROOT = Path(__file__).resolve().parent.parent

LIVE_ES1 = "es1-lymph"
DEAD_ES1 = "es1-immune-energetic-star-infoceutical"
DEAD_NAME = "ES1 Immune Energetic Star Infoceutical"
LIVE_NAME = "ES1 Lymph Energetic Star Infoceutical"
ES5_NAME = "ES5 Auto-Immune Energetic Star Infoceutical"


@pytest.fixture(autouse=True)
def _clear_caches():
    for fn in ("_deprecated_catalog_names", "_active_catalog_names", "_superseded_name_map", "_catalog_alias_map"):
        f = getattr(ba, fn, None)
        if f is not None and hasattr(f, "cache_clear"):
            f.cache_clear()
    yield
    for fn in ("_deprecated_catalog_names", "_active_catalog_names", "_superseded_name_map", "_catalog_alias_map"):
        f = getattr(ba, fn, None)
        if f is not None and hasattr(f, "cache_clear"):
            f.cache_clear()


def _products():
    return json.loads((ROOT / "data" / "products.json").read_text())["products"]


def _fmp_db():
    """FMP as it really is: one ES1 row, misnamed, plus its nearest neighbours."""
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE fmp_snap_products (product_name TEXT, active TEXT)")
    cx.executemany("INSERT INTO fmp_snap_products(product_name, active) VALUES (?, 'Yes')",
                   [(DEAD_NAME,), (ES5_NAME,),
                    ("ES8 Chill Energetic Star Infoceutical",),
                    ("ED9 Muscle Energetic Driver Infoceutical",)])
    cx.commit()
    return cx


# --- the catalog side ---------------------------------------------------------

def test_the_misnamed_twin_is_retired_and_points_at_the_survivor():
    p = _products()
    assert p[DEAD_ES1]["inactive"] is True
    assert p[DEAD_ES1]["superseded_by"] == LIVE_ES1


def test_the_survivor_is_the_sellable_storefront_record_named_lymph():
    p = _products()
    assert not p[LIVE_ES1].get("inactive")
    assert p[LIVE_ES1]["name"] == LIVE_NAME
    assert "es1-lymph" in p[LIVE_ES1]["url"]


def test_the_survivors_pinecone_title_is_unchanged():
    """Renaming pinecone_title would orphan its vector; `name` may move, the title may not."""
    assert _products()[LIVE_ES1]["pinecone_title"] == "ES1"


# --- change 1: active catalog names join the pool ------------------------------

def test_active_catalog_names_include_the_survivor_and_exclude_the_retired():
    names = ba._active_catalog_names()
    assert _norm_name(LIVE_NAME) in names
    assert _norm_name(DEAD_NAME) not in names


def test_a_catalog_only_survivor_is_matchable():
    """es1-lymph has no fmp_id, so before this it could never be matched."""
    assert resolve_remedy_name(_fmp_db(), LIVE_NAME) == LIVE_NAME


# --- change 2: a retired name redirects, it does not drift ---------------------

def test_saying_the_retired_name_redirects_to_the_survivor():
    """THE regression. Retiring the twin alone sent this to ES5 Auto-Immune."""
    got = resolve_remedy_name(_fmp_db(), DEAD_NAME)
    assert got == LIVE_NAME
    assert "Auto-Immune" not in got


def test_the_true_name_no_longer_autocorrects_to_the_false_one():
    """The original defect: saying Lymph came back as Immune."""
    assert resolve_remedy_name(_fmp_db(), LIVE_NAME) != DEAD_NAME


def test_the_bare_code_still_resolves_to_the_survivor_not_es15():
    """Renaming the survivor took the bare code "ES1" out of the pool and the matcher
    grabbed ES15 (Heavy Metals). `pinecone_title` keeps "ES1" spellable."""
    got = resolve_remedy_name(_fmp_db(), "ES1")
    assert got == LIVE_NAME, f"bare code drifted to {got!r}"


def test_an_alias_resolves_to_the_canonical_name_not_the_alias():
    assert resolve_remedy_name(_fmp_db(), "es1") == LIVE_NAME


def test_a_neighbouring_remedy_is_not_collateral_damage():
    assert resolve_remedy_name(_fmp_db(), ES5_NAME) == ES5_NAME


def test_an_unrelated_fmp_remedy_still_resolves_to_itself():
    assert resolve_remedy_name(_fmp_db(), "ED9 Muscle Energetic Driver Infoceutical") == \
        "ED9 Muscle Energetic Driver Infoceutical"


# --- and after FMP itself is renamed at the source -----------------------------

def _fmp_db_renamed():
    """FMP once its ES1 row has been renamed to Lymph (Glen, 2026-07-09). The local
    snapshot lags until the next sync, so both worlds must resolve identically."""
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE fmp_snap_products (product_name TEXT, active TEXT)")
    cx.executemany("INSERT INTO fmp_snap_products(product_name, active) VALUES (?, 'Yes')",
                   [(LIVE_NAME,), (ES5_NAME,),
                    ("ES8 Chill Energetic Star Infoceutical",)])
    cx.commit()
    return cx


@pytest.mark.parametrize("spoken", ["ES1", LIVE_NAME, DEAD_NAME])
def test_resolution_is_identical_before_and_after_the_fmp_rename(spoken):
    assert resolve_remedy_name(_fmp_db(), spoken) == LIVE_NAME
    assert resolve_remedy_name(_fmp_db_renamed(), spoken) == LIVE_NAME


def test_the_renamed_fmp_row_does_not_reintroduce_the_immune_name():
    got = resolve_remedy_name(_fmp_db_renamed(), "ES1 Immune Energetic Star Infoceutical")
    assert "Immune" not in got
    assert "Auto-Immune" not in got


# --- the pool extension must not strand anything -------------------------------

def test_every_superseded_pointer_still_lands_on_a_live_product():
    p = _products()
    for slug, rec in p.items():
        tgt = rec.get("superseded_by")
        if tgt:
            assert tgt in p, f"{slug} points at missing {tgt}"
            assert not p[tgt].get("inactive"), f"{slug} points at inactive {tgt}"


def test_no_active_catalog_name_is_also_a_retired_name():
    """A name that is both live and retired would make the redirect nondeterministic."""
    assert not (ba._active_catalog_names() & ba._deprecated_catalog_names())
