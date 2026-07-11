"""FF matcher scoped to DISTINCT, qty-eligible Functional Formulations.

Root cause fixed here: the FF candidate pool used to include the client's own
E4L infoceuticals (info_only, `_qty_eligible` False) because nothing filtered
on that flag. This scopes both the LLM candidate-build path and the vector
fallback to `_qty_eligible` products that are NOT already on the client's scan
card (`scan_slugs`, parsed from each infoceutical's `order_url`), so the FF
card surfaces distinct Functional Formulation supplements instead of echoing
the scan. TDD units for `_make_ff_items_for` (app.py) only -- `_ff_auto_excluded`
and `_parse_ff_rank` are unchanged and already covered by test_ff_llm_matcher.py."""
import importlib
import sys
from pathlib import Path

import pytest


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


EMAIL = "distinctclient@example.com"
SCAN_DATE = "2026-07-09"

# Scan card already shows these two E4L infoceuticals -- their slugs must
# never reappear as FF matches.
RECS = {
    "scan_date": SCAN_DATE,
    "scan_dates": [SCAN_DATE],
    "infoceuticals": [
        {"code": "BFA", "label": "Big Field Aligner", "rank": 1,
         "protocol_days": 15, "order_url": "/begin/product/big-field-aligner"},
        {"code": "ED6", "label": "Heart", "rank": 2,
         "protocol_days": 15, "order_url": "/begin/product/heart-syntropy"},
    ],
    "mihealth": [],
}

# Retrieval pool mixes:
#  - an E4L infoceuticals product (info_only, qty_eligible False) -- must be
#    dropped by the qty_eligible gate even though it's not on the scan card.
#  - the scan-card product itself (would otherwise pass qty_eligible) -- must
#    be dropped by the scan_slugs gate.
#  - a never-recommend product -- must be dropped by _ff_auto_excluded.
#  - three ordinary qty_eligible Functional Formulations -- the only ones
#    that should survive into candidates.
RAW_MATCHES = [
    {"id": "e4l", "score": 0.99, "metadata": {"name": "Heart Field Balance",
                                               "meaning": "E4L infoceutical, info only"}},
    {"id": "scancard", "score": 0.97, "metadata": {"name": "Heart Syntropy",
                                                    "meaning": "already on scan card"}},
    {"id": "excluded", "score": 0.93, "metadata": {"name": "AllerFree",
                                                    "meaning": "never-recommend"}},
    {"id": "ff1", "score": 0.9, "metadata": {"name": "Terrain Restore",
                                              "meaning": "broadly-effective terrain support"}},
    {"id": "ff2", "score": 0.88, "metadata": {"name": "Adrenal Syntropy",
                                               "meaning": "supports adrenal axis"}},
    {"id": "ff3", "score": 0.85, "metadata": {"name": "Immune Modulation",
                                               "meaning": "supports immune terrain"}},
]

SLUGS = {
    "Heart Field Balance": "heart-field-balance",
    "Heart Syntropy": "heart-syntropy",
    "AllerFree": "allerfree",
    "Terrain Restore": "terrain-restore",
    "Adrenal Syntropy": "adrenal-syntropy",
    "Immune Modulation": "immune-modulation",
}

# Product records keyed by slug. _get_product returns None for slugs not
# present (simulating "not resolvable / not sellable").
PRODUCTS = {
    "heart-field-balance": {"slug": "heart-field-balance", "name": "Heart Field Balance",
                             "info_only": True, "qty_pricing": False},
    "heart-syntropy": {"slug": "heart-syntropy", "name": "Heart Syntropy",
                       "qty_pricing": True, "info_only": False},
    "terrain-restore": {"slug": "terrain-restore", "name": "Terrain Restore",
                        "qty_pricing": True, "info_only": False},
    "adrenal-syntropy": {"slug": "adrenal-syntropy", "name": "Adrenal Syntropy",
                         "qty_pricing": True, "info_only": False},
    "immune-modulation": {"slug": "immune-modulation", "name": "Immune Modulation",
                          "qty_pricing": True, "info_only": False},
    # note: "allerfree" intentionally absent -- _ff_auto_excluded should drop
    # it before slug resolution is even relevant.
}

FF_SLUGS = {"terrain-restore", "adrenal-syntropy", "immune-modulation"}


def _resolve(name):
    return SLUGS.get(name)


def _get_product(slug):
    return PRODUCTS.get(slug)


@pytest.fixture()
def make_env(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_scan_recommendations_for", lambda e, d=None: RECS)
    monkeypatch.setattr(app, "_ff_query_specific_formulations",
                         lambda text, top_k: RAW_MATCHES)
    monkeypatch.setattr(app, "_resolve_buy_slug", _resolve)
    monkeypatch.setattr(app, "_get_product", _get_product)
    return app


def test_candidate_pool_scoped_to_qty_eligible_non_scan_slugs(make_env, monkeypatch):
    """The candidates handed to the LLM path are ONLY the 3 qty_eligible FFs --
    never the E4L info_only product, never the scan-card product, never the
    excluded product."""
    app = make_env
    seen_candidates = {}

    def _capture_rank(labels, candidates):
        seen_candidates["candidates"] = candidates
        return None  # force fallback so we don't need to also fake the LLM ranking here

    monkeypatch.setattr(app, "_ff_llm_rank", _capture_rank)
    app._make_ff_items_for(EMAIL, SCAN_DATE)

    names = {c["name"] for c in seen_candidates["candidates"]}
    assert names == {"Terrain Restore", "Adrenal Syntropy", "Immune Modulation"}
    slugs = {c["slug"] for c in seen_candidates["candidates"]}
    assert slugs == FF_SLUGS


def test_llm_path_returns_distinct_qty_eligible_items_no_dosing(make_env, monkeypatch):
    app = make_env
    monkeypatch.setattr(
        app, "_ff_llm_rank",
        lambda labels, candidates: [
            {"name": "Terrain Restore", "meaning": "LLM: broadly-effective terrain support"},
            {"name": "Immune Modulation", "meaning": "LLM: supports immune terrain"},
        ])
    items = app._make_ff_items_for(EMAIL, SCAN_DATE)
    assert [it["name"] for it in items] == ["Terrain Restore", "Immune Modulation"]
    for it in items:
        assert it["slug"] in FF_SLUGS
        assert "dosing" not in it
        assert set(it.keys()) == {"name", "slug", "url", "meaning"}
    # never the E4L / scan-card / excluded products
    names = [it["name"] for it in items]
    assert "Heart Field Balance" not in names
    assert "Heart Syntropy" not in names
    assert "AllerFree" not in names


def test_fallback_restricted_to_qty_eligible_non_scan_slug_ffs(make_env, monkeypatch):
    """When _ff_llm_rank signals unavailable (None), the vector fallback must
    still only ever emit qty_eligible, non-scan-slug FF products."""
    app = make_env
    monkeypatch.setattr(app, "_ff_llm_rank", lambda labels, candidates: None)
    items = app._make_ff_items_for(EMAIL, SCAN_DATE)
    names = {it["name"] for it in items}
    assert names <= {"Terrain Restore", "Adrenal Syntropy", "Immune Modulation"}
    assert "Heart Field Balance" not in names
    assert "Heart Syntropy" not in names
    assert "AllerFree" not in names
    for it in items:
        assert "dosing" not in it


def test_no_qty_eligible_candidates_returns_empty(make_env, monkeypatch):
    """If the retrieval pool has no surviving qty_eligible FF candidates,
    _make_ff_items_for must return [] -- NOT fall through to an unfiltered
    pool (e.g. the E4L infoceuticals)."""
    app = make_env
    monkeypatch.setattr(
        app, "_ff_query_specific_formulations",
        lambda text, top_k: [
            {"id": "e4l", "score": 0.99, "metadata": {"name": "Heart Field Balance",
                                                       "meaning": "info only"}},
            {"id": "scancard", "score": 0.97, "metadata": {"name": "Heart Syntropy",
                                                            "meaning": "already on scan card"}},
            {"id": "excluded", "score": 0.93, "metadata": {"name": "AllerFree",
                                                            "meaning": "never-recommend"}},
        ])
    # _ff_llm_rank should never even be reached with usable candidates, but
    # define it anyway to prove it's never asked to hallucinate a match.
    monkeypatch.setattr(app, "_ff_llm_rank", lambda labels, candidates: None)
    assert app._make_ff_items_for(EMAIL, SCAN_DATE) == []


def test_scan_slugs_parsed_from_order_url_last_segment(make_env, monkeypatch):
    """A product resolvable to a slug that matches the scan card's order_url
    tail must be excluded even though it is otherwise qty_eligible."""
    app = make_env
    captured = {}

    def _capture_rank(labels, candidates):
        captured["candidates"] = candidates
        return None

    monkeypatch.setattr(app, "_ff_llm_rank", _capture_rank)
    app._make_ff_items_for(EMAIL, SCAN_DATE)
    slugs = {c["slug"] for c in captured["candidates"]}
    assert "heart-syntropy" not in slugs  # scan card order_url tail
