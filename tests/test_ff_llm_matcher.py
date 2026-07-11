"""Inline LLM FF-matcher upgrade (option B). TDD units:
`_ff_auto_excluded` (never-recommend hard filter), `_parse_ff_rank` (tolerant
JSON parse constrained to allowed candidate names), and the rewritten
`_make_ff_items_for` (LLM-ranked with vector fallback, same exclusions in
both paths, no `dosing` key ever)."""
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


# ---------------------------------------------------------------------------
# _ff_auto_excluded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Living Water Bottle",
    "living water bottle",
    "Electrolyte Mineral Manna",
    "Dental Regen Powder",
    "Endocrine Restore",
    "endocrine restore",
    "Comfort",
    "comfort",
    "AllerFree",
    "Fungifuge",
    "Bioavailability Blend",
    "Bioavailability Blend Powder",
])
def test_ff_auto_excluded_true_for_never_recommend_products(name):
    app = _app()
    assert app._ff_auto_excluded(name) is True


@pytest.mark.parametrize("name", [
    "Endocrine Restore Powder",
    "Comfort Synovial Syntropy",
    "Immune Modulation",
    "Magnesium",
    "Adrenal Restore",
    "Terrain Restore",
])
def test_ff_auto_excluded_false_for_ordinary_and_canonical(name):
    app = _app()
    assert app._ff_auto_excluded(name) is False


def test_ff_auto_excluded_false_for_empty():
    app = _app()
    assert app._ff_auto_excluded("") is False
    assert app._ff_auto_excluded(None) is False


# ---------------------------------------------------------------------------
# _parse_ff_rank
# ---------------------------------------------------------------------------

ALLOWED = {"Adrenal Restore", "Terrain Restore", "Immune Modulation"}


def test_parse_ff_rank_clean_json_kept_in_order():
    app = _app()
    text = (
        '[{"name": "Terrain Restore", "why": "supports terrain pattern"}, '
        '{"name": "Adrenal Restore", "why": "supports adrenal axis"}]'
    )
    out = app._parse_ff_rank(text, ALLOWED)
    assert [o["name"] for o in out] == ["Terrain Restore", "Adrenal Restore"]
    assert out[0]["meaning"] == "supports terrain pattern"


def test_parse_ff_rank_drops_hallucinated_name_not_in_allowed():
    app = _app()
    text = (
        '[{"name": "Adrenal Restore", "why": "ok"}, '
        '{"name": "Totally Made Up Product", "why": "hallucinated"}]'
    )
    out = app._parse_ff_rank(text, ALLOWED)
    assert [o["name"] for o in out] == ["Adrenal Restore"]
    assert all(o["name"] != "Totally Made Up Product" for o in out)


def test_parse_ff_rank_tolerates_markdown_fence_and_prose():
    app = _app()
    text = (
        "Sure, here is my ranked selection:\n"
        "```json\n"
        '[{"name": "Immune Modulation", "why": "supports immune terrain"}]\n'
        "```\n"
        "Let me know if you need anything else."
    )
    out = app._parse_ff_rank(text, ALLOWED)
    assert [o["name"] for o in out] == ["Immune Modulation"]


@pytest.mark.parametrize("garbage", [
    "",
    "not json at all",
    "[}garbage{]",
    "null",
    "{}",
])
def test_parse_ff_rank_garbage_returns_empty_list_never_raises(garbage):
    app = _app()
    assert app._parse_ff_rank(garbage, ALLOWED) == []


# ---------------------------------------------------------------------------
# _make_ff_items_for (integration: LLM path + vector fallback, exclusions)
# ---------------------------------------------------------------------------

EMAIL = "ffclient@example.com"
SCAN_DATE = "2026-07-02"

RECS = {
    "scan_date": SCAN_DATE,
    "scan_dates": [SCAN_DATE],
    "infoceuticals": [
        {"code": "BFA", "label": "Big Field Aligner", "rank": 1,
         "protocol_days": 15, "order_url": ""},
        {"code": "ED6", "label": "Heart", "rank": 2,
         "protocol_days": 15, "order_url": ""},
    ],
    "mihealth": [],
}

# raw retrieval results: includes a never-recommend product that must never
# survive into candidates, plus two resolvable ordinary products.
RAW_MATCHES = [
    {"id": "a", "score": 0.95, "metadata": {"name": "Bioavailability Blend",
                                             "meaning": "adjunct only"}},
    {"id": "b", "score": 0.9, "metadata": {"name": "Adrenal Restore",
                                            "meaning": "supports adrenal axis"}},
    {"id": "c", "score": 0.85, "metadata": {"name": "Terrain Restore",
                                             "meaning": "supports terrain"}},
]

SLUGS = {"Adrenal Restore": "adrenal-restore", "Terrain Restore": "terrain-restore"}


def _resolve(name):
    return SLUGS.get(name)


@pytest.fixture()
def make_env(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_scan_recommendations_for", lambda e, d=None: RECS)
    monkeypatch.setattr(app, "_ff_query_specific_formulations",
                         lambda text, top_k: RAW_MATCHES)
    monkeypatch.setattr(app, "_resolve_buy_slug", _resolve)
    return app


def test_make_ff_items_for_llm_path_orders_by_llm_and_excludes_bad_product(make_env, monkeypatch):
    app = make_env
    monkeypatch.setattr(
        app, "_ff_llm_rank",
        lambda labels, candidates: [
            {"name": "Terrain Restore", "meaning": "LLM: fits the terrain pattern"},
            {"name": "Adrenal Restore", "meaning": "LLM: fits adrenal axis"},
        ])
    items = app._make_ff_items_for(EMAIL, SCAN_DATE)
    assert [it["name"] for it in items] == ["Terrain Restore", "Adrenal Restore"]
    assert items[0]["meaning"] == "LLM: fits the terrain pattern"
    assert items[0]["slug"] == "terrain-restore"
    assert items[0]["url"] == "/begin/product/terrain-restore"
    assert all("dosing" not in it for it in items)
    assert all(it["name"] != "Bioavailability Blend" for it in items)


def test_make_ff_items_for_falls_back_to_vector_path_when_llm_rank_none(make_env, monkeypatch):
    app = make_env
    monkeypatch.setattr(app, "_ff_llm_rank", lambda labels, candidates: None)
    items = app._make_ff_items_for(EMAIL, SCAN_DATE)
    names = [it["name"] for it in items]
    assert "Bioavailability Blend" not in names
    assert set(names) <= {"Adrenal Restore", "Terrain Restore"}
    assert all("dosing" not in it for it in items)


def test_make_ff_items_for_no_recs_returns_empty(make_env, monkeypatch):
    app = make_env
    monkeypatch.setattr(app, "_scan_recommendations_for", lambda e, d=None: None)
    assert app._make_ff_items_for(EMAIL, SCAN_DATE) == []


def test_make_ff_items_for_llm_hallucination_dropped_via_candidate_constraint(make_env, monkeypatch):
    app = make_env
    # LLM returns a name that was never a candidate (never resolved/never a
    # real product) -- _make_ff_items_for must not surface it even if
    # _ff_llm_rank somehow let it through (defense-in-depth at the join).
    monkeypatch.setattr(
        app, "_ff_llm_rank",
        lambda labels, candidates: [
            {"name": "Totally Fictional Product", "meaning": "hallucinated"},
            {"name": "Adrenal Restore", "meaning": "LLM: fits adrenal axis"},
        ])
    items = app._make_ff_items_for(EMAIL, SCAN_DATE)
    assert [it["name"] for it in items] == ["Adrenal Restore"]
