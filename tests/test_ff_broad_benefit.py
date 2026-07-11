"""Slice 2 of Condition Support Programs: wires the Slice-1 `broad_benefit`
store into the live FF matcher as a deterministic "broadly effective" signal.

TDD units for `_make_ff_items_for` (app.py) tagging candidates with
`broad_benefit` before handing them to `_ff_llm_rank`, and for `_ff_llm_rank`
rendering that flag as a "(broadly effective)" prompt marker. Reuses the
fixture shape from test_ff_llm_matcher.py / test_ff_matcher_distinct.py."""
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


EMAIL = "broadbenefitclient@example.com"
SCAN_DATE = "2026-07-10"

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

# Two ordinary, distinct, qty_eligible candidates -- one will be flagged
# broad_benefit, the other left un-flagged.
RAW_MATCHES = [
    {"id": "a", "score": 0.95, "metadata": {"name": "Terrain Restore",
                                             "meaning": "supports terrain"}},
    {"id": "b", "score": 0.9, "metadata": {"name": "Adrenal Restore",
                                            "meaning": "supports adrenal axis"}},
]

SLUGS = {"Terrain Restore": "terrain-restore", "Adrenal Restore": "adrenal-restore"}

PRODUCTS = {
    "terrain-restore": {"slug": "terrain-restore", "name": "Terrain Restore",
                        "qty_pricing": True, "info_only": False},
    "adrenal-restore": {"slug": "adrenal-restore", "name": "Adrenal Restore",
                        "qty_pricing": True, "info_only": False},
}


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
    # Only "terrain-restore" is in the broad_benefit store.
    monkeypatch.setattr(app, "_broad_benefit_slug_set", lambda: {"terrain-restore"})
    return app


# ---------------------------------------------------------------------------
# _make_ff_items_for tags candidates with broad_benefit (internal only)
# ---------------------------------------------------------------------------

def test_candidate_in_broad_benefit_store_gets_flag_true(make_env, monkeypatch):
    app = make_env
    captured = {}

    def _capture_rank(labels, candidates):
        captured["candidates"] = candidates
        return None  # force fallback path; we only care about the capture

    monkeypatch.setattr(app, "_ff_llm_rank", _capture_rank)
    app._make_ff_items_for(EMAIL, SCAN_DATE)

    by_slug = {c["slug"]: c for c in captured["candidates"]}
    assert by_slug["terrain-restore"]["broad_benefit"] is True


def test_candidate_not_in_broad_benefit_store_gets_flag_false(make_env, monkeypatch):
    app = make_env
    captured = {}

    def _capture_rank(labels, candidates):
        captured["candidates"] = candidates
        return None

    monkeypatch.setattr(app, "_ff_llm_rank", _capture_rank)
    app._make_ff_items_for(EMAIL, SCAN_DATE)

    by_slug = {c["slug"]: c for c in captured["candidates"]}
    assert by_slug["adrenal-restore"].get("broad_benefit") in (False, None)
    assert by_slug["adrenal-restore"]["broad_benefit"] is False


# ---------------------------------------------------------------------------
# broad_benefit never leaks into the final returned items
# ---------------------------------------------------------------------------

def test_broad_benefit_key_not_in_final_returned_items(make_env, monkeypatch):
    app = make_env
    monkeypatch.setattr(
        app, "_ff_llm_rank",
        lambda labels, candidates: [
            {"name": "Terrain Restore", "meaning": "LLM: broadly effective fit"},
            {"name": "Adrenal Restore", "meaning": "LLM: fits adrenal axis"},
        ])
    items = app._make_ff_items_for(EMAIL, SCAN_DATE)
    assert items  # sanity: got items
    for it in items:
        assert "broad_benefit" not in it
        assert set(it.keys()) == {"name", "slug", "url", "meaning"}


def test_broad_benefit_key_not_in_final_returned_items_fallback_path(make_env, monkeypatch):
    app = make_env
    monkeypatch.setattr(app, "_ff_llm_rank", lambda labels, candidates: None)
    items = app._make_ff_items_for(EMAIL, SCAN_DATE)
    assert items
    for it in items:
        assert "broad_benefit" not in it


# ---------------------------------------------------------------------------
# store-read failure fails closed -- matcher still returns items, never crashes
# ---------------------------------------------------------------------------

def test_real_broad_benefit_slug_set_fails_closed_on_store_error(monkeypatch):
    """The real `_broad_benefit_slug_set` must never raise -- even when the
    underlying store blows up -- and must return an empty set so every
    candidate is treated as not-broad."""
    app = _app()

    class _BoomConn:
        def __enter__(self):
            raise RuntimeError("cannot open db")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(app.sqlite3, "connect", lambda *a, **k: _BoomConn())
    result = app._broad_benefit_slug_set()
    assert result == set()


def test_matcher_still_returns_items_when_real_broad_benefit_lookup_errors(make_env, monkeypatch):
    """End-to-end: even if the real `_broad_benefit_slug_set` hits a broken
    store (sqlite3.connect itself raises), `_make_ff_items_for` must still
    return items -- never break the matcher."""
    app = make_env
    # Undo the fixture's stub so `_make_ff_items_for` calls the REAL
    # `_broad_benefit_slug_set`, then break its only DB dependency.
    monkeypatch.undo()
    monkeypatch.setattr(app, "_scan_recommendations_for", lambda e, d=None: RECS)
    monkeypatch.setattr(app, "_ff_query_specific_formulations",
                         lambda text, top_k: RAW_MATCHES)
    monkeypatch.setattr(app, "_resolve_buy_slug", _resolve)
    monkeypatch.setattr(app, "_get_product", _get_product)
    monkeypatch.setattr(app, "_ff_llm_rank", lambda labels, candidates: None)

    def _boom_connect(*a, **k):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(app.sqlite3, "connect", _boom_connect)

    items = app._make_ff_items_for(EMAIL, SCAN_DATE)
    assert items
    names = {it["name"] for it in items}
    assert names <= {"Terrain Restore", "Adrenal Restore"}


# ---------------------------------------------------------------------------
# _ff_llm_rank renders the "(broadly effective)" marker in the prompt
# ---------------------------------------------------------------------------

def test_ff_llm_rank_prompt_marks_broad_benefit_candidate(monkeypatch):
    app = _app()
    captured = {}

    class _FakeBlock:
        def __init__(self, text):
            self.text = text

    class _FakeMsg:
        def __init__(self, text):
            self.content = [_FakeBlock(text)]

    class _FakeMessages:
        def create(self, **kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            return _FakeMsg('[{"name": "Terrain Restore", "why": "fits"}]')

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(app, "_cl", _FakeClient())

    candidates = [
        {"name": "Terrain Restore", "slug": "terrain-restore",
         "url": "/begin/product/terrain-restore",
         "meaning": "supports terrain", "broad_benefit": True},
        {"name": "Adrenal Restore", "slug": "adrenal-restore",
         "url": "/begin/product/adrenal-restore",
         "meaning": "supports adrenal axis", "broad_benefit": False},
    ]
    out = app._ff_llm_rank(["Big Field Aligner"], candidates)
    assert out == [{"name": "Terrain Restore", "meaning": "fits"}]

    prompt = captured["prompt"]
    assert "Terrain Restore (broadly effective): supports terrain" in prompt
    assert "Adrenal Restore (broadly effective)" not in prompt
    assert "Adrenal Restore: supports adrenal axis" in prompt
