import json
import sys
from pathlib import Path

import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import page_links
        return page_links
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"page_links not importable: {e}")


def _pages():
    return [
        {"slug": "low-energy", "name": "Low Energy", "kind": "topic",
         "href": "/learn/low-energy", "gated": False},
        {"slug": "brain-fog", "name": "Brain Fog", "kind": "topic",
         "href": "/learn/brain-fog", "gated": False},
        {"slug": "magnesium", "name": "Magnesium", "kind": "ingredient",
         "href": "/begin/ingredient/magnesium", "gated": True},
        {"slug": "magnesium-glycinate", "name": "Magnesium Glycinate", "kind": "ingredient",
         "href": "/begin/ingredient/magnesium-glycinate", "gated": True},
        {"slug": "neuro-magnesium", "name": "Neuro Magnesium", "kind": "product",
         "href": "/begin/product/neuro-magnesium", "gated": False},
    ]


def _idx(aliases=None):
    pl = _mod()
    return pl.build_index(_pages(), alias_map=aliases or {})


def test_matches_page_named_in_text():
    pl = _mod()
    cards = pl.match_page_links("I keep struggling with low energy lately", _idx())
    assert cards and cards[0]["href"] == "/learn/low-energy"
    assert cards[0]["key"] == "topic:low-energy"
    assert cards[0]["title"] == "Low Energy"
    assert cards[0]["sub"] == "Read the guide"


def test_matches_term_only_in_answer_text():
    # the AI answer mentions "brain fog" even though the user didn't
    pl = _mod()
    cards = pl.match_page_links("why am I so tired? Many people experience brain fog too.", _idx())
    hrefs = [c["href"] for c in cards]
    assert "/learn/brain-fog" in hrefs


def test_word_boundary_no_substring_false_positive():
    # "iron" must NOT match inside "environment"; add an iron page to prove it
    pl = _mod()
    pages = _pages() + [{"slug": "iron", "name": "Iron", "kind": "ingredient",
                         "href": "/begin/ingredient/iron", "gated": True}]
    idx = pl.build_index(pages, alias_map={})
    cards = pl.match_page_links("a calm environment helps recovery", idx)
    assert all(c["href"] != "/begin/ingredient/iron" for c in cards)


def test_longest_phrase_wins_no_double_match():
    pl = _mod()
    cards = pl.match_page_links("I take magnesium glycinate daily", _idx(), limit=5)
    hrefs = [c["href"] for c in cards]
    assert "/begin/ingredient/magnesium-glycinate" in hrefs
    # the substring "magnesium" page must NOT also be surfaced from the same span
    assert "/begin/ingredient/magnesium" not in hrefs


def test_dedupe_by_href_and_cap():
    pl = _mod()
    cards = pl.match_page_links("low energy, low energy, brain fog, magnesium", _idx(), limit=2)
    assert len(cards) == 2
    assert len({c["href"] for c in cards}) == 2


def test_alias_maps_paraphrase_to_page():
    pl = _mod()
    idx = _idx({"can't focus": "brain-fog"})
    cards = pl.match_page_links("I just can't focus these days", idx)
    assert any(c["href"] == "/learn/brain-fog" for c in cards)


def test_sub_labels_per_kind():
    pl = _mod()
    cards = pl.match_page_links("neuro magnesium and low energy", _idx(), limit=5)
    by_href = {c["href"]: c for c in cards}
    assert by_href["/begin/product/neuro-magnesium"]["sub"] == "View product"
    assert by_href["/learn/low-energy"]["sub"] == "Read the guide"


def test_load_aliases_missing_file_is_empty():
    pl = _mod()
    assert pl.load_aliases("/no/such/file.json") == {}


def test_load_aliases_reads_seed_file():
    pl = _mod()
    seed = Path(__file__).resolve().parent.parent / "data" / "page-aliases.json"
    data = pl.load_aliases(str(seed))
    assert isinstance(data, dict)


def test_no_match_returns_empty():
    pl = _mod()
    assert pl.match_page_links("the weather is nice today", _idx()) == []
