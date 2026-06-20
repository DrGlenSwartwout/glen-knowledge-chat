"""Tests for dashboard/ingredient_copy.py.

Uses a mock Anthropic client - does NOT call the real model or Pinecone.
"""
import sys
from pathlib import Path
import pytest


def _mod():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import ingredient_copy
        return ingredient_copy
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


# ---------------------------------------------------------------------------
# Minimal fake Anthropic client fixtures
# ---------------------------------------------------------------------------

class _FakeContent:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _FakeMsg:
    def __init__(self, text):
        self.content = [_FakeContent(text)]


class _FakeClient:
    """Returns a canned JSON response for propose_curation calls."""

    def __init__(self, response_text):
        self._response_text = response_text
        self.messages = self

    def create(self, **kw):
        return _FakeMsg(self._response_text)


class _ErrorClient:
    """Always raises to exercise the safe-default path."""

    def __init__(self):
        self.messages = self

    def create(self, **kw):
        raise RuntimeError("simulated API failure")


_CURATION_JSON = """{
    "research_score": 8,
    "traditional_score": 6,
    "related_forms": [
        {"name": "Zinc Glycinate", "slug": "zinc-glycinate", "verdict": "superior", "note": "Higher bioavailability"},
        {"name": "Zinc Oxide", "slug": "zinc-oxide", "verdict": "inferior", "note": "Poor absorption"}
    ],
    "traditional_use": [
        {"system": "TCM", "formula": "Zhi Bai Di Huang Wan", "uses": "Kidney and Liver support", "forms": "decoction, pill"},
        {"system": "Ayurveda", "formula": "Jasad Bhasma", "uses": "Digestive support", "forms": "bhasma (calcined powder)"}
    ]
}"""

_INGREDIENT = {
    "name": "Zinc",
    "fmp": {
        "scientific": "Zinc (as Zinc Bisglycinate Chelate)",
        "label_form": "Zinc",
        "active": "Yes",
        "rda_mg": "11",
    },
    "studies": [
        {
            "study_title": "Zinc supplementation and immune function",
            "publication": "Journal of Nutrition",
            "year": 2020,
            "url": "https://example.com/study1",
            "text": "A randomized trial showing zinc supports T-cell activity.",
        },
        {
            "study_title": "Zinc and antioxidant defense",
            "publication": "Nutrients",
            "year": 2021,
            "url": "https://example.com/study2",
            "text": "Zinc promotes superoxide dismutase activity.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Tests: build_section_prompt
# ---------------------------------------------------------------------------

def test_build_section_prompt_returns_tuple():
    m = _mod()
    for section in ("what_it_is", "research"):
        result = m.build_section_prompt(section, _INGREDIENT)
        assert isinstance(result, tuple) and len(result) == 2
        system, user = result
        assert isinstance(system, str) and isinstance(user, str)


def test_build_section_prompt_grounded_in_name():
    m = _mod()
    for section in ("what_it_is", "research"):
        _, user = m.build_section_prompt(section, _INGREDIENT)
        assert "Zinc" in user, f"section '{section}' user prompt missing ingredient name"


def test_build_section_prompt_contains_fmp_data():
    m = _mod()
    _, user = m.build_section_prompt("what_it_is", _INGREDIENT)
    # The fmp block should surface at least one known field value
    assert "Zinc Bisglycinate" in user or "label_form" in user or "rda_mg" in user


def test_build_section_prompt_contains_study_titles():
    m = _mod()
    _, user = m.build_section_prompt("research", _INGREDIENT)
    assert "immune function" in user or "antioxidant" in user


def test_build_section_prompt_compliance_in_system():
    m = _mod()
    system, _ = m.build_section_prompt("what_it_is", _INGREDIENT)
    assert "disease" in system.lower() or "structure/function" in system.lower()
    assert "em dash" in system.lower() or "em dashes" in system.lower()


def test_build_section_prompt_invalid_section_raises():
    m = _mod()
    with pytest.raises(KeyError):
        m.build_section_prompt("bogus_section", _INGREDIENT)


# ---------------------------------------------------------------------------
# Tests: propose_curation - success path
# ---------------------------------------------------------------------------

def test_propose_curation_returns_clamped_scores():
    m = _mod()
    client = _FakeClient(_CURATION_JSON)
    result = m.propose_curation(_INGREDIENT, client)
    assert result["research_score"] == 8
    assert result["traditional_score"] == 6


def test_propose_curation_score_clamped_above_10():
    m = _mod()
    high_json = '{"research_score": 15, "traditional_score": 0, "related_forms": [], "traditional_use": []}'
    client = _FakeClient(high_json)
    result = m.propose_curation(_INGREDIENT, client)
    assert result["research_score"] == 10, "score above 10 must clamp to 10"
    # traditional_score 0 clamps to 1
    assert result["traditional_score"] == 1, "score 0 must clamp to 1"


def test_propose_curation_returns_related_forms():
    m = _mod()
    client = _FakeClient(_CURATION_JSON)
    result = m.propose_curation(_INGREDIENT, client)
    forms = result["related_forms"]
    assert isinstance(forms, list) and len(forms) == 2
    slugs = {f["slug"] for f in forms}
    verdicts = {f["verdict"] for f in forms}
    assert "zinc-glycinate" in slugs
    assert "zinc-oxide" in slugs
    assert "superior" in verdicts and "inferior" in verdicts


def test_propose_curation_returns_traditional_use():
    m = _mod()
    client = _FakeClient(_CURATION_JSON)
    result = m.propose_curation(_INGREDIENT, client)
    trad = result["traditional_use"]
    assert isinstance(trad, list) and len(trad) == 2
    systems = {t["system"] for t in trad}
    assert "TCM" in systems and "Ayurveda" in systems


def test_propose_curation_related_forms_slugs_are_kebab():
    """Verify that slugs in related_forms are computed via ingredients.slugify."""
    m = _mod()
    # provide a form whose slug needs normalization
    json_text = ('{"research_score": 7, "traditional_score": 5, '
                 '"related_forms": [{"name": "Zinc L-Carnosine", "slug": "ignored", '
                 '"verdict": "comparable", "note": "Gut-specific"}], '
                 '"traditional_use": []}')
    client = _FakeClient(json_text)
    result = m.propose_curation(_INGREDIENT, client)
    forms = result["related_forms"]
    assert len(forms) == 1
    # slug must be re-computed from the name by ingredients.slugify
    from dashboard import ingredients as _ing
    expected_slug = _ing.slugify("Zinc L-Carnosine")
    assert forms[0]["slug"] == expected_slug


# ---------------------------------------------------------------------------
# Tests: propose_curation - failure path (safe defaults)
# ---------------------------------------------------------------------------

def test_propose_curation_safe_defaults_on_api_error():
    m = _mod()
    client = _ErrorClient()
    result = m.propose_curation(_INGREDIENT, client)
    assert result == {
        "research_score": None,
        "traditional_score": None,
        "related_forms": [],
        "traditional_use": [],
    }


def test_propose_curation_safe_defaults_on_bad_json():
    m = _mod()
    client = _FakeClient("not valid json at all {{{")
    result = m.propose_curation(_INGREDIENT, client)
    assert result["research_score"] is None
    assert result["related_forms"] == []
    assert result["traditional_use"] == []


def test_propose_curation_never_raises():
    """propose_curation must never propagate an exception, even with a broken client."""
    m = _mod()

    class _BrokenClient:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            raise Exception("unexpected total failure")

    # Should NOT raise
    result = m.propose_curation(_INGREDIENT, _BrokenClient())
    assert isinstance(result, dict)


def test_propose_curation_strips_markdown_fences():
    """If the model wraps JSON in ```json ... ```, propose_curation should parse it."""
    m = _mod()
    wrapped = "```json\n" + _CURATION_JSON + "\n```"
    client = _FakeClient(wrapped)
    result = m.propose_curation(_INGREDIENT, client)
    assert result["research_score"] == 8


# ---------------------------------------------------------------------------
# Tests: NARRATIVE_SECTIONS constant
# ---------------------------------------------------------------------------

def test_narrative_sections_constant():
    m = _mod()
    assert set(m.NARRATIVE_SECTIONS) == {"what_it_is", "research"}
