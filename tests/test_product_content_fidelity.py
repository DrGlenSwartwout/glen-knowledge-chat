"""Ingredient-fidelity guardrail + research-source fidelity filter.

Root cause (found 2026-06-26): the public ATA-Mg `neuro-magnesium` page bled
Magnesium-L-Threonate copy from a legacy same-named formula via two vectors:
  1. specific-formulations page copy mis-titled "Neuro Magnesium" (fixed by a
     Pinecone re-title scrub, not by code), and
  2. the `ingredients` namespace semantic research pull returned L-Threonate
     studies for an ATA-Mg product.

These tests cover the code-side fixes: the prompt-level ingredient-fidelity
guardrail (so the model never mentions an ingredient not in THIS product) and
the research-source filter (so threonate studies never reach the learn_more
prompt for an ATA-Mg product).
"""
import dashboard.product_content as pc


# ── prompt-level ingredient-fidelity guardrail ───────────────────────────────
def test_fidelity_constant_defined():
    c = pc._FIDELITY.lower()
    assert "only" in c
    assert "ingredient" in c
    # must forbid carrying over other forms / similarly named products
    assert "magnesium l-threonate" in c or "other" in c
    assert "not listed" in c or "not in this product" in c


def test_fidelity_inherited_by_all_generator_prompts():
    for name in ("_CARD_SYSTEM", "_HOW_SYSTEM", "_LEARN_SYSTEM"):
        prompt = getattr(pc, name).lower()
        assert "only" in prompt and "ingredient" in prompt, \
            f"{name} missing the ingredient-fidelity guardrail"


def test_voice_includes_fidelity():
    assert pc._FIDELITY in pc._VOICE


# ── research-source fidelity filter ──────────────────────────────────────────
class _FakeMatch:
    def __init__(self, ingredient, url, text=""):
        self.metadata = {"ingredient": ingredient, "url": url, "study_title": "S",
                         "publication": "P", "year": "2020", "text": text}


class _FakeRes:
    def __init__(self, matches):
        self.matches = matches


def _fake_idx(matches):
    class _Idx:
        def query(self, **kw):
            return _FakeRes(matches)
    return _Idx()


def _patch_clients(monkeypatch, matches):
    idx = _fake_idx(matches)
    monkeypatch.setattr(pc, "_clients", lambda: (idx, None, lambda t: [0.0]))


def test_research_drops_nonmatching_ingredient(monkeypatch):
    # ATA-Mg product: a Magnesium L-Threonate study must NOT survive.
    matches = [
        _FakeMatch("Magnesium N-Acetyl-Taurate", "http://study/atamg"),
        _FakeMatch("Magnesium L-Threonate", "http://study/threonate"),
        _FakeMatch("L-Theanine", "http://study/theanine"),
    ]
    _patch_clients(monkeypatch, matches)
    prod_ings = [{"name": "Magnesium N-Acetyl-Taurate (ATA Mg)", "dose": "770 mg"},
                 {"name": "L-Theanine", "dose": "100 mg"}]
    out = pc._research_sources("Neuro Magnesium", ingredients=prod_ings)
    urls = {s["url"] for s in out}
    assert "http://study/atamg" in urls
    assert "http://study/theanine" in urls
    assert "http://study/threonate" not in urls   # the bleed source, dropped


def test_research_matches_via_substring_either_direction(monkeypatch):
    # product lists the full parenthetical name; research carries the short form.
    matches = [_FakeMatch("Pyridoxal-5-Phosphate", "http://study/p5p")]
    _patch_clients(monkeypatch, matches)
    prod_ings = [{"name": "Vitamin B6 (Pyridoxal-5-Phosphate)", "dose": "10 mg"}]
    out = pc._research_sources("X", ingredients=prod_ings)
    assert {s["url"] for s in out} == {"http://study/p5p"}


def test_research_unfiltered_when_no_ingredients(monkeypatch):
    # Back-compat: products with no ingredient list keep the old behavior.
    matches = [_FakeMatch("Magnesium L-Threonate", "http://study/threonate")]
    _patch_clients(monkeypatch, matches)
    out = pc._research_sources("X")
    assert {s["url"] for s in out} == {"http://study/threonate"}
