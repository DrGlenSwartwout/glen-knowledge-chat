import sys
from pathlib import Path

import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_copy
        return topic_copy
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_copy not importable: {e}")


class _FakeContent:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _FakeMsg:
    def __init__(self, text):
        self.content = [_FakeContent(text)]


class _FakeClient:
    def __init__(self, text):
        self._text = text
        self.messages = self

    def create(self, **kw):
        return _FakeMsg(self._text)


def test_build_section_prompt_carries_compliance():
    tc = _mod()
    system, user = tc.build_section_prompt("overview", {"name": "Low Energy", "kind": "symptom"})
    assert "diagnose, treat, cure" in system.lower()
    assert "Low Energy" in user


def test_validate_links_drops_unknown_slugs():
    tc = _mod()
    cleaned = tc.validate_links(
        {"ingredients": ["folate", "made-up"], "products": ["nope"], "topics": ["detox"]},
        ingredient_slugs={"folate": "Folate"},
        product_slugs={"neuro-magnesium": "Neuro Magnesium"},
        topic_slugs={"detox": "Detox"},
    )
    assert cleaned["ingredients"] == [{"slug": "folate", "name": "Folate"}]
    assert cleaned["products"] == []
    assert cleaned["topics"] == [{"slug": "detox", "name": "Detox"}]


def test_local_claim_flags_catches_disease_claim():
    tc = _mod()
    flags = tc.local_claim_flags({"overview": "This protocol cures cancer and treats diabetes."})
    phrases = " ".join(f["phrase"] for f in flags).lower()
    assert "cure" in phrases or "treat" in phrases
    assert flags  # non-empty


def test_local_claim_flags_clean_copy_passes():
    tc = _mod()
    flags = tc.local_claim_flags({"overview": "People exploring low energy often look into sleep and minerals."})
    assert flags == []


def test_compliance_scan_blocks_planted_claim_without_calling_model():
    tc = _mod()
    # local denylist trips first; model must NOT be consulted
    client = _FakeClient('{"passed": true, "flags": []}')
    res = tc.compliance_scan({"overview": "It cures cancer."}, client)
    assert res["passed"] is False
    assert res["flags"]


def test_compliance_scan_passes_clean_copy():
    tc = _mod()
    client = _FakeClient('{"passed": true, "flags": []}')
    res = tc.compliance_scan({"overview": "Supports healthy energy. People often explore minerals."}, client)
    assert res["passed"] is True
    assert res["flags"] == []


def test_compliance_scan_fails_closed_on_client_error():
    tc = _mod()

    class _Boom:
        messages = None

        def create(self, **kw):
            raise RuntimeError("api down")

    boom = _Boom()
    boom.messages = boom
    res = tc.compliance_scan({"overview": "Supports healthy energy."}, boom)
    assert res["passed"] is False


def test_propose_curation_safe_default_on_bad_json():
    tc = _mod()
    client = _FakeClient("not json")
    out = tc.propose_curation({"name": "Detox", "kind": "function"}, client)
    assert out["links"] == {"ingredients": [], "products": [], "topics": []}
    assert "title" in out and "meta_description" in out


# --- disease-anchored denylist (over-blocking fix) ---

def test_benign_treatment_words_pass():
    tc = _mod()
    benign = [
        "Our municipal water treatment removes many additives.",
        "It uses reverse osmosis filtration at home.",
        "Good nutrition helps prevent deficiency over time.",
        "A relaxing spa treatment can feel restorative.",
        "Talk with your provider about a sensible treatment plan.",
    ]
    for text in benign:
        assert tc.local_claim_flags({"overview": text}) == [], text


def test_real_disease_claims_still_flag():
    tc = _mod()
    claims = [
        "this protocol treats diabetes",
        "it cures cancer naturally",
        "this reverses heart disease",
        "proven to prevent Alzheimer's",
        "heals your disease for good",
    ]
    for text in claims:
        assert tc.local_claim_flags({"overview": text}), text


def test_generic_condition_anchor_flags():
    tc = _mod()
    assert tc.local_claim_flags({"overview": "cure any condition fast"})
    assert tc.local_claim_flags({"overview": "treats this disorder"})


def test_diagnose_and_guarantee_still_flag_unanchored():
    tc = _mod()
    assert tc.local_claim_flags({"overview": "we diagnose the root cause"})
    assert tc.local_claim_flags({"overview": "results are guaranteed"})


def test_existing_planted_claim_still_flags():
    # must preserve the pre-existing test_local_claim_flags_catches_disease_claim behavior
    tc = _mod()
    flags = tc.local_claim_flags({"overview": "This protocol cures cancer and treats diabetes."})
    assert flags


def test_existing_clean_copy_still_clean():
    tc = _mod()
    flags = tc.local_claim_flags(
        {"overview": "People exploring low energy often look into sleep and minerals."})
    assert flags == []
