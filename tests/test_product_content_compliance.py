"""The structure-function guardrail must be present in every product-content
generator prompt, so generated card/how-it-works/learn-more cannot emit disease
claims even when the page copy or research sources mention conditions."""
import dashboard.product_content as pc


def test_compliance_constant_defined():
    c = pc._COMPLIANCE.lower()
    assert "structure-function" in c
    assert "diagnoses, treats, cures, prevents" in c
    assert "name a disease, condition, or symptom" in c
    assert "never overclaim" in c


def test_guardrail_inherited_by_all_generator_prompts():
    # All three generators build their system prompt from _VOICE, which now
    # carries _COMPLIANCE. Assert it actually reached each prompt.
    for name in ("_CARD_SYSTEM", "_HOW_SYSTEM", "_LEARN_SYSTEM"):
        prompt = getattr(pc, name).lower()
        assert "structure-function" in prompt, f"{name} missing the compliance guardrail"
        assert "treats, cures, prevents" in prompt or "treats, cures" in prompt, \
            f"{name} missing the forbidden-verbs guidance"


def test_voice_includes_compliance():
    assert pc._COMPLIANCE in pc._VOICE
