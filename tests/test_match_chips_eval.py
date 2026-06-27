"""Eval: ⟦CHIPS⟧ directive emission rates for /begin/match/chat system prompt.

Samples Haiku N times per scenario, parses with parse_chips, and asserts
pass-rate thresholds (not single-shot). Skip if ANTHROPIC_API_KEY is absent.
"""
import importlib
import os
import pytest
import anthropic

SAMPLES = 5
THRESHOLD = 3  # must pass in at least THRESHOLD/SAMPLES

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    os.environ.setdefault("DATA_DIR", str(tmp))
    try:
        import app
        importlib.reload(app)
    except Exception as exc:
        pytest.skip(f"app import failed: {exc}")
    return app


def _sample_chips(system: str, messages: list, n: int):
    """Call Haiku n times with the given system+messages; return list of (clean, chips) tuples."""
    from dashboard.chat_cta import parse_chips
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = []
    for i in range(n):
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=900,
            system=system,
            messages=messages,
        )
        raw = msg.content[0].text
        clean, chips = parse_chips(raw)
        print(f"  sample {i+1}: chips={chips!r}")
        results.append((clean, chips))
    return results


def test_scenario_a_discrete_question(app_module):
    """Scenario A: history that should elicit a yes/no or 2-choice clarifying question.
    Expects: >= THRESHOLD/SAMPLES have 2-4 chips, each <= 4 words, sentinel absent from clean."""
    from dashboard.chat_cta import CHIPS_SENTINEL

    system = app_module._REMEDY_MATCH_SYSTEM
    # Provide enough context that the model should ask a clarifying yes/no or choice question
    # (e.g. confirm thyroid direction) rather than an open-ended first question.
    messages = [
        {"role": "user",
         "content": "I have thyroid issues and fatigue. Not sure if it's overactive or underactive."},
        {"role": "assistant",
         "content": "Thank you for sharing that. To point you toward the right remedy, "
                    "I need to know more about your thyroid. Do you run hot or cold, "
                    "feel anxious or sluggish?"},
        {"role": "user",
         "content": "Mostly cold and sluggish."},
    ]

    print("\n--- Scenario A: thyroid direction (discrete choice) ---")
    results = _sample_chips(system, messages, SAMPLES)

    passes = 0
    for clean, chips in results:
        has_chips = 2 <= len(chips) <= 4
        short_words = all(len(c.split()) <= 4 for c in chips)
        no_sentinel = CHIPS_SENTINEL not in clean
        if has_chips and short_words and no_sentinel:
            passes += 1

    print(f"  passes: {passes}/{SAMPLES} (need {THRESHOLD})")
    assert passes >= THRESHOLD, (
        f"Scenario A: only {passes}/{SAMPLES} samples had 2-4 valid chips. "
        f"Results: {[(c, ch) for c, ch in results]}"
    )


def test_scenario_b_open_ended_first_turn(app_module):
    """Scenario B: open-ended first turn ('hi') → model should ask an open question.
    Expects: >= THRESHOLD/SAMPLES have chips == [] (sentinel omitted)."""
    system = app_module._REMEDY_MATCH_SYSTEM
    messages = [{"role": "user", "content": "hi"}]

    print("\n--- Scenario B: open-ended first turn ('hi') ---")
    results = _sample_chips(system, messages, SAMPLES)

    passes = sum(1 for _, chips in results if chips == [])
    print(f"  passes: {passes}/{SAMPLES} (need {THRESHOLD})")
    assert passes >= THRESHOLD, (
        f"Scenario B: only {passes}/{SAMPLES} samples had chips == []. "
        f"Results: {[ch for _, ch in results]}"
    )
