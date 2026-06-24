"""Increment 4c Phase 2: turn a spoken testing transcript into structured causal-chain
rows, using Glen's mapped grammar. LLM injected so tests run offline."""
import json
from dashboard.biofield_interpret import interpret_transcript, build_interpret_prompt


TRANSCRIPT = (
    "This is a biofield analysis. First finding is BSI 21 times 1 to 1 times 1, phase 2. "
    "The phase 2 is toxicity and the location of the 21 is toxicity. "
    "Large intestine meridian is the head and tail of the first causal chain. "
    "It's balanced by microbiome and that also balances phase 2. "
    "Then the second layer is toxicity, head and tail, balances with neuromagnesium. That's it.")


def test_prompt_encodes_grammar_and_carries_transcript():
    p = build_interpret_prompt(TRANSCRIPT)
    sys, usr = p["system"], p["user"]
    assert "head and tail" in sys.lower()
    assert "balanced by" in sys.lower() or "balances with" in sys.lower()
    assert "json" in sys.lower()
    assert "large intestine meridian" in usr.lower()


def test_interpret_returns_layers_from_grammar():
    def fake(system, user):
        return json.dumps({
            "header": "BSI 21x1 / 1x1; phase 2 = toxicity; location of 21 = toxicity",
            "layers": [
                {"layer": 1, "head": "Large Intestine Meridian",
                 "most_affected": "Large Intestine Meridian", "remedy": "Microbiome"},
                {"layer": 2, "head": "Toxicity", "most_affected": "Toxicity",
                 "remedy": "Neuro-Magnesium"}]})
    out = interpret_transcript(TRANSCRIPT, fake)
    assert out["header"].startswith("BSI")
    assert [(l["layer"], l["remedy"]) for l in out["layers"]] == [(1, "Microbiome"), (2, "Neuro-Magnesium")]
    assert out["layers"][0]["head"] == out["layers"][0]["most_affected"]  # head and tail


def test_interpret_tolerates_messy_llm_output():
    def fake(system, user):  # model wraps JSON in prose/fences
        return "Sure:\n```json\n" + json.dumps({"layers": [
            {"layer": 1, "head": "Acid", "most_affected": "Liver", "remedy": "Sterol Max"}]}) + "\n```"
    out = interpret_transcript("acid balanced by sterol max", fake)
    assert out["layers"][0]["remedy"] == "Sterol Max"


def test_interpret_empty_transcript_returns_empty():
    out = interpret_transcript("   ", lambda s, u: '{"layers":[]}')
    assert out["layers"] == []
