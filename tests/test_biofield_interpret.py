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
    assert "terrain restore" in sys.lower()   # liquid-class remedies formatting rule
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


def test_interpret_splits_spoken_dose_from_remedy():
    def fake(system, user):
        return json.dumps({"layers": [
            {"layer": 2, "head": "Toxicity", "most_affected": "Toxicity",
             "remedy": "Neuro-Magnesium", "dosage": "one scoop", "frequency": "twice a day",
             "timing": ""}]})
    l = interpret_transcript("toxicity balanced by neuromagnesium one scoop twice a day", fake)["layers"][0]
    assert l["remedy"] == "Neuro-Magnesium"
    assert l["dosage"] == "one scoop" and l["frequency"] == "twice a day"


def test_interpret_empty_transcript_returns_empty():
    out = interpret_transcript("   ", lambda s, u: '{"layers":[]}')
    assert out["layers"] == []


def test_prompt_tells_model_a_layer_can_have_multiple_remedies():
    sys = build_interpret_prompt(TRANSCRIPT)["system"].lower()
    # The grammar must instruct that one causal layer can need more than one remedy.
    assert "remedies" in sys
    assert "more than one" in sys or "multiple" in sys


def test_interpret_keeps_repeated_layer_objects_for_multi_remedy_layer():
    # Model emits one object per remedy, both tagged layer 1 (Kauilani: layer 1 had two).
    def fake(system, user):
        return json.dumps({"layers": [
            {"layer": 1, "head": "Large Intestine Meridian",
             "most_affected": "Large Intestine Meridian", "remedy": "Microbiome"},
            {"layer": 1, "head": "Large Intestine Meridian",
             "most_affected": "Large Intestine Meridian", "remedy": "Cistus Synergy"},
            {"layer": 2, "head": "Toxicity", "most_affected": "Toxicity",
             "remedy": "Neuro-Magnesium"}]})
    out = interpret_transcript(TRANSCRIPT, fake)
    assert [(l["layer"], l["remedy"]) for l in out["layers"]] == [
        (1, "Microbiome"), (1, "Cistus Synergy"), (2, "Neuro-Magnesium")]


def test_interpret_expands_remedies_array_into_one_entry_per_remedy():
    # Model puts multiple remedies for a layer in a `remedies` array, each its own dose.
    def fake(system, user):
        return json.dumps({"layers": [
            {"layer": 1, "head": "Large Intestine Meridian",
             "most_affected": "Large Intestine Meridian",
             "remedies": [
                 {"remedy": "Microbiome", "dosage": "1 cap", "frequency": "twice a day", "timing": ""},
                 {"remedy": "Cistus Synergy", "dosage": "10 drops", "frequency": "3x a day",
                  "timing": "before food"}]},
            {"layer": 2, "head": "Toxicity", "most_affected": "Toxicity", "remedy": "Neuro-Magnesium"}]})
    out = interpret_transcript(TRANSCRIPT, fake)
    assert [(l["layer"], l["remedy"]) for l in out["layers"]] == [
        (1, "Microbiome"), (1, "Cistus Synergy"), (2, "Neuro-Magnesium")]
    # Per-remedy dose is preserved through the expansion.
    assert out["layers"][1]["dosage"] == "10 drops"
    assert out["layers"][1]["timing"] == "before food"
    # head/most_affected are carried onto each expanded remedy.
    assert out["layers"][0]["head"] == "Large Intestine Meridian"


def test_interpret_expands_remedies_array_of_plain_strings():
    def fake(system, user):
        return json.dumps({"layers": [
            {"layer": 1, "head": "Acid", "most_affected": "Liver",
             "remedies": ["Sterol Max", "Bile Flow"]}]})
    out = interpret_transcript("acid balanced by sterol max and bile flow", fake)
    assert [(l["layer"], l["remedy"]) for l in out["layers"]] == [(1, "Sterol Max"), (1, "Bile Flow")]
