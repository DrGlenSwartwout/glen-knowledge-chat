"""Increment 2: verbal-notes + narrative store, the Glen-voice prompt, and a
generate function with an injectable LLM (no live API call in tests)."""
import sqlite3
from dashboard.biofield_narrative import (
    init_notes_tables, get_notes, save_notes, get_narrative, save_narrative,
    build_narrative_prompt, generate_narrative,
)


def _report():
    return {
        "test_id": "10", "client": {"name": "Lewis Zardo", "email": "lz@x.com"},
        "date": "2026-06-01",
        "layers": [
            {"layer": 1, "head": "Night", "most_affected": "Night",
             "remedy": "TMG Powder", "dosage": "1 scoop", "frequency": "daily", "timing": "at night"},
            {"layer": 2, "head": "Acid", "most_affected": "Liver",
             "remedy": "Sterol Max", "dosage": "3 caps", "frequency": "daily", "timing": "with food"},
        ],
        "schedule": {"slots": [], "entries": []},
    }


def test_notes_roundtrip(tmp_path):
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_notes_tables(cx)
    assert get_notes(cx, "10") == ""
    save_notes(cx, "10", "kidney felt weak; mercury history")
    assert get_notes(cx, "10") == "kidney felt weak; mercury history"
    save_notes(cx, "10", "updated note")
    assert get_notes(cx, "10") == "updated note"


def test_narrative_roundtrip(tmp_path):
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_notes_tables(cx)
    assert get_narrative(cx, "10") == ""
    save_narrative(cx, "10", "Aloha Lewis,")
    assert get_narrative(cx, "10") == "Aloha Lewis,"


def test_prompt_carries_voice_rules_layers_and_notes():
    p = build_narrative_prompt(_report(), "kidney felt weak; mercury history")
    sys, usr = p["system"], p["user"]
    # voice rules
    assert "Aloha" in sys
    assert "Dr. Glen & Rae" in sys
    assert "observation" in sys.lower()
    # the chain, top-down, with remedies
    assert usr.index("Night") < usr.index("Acid")
    assert "TMG Powder" in usr and "Sterol Max" in usr
    # the verbal notes are handed to the model
    assert "kidney felt weak; mercury history" in usr
    assert "Lewis Zardo" in usr


def test_generate_uses_injected_llm_and_returns_text():
    seen = {}
    def fake_llm(system, user):
        seen["system"] = system
        seen["user"] = user
        return "Aloha Lewis,\n\nYour body identified..."
    out = generate_narrative(_report(), "mercury history", fake_llm)
    assert out.startswith("Aloha Lewis,")
    assert "mercury history" in seen["user"]
