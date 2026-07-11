"""Increment 2: verbal-notes + narrative store, the Glen-voice prompt, and a
generate function with an injectable LLM (no live API call in tests)."""
import sqlite3
from dashboard.biofield_narrative import (
    init_notes_tables, get_notes, save_notes, get_narrative, save_narrative,
    build_narrative_prompt, generate_narrative,
    get_video_script, save_video_script, build_video_script_prompt, generate_video_script,
    get_notes_updated, fmt_saved_hst,
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


def test_notes_updated_tracks_last_save(tmp_path):
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_notes_tables(cx)
    # never saved -> no timestamp
    assert get_notes_updated(cx, "10") == ""
    # save returns a UTC 'Z' timestamp that get_notes_updated reads back
    ts = save_notes(cx, "10", "first pass")
    assert ts.endswith("Z")
    assert get_notes_updated(cx, "10") == ts


def test_fmt_saved_hst_converts_utc_to_hst():
    # 22:14 UTC minus 10h = 12:14 PM HST, same calendar day
    assert fmt_saved_hst("2026-07-10T22:14:03Z") == "Jul 10, 2026 · 12:14 PM HST"
    # crosses midnight backward: 05:30 UTC -> 19:30 (7:30 PM) prior day HST
    assert fmt_saved_hst("2026-07-10T05:30:00Z") == "Jul 9, 2026 · 7:30 PM HST"
    # midnight HST reads as 12:00 AM, not 0:00
    assert fmt_saved_hst("2026-07-10T10:00:00Z") == "Jul 10, 2026 · 12:00 AM HST"
    # empty / unparseable -> "" so the caller shows nothing
    assert fmt_saved_hst("") == ""
    assert fmt_saved_hst("not-a-date") == ""


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
    # voice: plain and grounded, no literary flourish
    assert "metaphor" in sys.lower()
    # the chain, top-down, with remedies
    assert usr.index("Night") < usr.index("Acid")
    assert "TMG Powder" in usr and "Sterol Max" in usr
    # the verbal notes are handed to the model
    assert "kidney felt weak; mercury history" in usr
    assert "Lewis Zardo" in usr


def test_video_script_roundtrip(tmp_path):
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_notes_tables(cx)
    assert get_video_script(cx, "10") == ""
    save_video_script(cx, "10", "Aloha Lewis, let me walk you through this.")
    assert get_video_script(cx, "10") == "Aloha Lewis, let me walk you through this."


def test_video_script_prompt_is_short_spoken_and_carries_chain():
    p = build_video_script_prompt(_report(), "mercury history")
    sys, usr = p["system"], p["user"]
    assert "Aloha" in sys
    assert "spoken" in sys.lower() or "out loud" in sys.lower() or "say" in sys.lower()
    assert "short" in sys.lower() or "brief" in sys.lower() or "60" in sys or "90" in sys
    assert "TMG Powder" in usr or "Night" in usr
    assert "mercury history" in usr


def test_generate_video_script_uses_injected_llm():
    seen = {}
    def fake(system, user):
        seen["user"] = user
        return "Aloha Lewis, here's the short version."
    out = generate_video_script(_report(), "mercury history", fake)
    assert out.startswith("Aloha Lewis")
    assert "mercury history" in seen["user"]


def test_generate_uses_injected_llm_and_returns_text():
    seen = {}
    def fake_llm(system, user):
        seen["system"] = system
        seen["user"] = user
        return "Aloha Lewis,\n\nYour body identified..."
    out = generate_narrative(_report(), "mercury history", fake_llm)
    assert out.startswith("Aloha Lewis,")
    assert "mercury history" in seen["user"]
