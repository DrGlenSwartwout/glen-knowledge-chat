# dashboard/ash_map.py
"""Email-keyed ASH coverage map + ally memory (SP2a).

A durable per-person record of which of the 12 ASH dimensions a conversation
has touched, plus a rolling "who they are" summary and verbatim opening
excerpts. Pure module: all DB functions take a caller-supplied sqlite3
connection (no app/Flask import), mirroring dashboard/journal_store.py. The
per-turn updater (_haiku_extract) mirrors journal_blueprint._haiku_analyze's
forced-tool-use structured output, but never raises — it runs fire-and-forget
after an ally reply, so any failure degrades to "learned nothing this turn".
"""
import json
import os
import sqlite3
from datetime import datetime, timezone

import requests  # module-level so tests can monkeypatch ash_map.requests.post

ANTHROPIC_MESSAGES = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# The 12 ASH dimensions — canonical keys, display names, and one-line meanings
# the updater prompt renders so Haiku can map turn content -> dimensions.
ASH_DIMENSIONS = [
    {"key": "body", "name": "Body / States of Matter",
     "meaning": "the physical body's substance, density, structure"},
    {"key": "mind", "name": "Mind / 5 C's",
     "meaning": "mental focus, emotional patterns, how they connect and communicate"},
    {"key": "spirit", "name": "Spirit / 5 Elements",
     "meaning": "meaning, purpose, emotional-elemental balance"},
    {"key": "inheritance", "name": "Inheritance / 5 Generations",
     "meaning": "family, genetic, lineage health patterns"},
    {"key": "personal_history", "name": "Personal History / 5 Penetration",
     "meaning": "their own health history and how deep issues have gone"},
    {"key": "epigenetics", "name": "Epigenetics / 5 Infoceuticals",
     "meaning": "bioenergetic / informational regulation (terrain, organs, meridians, systems)"},
    {"key": "symptoms", "name": "Symptoms / 5 Cardinal Signs",
     "meaning": "active symptoms: pain, heat, swelling, redness, loss of function"},
    {"key": "terrain", "name": "Terrain / 5 R's",
     "meaning": "the body's vitality and capacity to heal"},
    {"key": "diagnosis", "name": "Diagnosis / 5 Pathology Types",
     "meaning": "diagnosed conditions or tissue changes"},
    {"key": "treatment", "name": "Treatment / 5 Therapy Levels",
     "meaning": "treatments they use and how invasive vs. supportive"},
    {"key": "regulation", "name": "Regulation / 5 Levels",
     "meaning": "how the body responds when they try to heal"},
    {"key": "prognosis", "name": "Prognosis / 5 Stages",
     "meaning": "seriousness or trajectory of their main concern"},
]
DIM_KEYS = [d["key"] for d in ASH_DIMENSIONS]

STATE_ORDER = {"untouched": 0, "opened": 1, "explored": 2, "deep": 3}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


def _blank_map() -> dict:
    return {
        k: {"state": "untouched", "opened_excerpt": "",
            "notes": "", "last_touched_at": None}
        for k in DIM_KEYS
    }
