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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


def _blank_map() -> dict:
    return {
        k: {"state": "untouched", "opened_excerpt": "",
            "notes": "", "last_touched_at": None}
        for k in DIM_KEYS
    }


import copy as _copy


def merge_turn(memory: dict, updater_output: dict) -> dict:
    """Apply one updater result to a memory, PURELY. Forward-only state ladder,
    set-once excerpt, deduped note accumulation. Returns a new dict."""
    merged = _copy.deepcopy(memory)
    dims = merged.setdefault("dimensions", _blank_map())
    now = _now_iso()

    for key, delta in (updater_output.get("dimensions") or {}).items():
        if key not in DIM_KEYS or not isinstance(delta, dict):
            continue
        cell = dims.setdefault(key, {
            "state": "untouched", "opened_excerpt": "",
            "notes": "", "last_touched_at": None})

        proposed = delta.get("state", "untouched")
        cur_rank = STATE_ORDER.get(cell.get("state", "untouched"), 0)
        prop_rank = STATE_ORDER.get(proposed, 0)
        if prop_rank > cur_rank:
            cell["state"] = proposed

        excerpt = (delta.get("excerpt") or "").strip()
        if excerpt and not cell.get("opened_excerpt"):
            cell["opened_excerpt"] = excerpt

        note = (delta.get("notes") or "").strip()
        if note:
            existing = cell.get("notes", "")
            existing_lines = existing.split("\n") if existing else []
            if note not in existing_lines:
                existing_lines.append(note)
                cell["notes"] = "\n".join(line for line in existing_lines if line)

        cell["last_touched_at"] = now

    new_summary = (updater_output.get("summary") or "").strip()
    if new_summary:
        merged["summary"] = new_summary

    return merged


def init_table(cx) -> None:
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS ash_ally_memory (
          email           TEXT PRIMARY KEY,
          summary         TEXT NOT NULL DEFAULT '',
          dimensions_json TEXT NOT NULL DEFAULT '{}',
          created_at      TEXT NOT NULL,
          updated_at      TEXT NOT NULL
        )
        """
    )
    cx.commit()


def _full_dimensions(stored: dict) -> dict:
    """Backfill any missing of the 12 keys from a blank map so callers see all 12."""
    full = _blank_map()
    for k, v in (stored or {}).items():
        if k in full and isinstance(v, dict):
            full[k].update(v)
    return full


def get(cx, email: str) -> dict:
    init_table(cx)
    em = _norm_email(email)
    cx.row_factory = sqlite3.Row
    row = cx.execute(
        "SELECT summary, dimensions_json, created_at, updated_at "
        "FROM ash_ally_memory WHERE email = ?", (em,)
    ).fetchone()
    if row is None:
        return {"email": em, "summary": "", "dimensions": _blank_map(),
                "created_at": None, "updated_at": None}
    try:
        stored = json.loads(row["dimensions_json"]) or {}
    except (ValueError, TypeError):
        stored = {}
    return {
        "email": em,
        "summary": row["summary"] or "",
        "dimensions": _full_dimensions(stored),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _upsert(cx, email: str, summary: str, dimensions: dict) -> None:
    init_table(cx)
    em = _norm_email(email)
    now = _now_iso()
    existing = cx.execute(
        "SELECT created_at FROM ash_ally_memory WHERE email = ?", (em,)
    ).fetchone()
    created_at = existing[0] if existing else now
    cx.execute(
        "INSERT OR REPLACE INTO ash_ally_memory "
        "(email, summary, dimensions_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (em, summary or "", json.dumps(dimensions or {}), created_at, now),
    )
    cx.commit()
