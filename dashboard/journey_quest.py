"""Member quest-progress persistence for the 'Find the Hidden Links' journey.

Pure helpers (no Flask): normalize/merge quest state.
Storage: sqlite quest_state table in the app's existing LOG_DB.
No Flask dependency — the caller passes a sqlite3 connection.
"""
import json
import sqlite3

STAGE_KEYS = ["home", "scan", "find", "heal", "give"]
RAILS = ["hunt", "video", "chat"]


def empty_state() -> dict:
    s = {"paths": [], "entered": False}
    for k in STAGE_KEYS:
        s[k] = {"found": False, "done": False}
    return s


def normalize(state) -> dict:
    """Coerce an arbitrary/partial client dict into the canonical shape.

    Drops unknown keys, restricts paths to RAILS, coerces booleans.
    Never raises — malformed input becomes empty_state().
    """
    if not isinstance(state, dict):
        return empty_state()
    result = empty_state()
    result["entered"] = bool(state.get("entered"))
    raw_paths = state.get("paths") or []
    if isinstance(raw_paths, list):
        seen = set()
        for p in raw_paths:
            if p in RAILS and p not in seen:
                result["paths"].append(p)
                seen.add(p)
    for k in STAGE_KEYS:
        raw = state.get(k)
        if isinstance(raw, dict):
            result[k]["found"] = bool(raw.get("found"))
            result[k]["done"] = bool(raw.get("done"))
    return result


def merge_quest(a: dict, b: dict) -> dict:
    """Monotonic merge: progress only ever moves forward.

    For each stage: found/done = OR.
    paths = sorted unique union restricted to RAILS.
    entered = OR.
    """
    result = empty_state()
    result["entered"] = bool(a.get("entered")) or bool(b.get("entered"))
    paths = set()
    for p in (a.get("paths") or []) + (b.get("paths") or []):
        if p in RAILS:
            paths.add(p)
    result["paths"] = sorted(paths)
    for k in STAGE_KEYS:
        af = a.get(k) or {}
        bf = b.get(k) or {}
        result[k]["found"] = bool(af.get("found")) or bool(bf.get("found"))
        result[k]["done"] = bool(af.get("done")) or bool(bf.get("done"))
    return result


# ---------- sqlite storage ----------

def init_quest_store(cx: sqlite3.Connection):
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS quest_state (
          email      TEXT PRIMARY KEY,
          state_json TEXT NOT NULL,
          updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """
    )
    cx.commit()


def load(cx: sqlite3.Connection, email: str) -> dict:
    """Return stored state for email, or empty_state() if none."""
    init_quest_store(cx)
    row = cx.execute(
        "SELECT state_json FROM quest_state WHERE email=?",
        (email.strip().lower(),)
    ).fetchone()
    if not row:
        return empty_state()
    try:
        return normalize(json.loads(row[0]))
    except (ValueError, TypeError):
        return empty_state()


def save(cx: sqlite3.Connection, email: str, state: dict):
    """Upsert the canonical state for email."""
    init_quest_store(cx)
    key = email.strip().lower()
    cx.execute(
        """
        INSERT INTO quest_state (email, state_json, updated_at)
        VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        ON CONFLICT(email) DO UPDATE SET
          state_json = excluded.state_json,
          updated_at = excluded.updated_at
        """,
        (key, json.dumps(state))
    )
    cx.commit()
