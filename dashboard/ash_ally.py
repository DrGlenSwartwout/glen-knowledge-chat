# dashboard/ash_ally.py
"""Cross-surface ASH ally memory (SP2b-1).

A thin, fail-open, flag-gated layer over dashboard/ash_map.py that lets the ally
"follow" an identified person across chat surfaces: ally_overlay() reads their
coverage map into a system-prompt block; record_turn() updates it after a reply.
Pure module: no app/Flask import. The app's LOG_DB path and _db_lock are passed in
as arguments (never imported), so this stays unit-testable and avoids a circular
import. Every public function degrades silently on error — a memory failure must
never break a chat.
"""
import os
import sqlite3

from dashboard import ash_map

_TRUE = {"1", "true", "yes"}

FRAME_HEADER = "━━━ WHAT YOU ALREADY KNOW ABOUT THIS PERSON ━━━\n"
FRAME_FOOTER = (
    "\n\nGreet them with continuity and pick up the threads they've opened. Don't re-ask "
    "what they've already shared. Never read this back as a list, never mention "
    "\"dimensions\", a \"map\", or that you track anything — just let it make you feel like "
    "someone who remembers them."
)


def ENABLED() -> bool:
    return os.environ.get("ASH_ALLY_ENABLED", "").strip().lower() in _TRUE


def _is_blank(memory: dict) -> bool:
    if (memory.get("summary") or "").strip():
        return False
    dims = memory.get("dimensions") or {}
    return all((c or {}).get("state", "untouched") == "untouched" for c in dims.values())


def ally_overlay(db_path, subject_email: str) -> str:
    """Framed memory block for a surface's system prompt, or '' (disabled / no email /
    nothing learned yet / any error). Fail-open: never raises."""
    try:
        if not ENABLED() or not (subject_email or "").strip():
            return ""
        with sqlite3.connect(db_path) as cx:
            memory = ash_map.get(cx, subject_email)
        if _is_blank(memory):
            return ""
        return FRAME_HEADER + ash_map.context_block(memory) + FRAME_FOOTER
    except Exception:
        return ""


def record_turn(db_path, lock, subject_email: str, user_text: str, ally_text: str = "") -> None:
    """Update a person's ASH memory from one turn. Designed to be dispatched off the
    request path (a daemon thread). No-op when disabled / no email. Fail-open: never
    raises. Lock-split: the slow Haiku extract runs with NO lock held; only the fast
    sqlite read and write hold `lock`."""
    try:
        if not ENABLED() or not (subject_email or "").strip():
            return
        # (1) locked read of current memory (for extract context)
        with lock:
            with sqlite3.connect(db_path) as cx:
                memory = ash_map.get(cx, subject_email)
        # (2) UNLOCKED slow LLM call
        extracted = ash_map._haiku_extract(memory, user_text, ally_text)
        # (3) locked merge + persist (re-reads under the lock so concurrent same-email
        #     turns converge; merge_turn is forward-only)
        with lock:
            with sqlite3.connect(db_path) as cx:
                ash_map.persist_extract(cx, subject_email, extracted)
    except Exception:
        return
