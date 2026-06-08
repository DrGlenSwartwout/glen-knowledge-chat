"""Per-action state for the dashboard Intelligence cards.

The cards' recommended actions ([HIGH]/[MED]/[LOW] lines) are AI-generated and
regenerated daily, so their "done / not-needed / snoozed" state can't live in the
markdown. This stores it in chat_log.db keyed by (slug, hash-of-action-text), and
filters handled actions out of the markdown at serve time. State is reset whenever
a card regenerates (fresh actions => clean slate).
"""

import hashlib
import os
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

_DB = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))) / "chat_log.db"

# Matches an action line, optionally a list bullet: "[HIGH] do the thing"
_ACTION_RE = re.compile(r"^\s*(?:[-*]\s*)?\[(?:HIGH|MED|LOW)\]\s*(.+?)\s*$", re.IGNORECASE)


def _conn():
    cx = sqlite3.connect(str(_DB), timeout=5)
    cx.execute("""
        CREATE TABLE IF NOT EXISTS briefing_actions (
            slug          TEXT NOT NULL,
            action_hash   TEXT NOT NULL,
            action_text   TEXT,
            state         TEXT NOT NULL,          -- done | dismissed | snoozed
            snooze_until  TEXT DEFAULT '',
            ts            TEXT NOT NULL,
            UNIQUE(slug, action_hash)
        )
    """)
    return cx


def _hash(text):
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def set_state(slug, action_text, state, snooze_days=0):
    if state not in ("done", "dismissed", "snoozed"):
        raise ValueError(f"bad state: {state}")
    now = datetime.now(timezone.utc)
    snooze_until = (now + timedelta(days=int(snooze_days or 0))).isoformat() if state == "snoozed" else ""
    with _conn() as cx:
        cx.execute("""
            INSERT INTO briefing_actions (slug, action_hash, action_text, state, snooze_until, ts)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(slug, action_hash) DO UPDATE SET
              state=excluded.state, snooze_until=excluded.snooze_until,
              action_text=excluded.action_text, ts=excluded.ts
        """, (slug, _hash(action_text), action_text, state, snooze_until, now.isoformat()))
        cx.commit()
    return True


def hidden_hashes(slug):
    """Hashes currently hidden: done/dismissed always; snoozed while snooze_until > now."""
    now_iso = datetime.now(timezone.utc).isoformat()
    out = set()
    try:
        with _conn() as cx:
            for h, state, until in cx.execute(
                "SELECT action_hash, state, snooze_until FROM briefing_actions WHERE slug=?",
                (slug,)):
                if state in ("done", "dismissed"):
                    out.add(h)
                elif state == "snoozed" and (until or "") > now_iso:
                    out.add(h)
    except Exception:
        pass
    return out


def filter_markdown(slug, md):
    """Drop action lines whose normalized text is in a hidden state."""
    if not md:
        return md
    hidden = hidden_hashes(slug)
    if not hidden:
        return md
    kept = []
    for line in md.split("\n"):
        m = _ACTION_RE.match(line)
        if m and _hash(m.group(1)) in hidden:
            continue
        kept.append(line)
    return "\n".join(kept)


def reset_slug(slug):
    """Clear all stored states for a slug (called when its briefing regenerates)."""
    try:
        with _conn() as cx:
            cx.execute("DELETE FROM briefing_actions WHERE slug=?", (slug,))
            cx.commit()
    except Exception:
        pass
