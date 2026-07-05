"""Community signal layer (Layer B): reactions on content + private like/block
on topics and people. Pure sqlite; no app-layer imports. These signals feed
Layer C's curation. Reaction COUNTS are aggregate; who reacted is never exposed
by this module. Like/block signals are per-member private."""

REACTIONS = ["helpful", "inspiring", "this_is_me"]
TARGET_TYPES = ["topic", "person"]
SIGNALS = ["like", "block"]

_DDL = """
CREATE TABLE IF NOT EXISTS community_reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    content_id INTEGER NOT NULL,
    reaction TEXT NOT NULL,
    created_at TEXT,
    UNIQUE(email, content_id, reaction)
);
CREATE INDEX IF NOT EXISTS ix_reactions_content ON community_reactions(content_id);
CREATE TABLE IF NOT EXISTS community_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_key TEXT NOT NULL,
    signal TEXT NOT NULL,
    created_at TEXT,
    UNIQUE(email, target_type, target_key)
);
CREATE INDEX IF NOT EXISTS ix_signals_email ON community_signals(email);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_signal_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def toggle_reaction(cx, email, content_id, reaction):
    """Add the reaction if absent, remove it if present. Returns True if now on."""
    email = _lc(email)
    row = cx.execute("SELECT id FROM community_reactions WHERE email=? AND content_id=? "
                     "AND reaction=?", (email, content_id, reaction)).fetchone()
    if row:
        cx.execute("DELETE FROM community_reactions WHERE id=?", (row[0],))
        cx.commit()
        return False
    cx.execute("INSERT INTO community_reactions (email,content_id,reaction,created_at) "
               "VALUES (?,?,?,?)", (email, content_id, reaction, _now()))
    cx.commit()
    return True


def reaction_counts(cx, content_id):
    """Aggregate counts per reaction for one content item. No emails, ever."""
    rows = cx.execute("SELECT reaction, COUNT(*) c FROM community_reactions "
                      "WHERE content_id=? GROUP BY reaction", (content_id,)).fetchall()
    return {r["reaction"]: r["c"] for r in rows}


def my_reactions(cx, email, content_id):
    rows = cx.execute("SELECT reaction FROM community_reactions WHERE email=? AND content_id=? "
                      "ORDER BY reaction", (_lc(email), content_id)).fetchall()
    return [r["reaction"] for r in rows]


def set_signal(cx, email, target_type, target_key, signal):
    """Upsert a like/block on a target. One row per (email, target_type, target_key)."""
    cx.execute(
        "INSERT INTO community_signals (email,target_type,target_key,signal,created_at) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT(email,target_type,target_key) DO UPDATE SET signal=excluded.signal",
        (_lc(email), target_type, target_key, signal, _now()))
    cx.commit()


def clear_signal(cx, email, target_type, target_key):
    cx.execute("DELETE FROM community_signals WHERE email=? AND target_type=? AND target_key=?",
               (_lc(email), target_type, target_key))
    cx.commit()


def my_signals(cx, email):
    rows = cx.execute("SELECT target_type, target_key, signal FROM community_signals "
                      "WHERE email=? ORDER BY created_at", (_lc(email),)).fetchall()
    out = {"likes": [], "blocks": []}
    for r in rows:
        entry = {"target_type": r["target_type"], "target_key": r["target_key"]}
        out["likes" if r["signal"] == "like" else "blocks"].append(entry)
    return out
