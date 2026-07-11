"""Slice 1 of Condition Support Programs: sqlite store flagging formulations
that are frequently a good FF match and broadly beneficial (Slice 2 wires this
into `_make_ff_items_for`/`_ff_llm_rank`). Pure sqlite, no Flask.
"""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


_SEED_NAME = "broad_benefit"


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS broad_benefit (
            slug TEXT PRIMARY KEY,
            added_at TEXT
        )""")
    _ensure_seed_state_table(cx)


def _ensure_seed_state_table(cx):
    # Shared-shape marker table (also created by dashboard/condition_programs.py)
    # tracking which stores have been seeded — ONCE — regardless of their
    # current row count. A row-count check ("seed when COUNT(*)==0") would
    # mistake an operator's intentional "remove everything" for "never
    # seeded" and silently resurrect the full seed list on the next request.
    cx.execute("""
        CREATE TABLE IF NOT EXISTS _seed_state (
            name TEXT PRIMARY KEY,
            seeded_at TEXT
        )""")


def seed_if_empty(cx, slugs_list):
    """Insert each slug exactly ONCE ever, tracked by a persisted
    `_seed_state` marker (not the table's current row count). Idempotent
    (safe to call on every request) and never re-seeds after the first
    attempt — even if an operator later removes every slug via `remove()`,
    that empty state is never mistaken for "not yet seeded" and resurrected.

    On that first-ever attempt, seeding still only inserts rows if the table
    is currently empty (preserves the original guard against clobbering rows
    that were already present some other way) — but the marker is recorded
    either way, so this function never runs its insert logic more than once
    per store."""
    _ensure_seed_state_table(cx)
    already = cx.execute("SELECT 1 FROM _seed_state WHERE name=?",
                          (_SEED_NAME,)).fetchone()
    if already:
        return
    now = _now()
    (count,) = cx.execute("SELECT COUNT(*) FROM broad_benefit").fetchone()
    if count == 0:
        for slug in (slugs_list or []):
            cx.execute("INSERT OR IGNORE INTO broad_benefit (slug, added_at) VALUES (?,?)",
                       (slug, now))
    cx.execute("INSERT OR IGNORE INTO _seed_state (name, seeded_at) VALUES (?,?)",
               (_SEED_NAME, now))
    cx.commit()


def is_broad(cx, slug):
    r = cx.execute("SELECT 1 FROM broad_benefit WHERE slug=?", (slug,)).fetchone()
    return r is not None


def all_slugs(cx):
    rows = cx.execute("SELECT slug FROM broad_benefit ORDER BY added_at").fetchall()
    return [r[0] for r in rows]


def add(cx, slug):
    cx.execute("INSERT OR IGNORE INTO broad_benefit (slug, added_at) VALUES (?,?)",
               (slug, _now()))
    cx.commit()


def remove(cx, slug):
    cx.execute("DELETE FROM broad_benefit WHERE slug=?", (slug,))
    cx.commit()
