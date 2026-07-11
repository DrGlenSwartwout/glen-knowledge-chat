"""Slice 1 of Condition Support Programs: sqlite store flagging formulations
that are frequently a good FF match and broadly beneficial (Slice 2 wires this
into `_make_ff_items_for`/`_ff_llm_rank`). Pure sqlite, no Flask.
"""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS broad_benefit (
            slug TEXT PRIMARY KEY,
            added_at TEXT
        )""")


def seed_if_empty(cx, slugs_list):
    """Insert each slug ONLY if the table is currently empty. Idempotent and
    never overwrites operator add/remove edits."""
    (count,) = cx.execute("SELECT COUNT(*) FROM broad_benefit").fetchone()
    if count:
        return
    now = _now()
    for slug in (slugs_list or []):
        cx.execute("INSERT OR IGNORE INTO broad_benefit (slug, added_at) VALUES (?,?)",
                   (slug, now))
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
