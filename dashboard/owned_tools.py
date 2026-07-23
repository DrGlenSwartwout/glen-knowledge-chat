"""Client-maintained external tools inventory. A client can self-report
devices/tools they already own — whether or not the item maps to one of our
products (``slug`` set) or is purely external (``slug`` NULL). Pure +
LOG_DB-only so it is unit-testable without app.py, modeled on wishlist.py /
supplement_reviews.py. One row per (email, tool_key)."""
import re


def _norm(e):
    return (e or "").strip().lower()


def _key(name, brand):
    """Stable dedupe key: lowercased, whitespace-collapsed name|brand.
    Same normalization idiom as supplement_reviews._key."""
    raw = "%s|%s" % ((name or "").strip().lower(), (brand or "").strip().lower())
    return re.sub(r"\s+", " ", raw)


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS owned_tools (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            email    TEXT NOT NULL,
            name     TEXT NOT NULL,
            brand    TEXT,
            slug     TEXT,
            source   TEXT,
            tool_key TEXT NOT NULL,
            added_at TEXT,
            UNIQUE(email, tool_key)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_owned_tools_email ON owned_tools(email)")
    cx.commit()


def add(cx, email, name, brand="", slug=None, source="external"):
    e = _norm(email)
    n = (name or "").strip()
    if not e or not n:
        return {"created": False, "id": None}
    key = _key(n, brand)
    cur = cx.execute(
        "INSERT OR IGNORE INTO owned_tools "
        "(email, name, brand, slug, source, tool_key, added_at) "
        "VALUES (?,?,?,?,?,?, datetime('now'))",
        (e, n, (brand or "").strip(), slug, source, key))
    cx.commit()
    if cur.rowcount:
        return {"created": True, "id": cur.lastrowid}
    row = cx.execute("SELECT id FROM owned_tools WHERE email=? AND tool_key=?", (e, key)).fetchone()
    return {"created": False, "id": row[0] if row else None}


def list_for(cx, email):
    e = _norm(email)
    if not e:
        return []
    rows = cx.execute(
        "SELECT id, name, brand, slug, source, added_at FROM owned_tools "
        "WHERE email=? ORDER BY id", (e,)).fetchall()
    return [dict(r) for r in rows]


def remove(cx, email, tool_id):
    e = _norm(email)
    if not e or not tool_id:
        return {"removed": False}
    cur = cx.execute("DELETE FROM owned_tools WHERE email=? AND id=?", (e, tool_id))
    cx.commit()
    return {"removed": cur.rowcount > 0}


def owned_slugs(cx, email):
    e = _norm(email)
    if not e:
        return set()
    return {r[0] for r in cx.execute(
        "SELECT slug FROM owned_tools WHERE email=? AND slug IS NOT NULL", (e,))}
