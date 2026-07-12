"""Wishlist store. Pure + LOG_DB-only so it is unit-testable without app.py.
Owner keys: 'email:<lowercased email>' or 'sess:<amg_session>'; email wins."""


def init_wishlist_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS wishlist (
          owner    TEXT NOT NULL,
          slug     TEXT NOT NULL,
          added_at TEXT NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY (owner, slug)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_wishlist_owner ON wishlist(owner)")


def resolve_owner(email, session_id):
    e = (email or "").strip().lower()
    if e:
        return "email:" + e
    if session_id:
        return "sess:" + session_id
    return None


def toggle(cx, owner, slug):
    if not owner or not slug:
        return False
    row = cx.execute("SELECT 1 FROM wishlist WHERE owner=? AND slug=?", (owner, slug)).fetchone()
    if row:
        cx.execute("DELETE FROM wishlist WHERE owner=? AND slug=?", (owner, slug))
        cx.commit()
        return False
    cx.execute("INSERT OR IGNORE INTO wishlist(owner, slug) VALUES (?, ?)", (owner, slug))
    cx.commit()
    return True


def list_for(cx, owner):
    if not owner:
        return []
    return [r[0] for r in cx.execute(
        "SELECT slug FROM wishlist WHERE owner=? ORDER BY rowid DESC", (owner,))]


def merge_wishlist(cx, session_id, email):
    e = (email or "").strip().lower()
    if not session_id or not e:
        return 0
    src, dst = "sess:" + session_id, "email:" + e
    rows = [r[0] for r in cx.execute("SELECT slug FROM wishlist WHERE owner=?", (src,))]
    if not rows:
        return 0
    for slug in rows:
        cx.execute("INSERT OR IGNORE INTO wishlist(owner, slug) VALUES (?, ?)", (dst, slug))
    cx.execute("DELETE FROM wishlist WHERE owner=?", (src,))
    cx.commit()
    return len(rows)


def list_union(cx, email, session_id):
    seen, out = set(), []
    for owner in (resolve_owner(email, None), resolve_owner(None, session_id)):
        if not owner:
            continue
        for slug in list_for(cx, owner):
            if slug not in seen:
                seen.add(slug)
                out.append(slug)
    return out
