"""Request gate for purity ratings. A client may request a rating on paid
membership OR an explicit access grant. Pure sqlite; caller passes cx.
Default is CLOSED for non-paid clients (unlike the free product review, which
defaults open) -- purity ratings are a paid/explicit-request perk."""


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS purity_ratings_access (
        email TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 0,
        set_by TEXT, updated_at TEXT)""")
    cx.commit()


def can_request(cx, email, membership_category):
    if (membership_category or "").strip().lower() == "full":
        return True
    e = _norm(email)
    if not e:
        return False
    row = cx.execute("SELECT enabled FROM purity_ratings_access WHERE email=?", (e,)).fetchone()
    return bool(row and row[0])


def set_access(cx, email, enabled, set_by):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    e = _norm(email)
    row = cx.execute("SELECT email FROM purity_ratings_access WHERE email=?", (e,)).fetchone()
    if row:
        cx.execute("UPDATE purity_ratings_access SET enabled=?, set_by=?, updated_at=? WHERE email=?",
                   (1 if enabled else 0, set_by, now, e))
    else:
        cx.execute("INSERT INTO purity_ratings_access (email, enabled, set_by, updated_at) "
                   "VALUES (?,?,?,?)", (e, 1 if enabled else 0, set_by, now))
    cx.commit()
