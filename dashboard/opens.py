"""Read-receipt opens: record when a client explicitly opens (expand-clicks) a
report or an invoice. One row per (kind, key). LOG_DB (SQLite). The 5s debounce
keeps a double-click from inflating open_count; first_opened is set once."""
import datetime

_DEBOUNCE_SECONDS = 5
_FMT = "%Y-%m-%d %H:%M:%S"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(_FMT)


def report_key(email, scan_date):
    return f"{(email or '').strip().lower()}|{(scan_date or '').strip()}"


def invoice_key(token):
    return (token or "").strip()


def init_opens_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS portal_opens (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            kind         TEXT NOT NULL,
            key          TEXT NOT NULL,
            first_opened TEXT,
            last_opened  TEXT,
            open_count   INTEGER NOT NULL DEFAULT 0,
            UNIQUE(kind, key)
        )
    """)
    cx.commit()


def _secs(a, b):
    try:
        return abs((datetime.datetime.strptime(b, _FMT) - datetime.datetime.strptime(a, _FMT)).total_seconds())
    except Exception:
        return 10 ** 9  # unparseable → treat as far apart (bump)


def record_open(cx, kind, key, *, now=None):
    now = now or _now()
    row = cx.execute("SELECT first_opened, last_opened, open_count FROM portal_opens WHERE kind=? AND key=?",
                     (kind, key)).fetchone()
    if row is None:
        cx.execute("INSERT INTO portal_opens (kind, key, first_opened, last_opened, open_count) VALUES (?,?,?,?,1)",
                   (kind, key, now, now))
        cx.commit()
        return {"first_opened": now, "last_opened": now, "open_count": 1}
    first, last, count = row
    new_count = count + 1 if _secs(last, now) >= _DEBOUNCE_SECONDS else count
    cx.execute("UPDATE portal_opens SET last_opened=?, open_count=? WHERE kind=? AND key=?",
               (now, new_count, kind, key))
    cx.commit()
    return {"first_opened": first, "last_opened": now, "open_count": new_count}


def get_open(cx, kind, key):
    row = cx.execute("SELECT first_opened, last_opened, open_count FROM portal_opens WHERE kind=? AND key=?",
                     (kind, key)).fetchone()
    return None if row is None else {"first_opened": row[0], "last_opened": row[1], "open_count": row[2]}


def opens_for(cx, kind, keys):
    out = {}
    for k in keys:
        r = get_open(cx, kind, k)
        if r:
            out[k] = r
    return out


def clear_open(cx, kind, key):
    cx.execute("DELETE FROM portal_opens WHERE kind=? AND key=?", (kind, key))
    cx.commit()
