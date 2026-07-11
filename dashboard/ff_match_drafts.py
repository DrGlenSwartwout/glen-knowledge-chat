"""Per-(email, scan_date) cache of generated FF matches + review status.
Generate-once: get_or_create never regenerates an existing row."""
import json
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS ff_match_drafts (
            email TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            items_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'draft',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            published_at TEXT,
            PRIMARY KEY (email, scan_date)
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_ffmd_status ON ff_match_drafts(status)")


def _row(r):
    if r is None:
        return None
    return {"email": r["email"], "scan_date": r["scan_date"],
            "items": json.loads(r["items_json"] or "[]"), "status": r["status"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
            "published_at": r["published_at"]}


def get(cx, email, scan_date):
    r = cx.execute("SELECT * FROM ff_match_drafts WHERE email=? AND scan_date=?",
                   (email.lower(), scan_date)).fetchone()
    return _row(r)


def get_or_create(cx, email, scan_date, make_items):
    email = email.lower()
    existing = get(cx, email, scan_date)
    if existing is not None:
        return existing
    items = make_items() or []
    now = _now()
    cx.execute("INSERT INTO ff_match_drafts "
               "(email, scan_date, items_json, status, created_at, updated_at) "
               "VALUES (?,?,?,?,?,?)",
               (email, scan_date, json.dumps(items), "draft", now, now))
    cx.commit()
    return get(cx, email, scan_date)


def set_items(cx, email, scan_date, items):
    cx.execute("UPDATE ff_match_drafts SET items_json=?, updated_at=? WHERE email=? AND scan_date=?",
               (json.dumps(items or []), _now(), email.lower(), scan_date))
    cx.commit()


def publish(cx, email, scan_date):
    now = _now()
    cur = cx.execute("UPDATE ff_match_drafts SET status='published', published_at=?, updated_at=? "
                     "WHERE email=? AND scan_date=?", (now, now, email.lower(), scan_date))
    cx.commit()
    return cur.rowcount == 1


def list_by_status(cx, status=None, limit=200):
    if status:
        rows = cx.execute("SELECT * FROM ff_match_drafts WHERE status=? "
                          "ORDER BY updated_at DESC LIMIT ?", (status, limit)).fetchall()
    else:
        rows = cx.execute("SELECT * FROM ff_match_drafts ORDER BY updated_at DESC LIMIT ?",
                          (limit,)).fetchall()
    return [_row(r) for r in rows]
