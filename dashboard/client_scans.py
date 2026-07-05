"""Synced manifest of every client's E4L scan dates, for the portal Scan-history list.
Populated from the local e4l.db by the e4l-scan-manifest-push sync (prod can't read e4l.db).
One row per (email, scan_date). LOG_DB (SQLite)."""
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(e):
    return (e or "").strip().lower()


def init_client_scans_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS client_scans (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT NOT NULL,
            scan_date  TEXT NOT NULL,
            scan_id    TEXT,
            synced_at  TEXT,
            UNIQUE(email, scan_date)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_cs_email ON client_scans(email)")
    cx.commit()


def upsert_scans(cx, email, scans):
    e = _norm(email)
    if not e:
        return 0
    n = 0
    for s in scans or []:
        if not isinstance(s, dict):
            continue
        d = (s.get("scan_date") or "").strip()
        if not d:
            continue
        sid = str(s.get("scan_id") or "")
        cur = cx.execute("UPDATE client_scans SET scan_id=?, synced_at=? WHERE email=? AND scan_date=?",
                         (sid, _now(), e, d))
        if cur.rowcount == 0:
            cx.execute("INSERT OR IGNORE INTO client_scans (email, scan_date, scan_id, synced_at) "
                       "VALUES (?,?,?,?)", (e, d, sid, _now()))
        n += 1
    cx.commit()
    return n


def scans_for(cx, email):
    rows = cx.execute(
        "SELECT scan_date, scan_id FROM client_scans WHERE email=? ORDER BY scan_date DESC, id DESC",
        (_norm(email),)).fetchall()
    return [{"scan_date": r[0], "scan_id": r[1] or ""} for r in rows]
