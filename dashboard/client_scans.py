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
    try:
        cx.execute("ALTER TABLE client_scans ADD COLUMN notified_at TEXT")
    except Exception:
        pass
    cx.commit()


def upsert_scans(cx, email, scans):
    """Upsert a client's scan dates. Returns the list of GENUINELY newly-inserted rows
    ({email, scan_date, scan_id}) — an existing date hitting the UPDATE path is NOT
    returned. Callers that want a touched-count use len() of the return. The new-scan
    email keys off this list so a re-pushed manifest (all rows already present) never
    re-emails, and a flag-flip can't mass-email the historical backlog."""
    e = _norm(email)
    if not e:
        return []
    inserted = []
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
            ins = cx.execute("INSERT OR IGNORE INTO client_scans (email, scan_date, scan_id, synced_at) "
                             "VALUES (?,?,?,?)", (e, d, sid, _now()))
            if ins.rowcount:
                inserted.append({"email": e, "scan_date": d, "scan_id": sid})
    cx.commit()
    return inserted


def scans_for(cx, email):
    rows = cx.execute(
        "SELECT scan_date, scan_id FROM client_scans WHERE email=? ORDER BY scan_date DESC, id DESC",
        (_norm(email),)).fetchall()
    return [{"scan_date": r[0], "scan_id": r[1] or ""} for r in rows]


def unnotified(cx, email=None, limit=500):
    if email:
        rows = cx.execute("SELECT email, scan_date, scan_id FROM client_scans "
                          "WHERE email=? AND notified_at IS NULL ORDER BY scan_date DESC LIMIT ?",
                          (_norm(email), int(limit))).fetchall()
    else:
        rows = cx.execute("SELECT email, scan_date, scan_id FROM client_scans "
                          "WHERE notified_at IS NULL ORDER BY id LIMIT ?", (int(limit),)).fetchall()
    return [{"email": r[0], "scan_date": r[1], "scan_id": r[2] or ""} for r in rows]


def mark_notified(cx, email, scan_date):
    cx.execute("UPDATE client_scans SET notified_at=? WHERE email=? AND scan_date=?",
               (_now(), _norm(email), (scan_date or "").strip()))
    cx.commit()
