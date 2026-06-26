"""Per-scan biofield reports: one evolving row per (email, scan_date).
Source of truth for the biofield analysis; client_portals.content_json is the
legacy fallback when a client has no rows here. See the 2026-06-17 spec."""
import datetime
import json
import sqlite3


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_table(cx) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS portal_biofield_reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email        TEXT,
            scan_date    TEXT,
            scan_id      TEXT,
            content_json TEXT,
            status       TEXT,
            created_at   TEXT,
            updated_at   TEXT,
            UNIQUE(email, scan_date)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_pbr_email ON portal_biofield_reports(email)")
    cx.commit()


def upsert_report(cx, email, scan_date, scan_id, content, status):
    email = (email or "").strip().lower()
    now = _now_iso()
    row = cx.execute(
        "SELECT id FROM portal_biofield_reports WHERE email=? AND scan_date=?",
        (email, scan_date)).fetchone()
    cj = json.dumps(content or {})
    if row:
        cx.execute("UPDATE portal_biofield_reports SET scan_id=?, content_json=?, "
                   "status=?, updated_at=? WHERE id=?",
                   (scan_id, cj, status, now, row[0]))
    else:
        cx.execute("INSERT INTO portal_biofield_reports "
                   "(email, scan_date, scan_id, content_json, status, created_at, updated_at) "
                   "VALUES (?,?,?,?,?,?,?)",
                   (email, scan_date, scan_id, cj, status, now, now))
    cx.commit()


def _row_to_dict(row):
    try:
        content = json.loads(row[3] or "{}")
    except Exception:
        content = {}
    return {"scan_date": row[1], "scan_id": row[2], "content": content, "status": row[4]}


def get_report(cx, email, scan_date):
    email = (email or "").strip().lower()
    row = cx.execute("SELECT id, scan_date, scan_id, content_json, status "
                     "FROM portal_biofield_reports WHERE email=? AND scan_date=?",
                     (email, scan_date)).fetchone()
    return _row_to_dict(row) if row else None


def list_report_dates(cx, email):
    email = (email or "").strip().lower()
    rows = cx.execute("SELECT scan_date FROM portal_biofield_reports WHERE email=? "
                      "ORDER BY scan_date DESC", (email,)).fetchall()
    return [r[0] for r in rows]


def latest_report(cx, email):
    email = (email or "").strip().lower()
    row = cx.execute("SELECT id, scan_date, scan_id, content_json, status "
                     "FROM portal_biofield_reports WHERE email=? "
                     "ORDER BY scan_date DESC LIMIT 1", (email,)).fetchone()
    return _row_to_dict(row) if row else None


def set_report_status(cx, email, scan_date, status):
    email = (email or "").strip().lower()
    cur = cx.execute("UPDATE portal_biofield_reports SET status=?, updated_at=? "
                     "WHERE email=? AND scan_date=?", (status, _now_iso(), email, scan_date))
    cx.commit()
    return cur.rowcount > 0


def is_actionable(scan_date, today):
    """A scan is actionable (CTAs/transitions allowed) within 30 days of `today`.
    Both args are 'YYYY-MM-DD'. Bad/empty scan_date -> False."""
    try:
        sd = datetime.date.fromisoformat(scan_date)
        td = datetime.date.fromisoformat(today)
    except (ValueError, TypeError):
        return False
    return 0 <= (td - sd).days <= 30


def report_pdf_urls(cx, emails):
    """{email_lower: url} for each email whose LATEST confirmed report carries a
    non-empty content.report_pdf.url. Emails without one are omitted. None-raising."""
    wanted = sorted({(e or "").strip().lower() for e in (emails or []) if (e or "").strip()})
    if not wanted:
        return {}
    ph = ",".join("?" * len(wanted))
    rows = cx.execute(
        f"SELECT lower(email), content_json FROM portal_biofield_reports "
        f"WHERE lower(email) IN ({ph}) AND status='confirmed' "
        f"ORDER BY scan_date DESC", wanted).fetchall()
    out = {}
    for em, content_json in rows:
        if em in out:
            continue                      # rows are newest-first; keep the first per email
        try:
            url = ((json.loads(content_json or "{}").get("report_pdf") or {}).get("url") or "").strip()
        except Exception:
            url = ""
        if url:
            out[em] = url
    return out
