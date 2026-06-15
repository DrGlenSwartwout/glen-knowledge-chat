"""Server-side mirror of e4l scan freshness (email -> latest scan date), pushed from
the local e4l ingestion, so the Biofield gate can auto-verify a fresh voice scan."""

def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS scan_freshness (
        email TEXT PRIMARY KEY, last_scan_date TEXT, updated_at TEXT)""")
    cx.commit()

def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def upsert(cx, rows):
    """rows: [{email, last_scan_date}]. Keeps the NEWEST date per email (ISO dates sort lexically)."""
    for r in rows or []:
        email = (r.get("email") or "").strip().lower()
        d = (r.get("last_scan_date") or "").strip()
        if not email or not d:
            continue
        cx.execute("""INSERT INTO scan_freshness (email, last_scan_date, updated_at)
                      VALUES (?,?,?)
                      ON CONFLICT(email) DO UPDATE SET
                        last_scan_date=MAX(scan_freshness.last_scan_date, excluded.last_scan_date),
                        updated_at=excluded.updated_at""",
                   (email, d, _now()))
    cx.commit()

def latest_scan_date(cx, email):
    row = cx.execute("SELECT last_scan_date FROM scan_freshness WHERE email=lower(?)",
                     (str(email or "").strip(),)).fetchone()
    return row[0] if row else None

def is_fresh(cx, email, *, today, window_days=7):
    d = latest_scan_date(cx, email)
    if not d:
        return False
    from datetime import date
    try:
        sd = date.fromisoformat(d); td = date.fromisoformat(today)
    except ValueError:
        return False
    return 0 <= (td - sd).days <= window_days
