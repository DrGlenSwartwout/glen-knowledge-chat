"""Pull a client's most recent E4L voice scan into the local Biofield Analysis tool.

E4L scans live in a SEPARATE database, ~/AI-Training/e4l.db (kept fresh by the
e4l-daily-watch cron) — not the chat_log.db the Biofield app otherwise uses. As soon
as the client is identified, `scan_context()` reads that DB read-only and reports:
freshness (within a 2-week window), how many days ago the scan was, and the scan's
ranked findings. It NEVER raises — a missing DB, blank email, unknown client, or no
scan all return status "none" so the intake flow is never blocked.

The query (latest-scan + identity-merge handling + ranked findings) is vendored from
the vault tool `02 Skills/e4l_synthesis.py`, which is not importable by this app.
"""
import datetime
import os
import sqlite3


def _db_path(db_path=None):
    return db_path or os.path.expanduser(os.environ.get("E4L_DB", "~/AI-Training/e4l.db"))


def _connect_ro(path):
    """Open the e4l DB read-only. Returns None if the file is missing/unopenable
    (so we never silently create an empty e4l.db beside the app)."""
    if not os.path.exists(path):
        return None
    try:
        cx = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cx.row_factory = sqlite3.Row
        return cx
    except sqlite3.Error:
        return None


def _merge_group(cx, client_id):
    """All client_ids sharing this client's confirmed identity (inclusive), so a
    person's split duplicate accounts read as one history. {client_id} when the
    merges table is absent or untouched."""
    try:
        rows = cx.execute("SELECT dup_client_id, canonical_client_id "
                          "FROM e4l_identity_merges").fetchall()
    except sqlite3.Error:
        return {client_id}
    canon = {int(d): int(c) for d, c in rows}
    c = canon.get(client_id, client_id)
    group = {client_id, c}
    for dup, can in canon.items():
        if can == c:
            group.add(dup)
    return group


def _latest_scan(cx, email):
    """{scan_id, scan_date} for the most recent scan across the client's merged
    identity, or None."""
    crows = cx.execute("SELECT client_id FROM e4l_clients WHERE lower(email)=lower(?)",
                       (str(email or "").strip(),)).fetchall()
    if not crows:
        return None
    group = set()
    for cr in crows:
        group |= _merge_group(cx, cr["client_id"])
    ph = ",".join("?" for _ in group)
    r = cx.execute(f"""SELECT s.scan_id, s.scan_date FROM e4l_scans s
                       WHERE s.client_id IN ({ph})
                       ORDER BY s.scan_date DESC, s.scan_id DESC LIMIT 1""",
                   tuple(group)).fetchone()
    return dict(r) if r else None


def _findings(cx, scan_id, limit):
    """Ranked findings for a scan: {rank, code, name, description}, by priority."""
    rows = cx.execute(
        """SELECT r.item_code, r.priority_rank, i.name, i.full_name, i.e4l_description
           FROM e4l_scan_results r LEFT JOIN e4l_items i ON i.code = r.item_code
           WHERE r.scan_id=? ORDER BY (r.priority_rank IS NULL), r.priority_rank ASC, r.id ASC""",
        (scan_id,)).fetchall()
    out = []
    for r in rows:
        code = (r["item_code"] or "").strip()
        if not code:
            continue
        out.append({"rank": r["priority_rank"], "code": code,
                    "name": (r["full_name"] or r["name"] or code).strip(),
                    "description": (r["e4l_description"] or "").strip()})
    return out[:limit] if limit else out


def _days_ago(scan_date, today):
    """Whole days from scan_date to today (both YYYY-MM-DD). Clamped >= 0 so a future
    scan_date (data glitch) never reads as negative. None if either is unparseable."""
    try:
        s = datetime.date.fromisoformat((scan_date or "").strip())
        t = datetime.date.fromisoformat((today or "").strip())
    except ValueError:
        return None
    return max(0, (t - s).days)


def _none(window_days):
    return {"status": "none", "found": False, "scan_id": None, "scan_date": None,
            "days_ago": None, "fresh": False, "window_days": window_days,
            "findings": [], "message": "No E4L scan on file"}


def scan_context(email, today, *, db_path=None, window_days=14, limit=12):
    """Most recent E4L scan for `email` as of `today` (YYYY-MM-DD). Returns a dict:
      status: "fresh" | "stale" | "none"
      found, scan_id, scan_date, days_ago, fresh, window_days, findings, message
    Fresh = a scan exists within `window_days`. Never raises."""
    none = _none(window_days)
    if not (email or "").strip():
        return none
    cx = _connect_ro(_db_path(db_path))
    if cx is None:
        return none
    try:
        scan = _latest_scan(cx, email)
        if not scan:
            return none
        days = _days_ago(scan["scan_date"], today)
        findings = _findings(cx, scan["scan_id"], limit)
    except sqlite3.Error:
        return none
    finally:
        cx.close()
    fresh = days is not None and days <= window_days
    status = "fresh" if fresh else "stale"
    if fresh:
        message = f"Recent E4L scan · {days} day{'s' if days != 1 else ''} ago"
    elif days is None:
        message = "No fresh voice scan — last scan date unreadable"
    else:
        message = (f"No fresh voice scan — last scan {days} day"
                   f"{'s' if days != 1 else ''} ago (stale)")
    return {"status": status, "found": True, "scan_id": scan["scan_id"],
            "scan_date": scan["scan_date"], "days_ago": days, "fresh": fresh,
            "window_days": window_days, "findings": findings, "message": message}
