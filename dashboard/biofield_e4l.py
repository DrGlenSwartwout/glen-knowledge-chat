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
import subprocess
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


# Item categories NOT listed for the manual Biofield Test (Glen, 2026-06-24): MR
# (MR1..MR10 — Super Cell Driver, Calm Mind, …). Items with no/other category are kept.
_EXCLUDE_CATEGORIES = ("MR",)


def _findings(cx, scan_id, limit, exclude_categories=_EXCLUDE_CATEGORIES):
    """Ranked findings for a scan: {rank, code, name, description}, by priority.
    Findings whose e4l_items.category is in `exclude_categories` are dropped."""
    excl = {str(c).strip().upper() for c in (exclude_categories or ())}
    rows = cx.execute(
        """SELECT r.item_code, r.priority_rank, i.name, i.full_name, i.e4l_description, i.category
           FROM e4l_scan_results r LEFT JOIN e4l_items i ON i.code = r.item_code
           WHERE r.scan_id=? ORDER BY (r.priority_rank IS NULL), r.priority_rank ASC, r.id ASC""",
        (scan_id,)).fetchall()
    out = []
    for r in rows:
        code = (r["item_code"] or "").strip()
        if not code:
            continue
        if (r["category"] or "").strip().upper() in excl:
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


# --- Client name picker -----------------------------------------------------

def search_clients(q, *, db_path=None, limit=20):
    """Match e4l_clients by name (or email substring) for the intake autocomplete.
    Groups each NAME's distinct emails so a same-name/different-email client is
    pickable. Returns [] for a blank query or missing DB (never raises):
      [{"name": str, "emails": [{"email", "client_id", "last_scan_date"}, ...]}]
    Name-groups are capped at `limit` (ordered by name)."""
    q = (q or "").strip()
    if not q:
        return []
    cx = _connect_ro(_db_path(db_path))
    if cx is None:
        return []
    like = f"%{q}%"
    try:
        rows = cx.execute(
            """SELECT c.client_id, c.name, c.email,
                      (SELECT MAX(s.scan_date) FROM e4l_scans s
                         WHERE s.client_id = c.client_id) AS last_scan_date
               FROM e4l_clients c
               WHERE c.name LIKE ? COLLATE NOCASE OR c.email LIKE ? COLLATE NOCASE
               ORDER BY c.name""", (like, like)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        cx.close()
    groups, order = {}, []
    for r in rows:
        name = (r["name"] or "").strip() or "(unnamed)"
        email = (r["email"] or "").strip()
        if name not in groups:
            if len(order) >= limit:
                continue
            groups[name] = []
            order.append(name)
        if email and not any(e["email"] == email for e in groups[name]):
            groups[name].append({"email": email, "client_id": r["client_id"],
                                 "last_scan_date": r["last_scan_date"]})
    return [{"name": n, "emails": groups[n]} for n in order]


# --- On-demand live E4L fetch ----------------------------------------------

def _vault_dir():
    return os.path.expanduser(os.environ.get("E4L_VAULT", "~/AI-Training"))


def _default_fetch_runner(client_id=None, name=None):
    """Pull this one client's NEW scans from the LIVE E4L portal, then parse them
    into e4l.db — the same two vault steps the daily cron runs, minus the Pinecone
    vectorize (not needed for the biofield panel). Inherits the process env, so
    E4L_USERNAME/E4L_PASSWORD must already be present (the app runs under `doppler
    run`). Raises on a non-zero exit (fetch_live converts that to {"ok": False})."""
    vault = _vault_dir()
    scraper = os.path.join(vault, "02 Skills", "scrape-e4l-http.py")
    parser = os.path.join(vault, "02 Skills", "parse-e4l-scans.py")
    sel = (["--client", str(client_id)] if client_id is not None
           else ["--client-name", str(name)])
    subprocess.run(["python3", scraper, *sel], check=True, capture_output=True,
                   text=True, timeout=180)
    subprocess.run(["python3", parser], check=True, capture_output=True,
                   text=True, timeout=120)
    return {"ok": True}


def fetch_live(client_id=None, name=None, *, runner=None):
    """Trigger an on-demand live fetch+parse for one client. `runner` is injectable
    so tests never hit the portal. Never raises — a scraper/login failure comes back
    as {"ok": False, "error": ...}."""
    if client_id is None and not (name or "").strip():
        return {"ok": False, "error": "no client identifier (client_id or name)"}
    runner = runner or _default_fetch_runner
    try:
        out = runner(client_id=client_id, name=name) or {}
        return {"ok": bool(out.get("ok", True)), **{k: v for k, v in out.items() if k != "ok"}}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": (e.stderr or str(e))[:300]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}
