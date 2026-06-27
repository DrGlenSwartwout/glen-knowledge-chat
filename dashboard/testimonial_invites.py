"""Phase 4: candidate store for suggested testimonial invites (review-queue-first).

One open candidate per email. A human approves (-> send) or dismisses each. De-dup/cooldown
keeps the queue clean and avoids re-nagging.
"""
import datetime
import sqlite3

COOLDOWN_DAYS = 180


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _within_days(ts_iso, days):
    if not ts_iso:
        return False
    try:
        t = datetime.datetime.fromisoformat(ts_iso)
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
        return (datetime.datetime.now(datetime.timezone.utc) - t).days < days
    except Exception:
        return False


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS testimonial_invite_candidates ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, name TEXT, "
        "quote TEXT DEFAULT '', source TEXT DEFAULT '', kind TEXT DEFAULT 'general', "
        "confidence REAL DEFAULT 0, status TEXT DEFAULT 'pending', "
        "detected_at TEXT, decided_at TEXT, decided_by TEXT, sent_at TEXT)")
    cx.commit()


def upsert_candidate(cx, email, name, quote, source, kind, confidence):
    init_table(cx)
    e = (email or "").strip().lower()
    cx.execute(
        "INSERT INTO testimonial_invite_candidates "
        "(email,name,quote,source,kind,confidence,status,detected_at) "
        "VALUES (?,?,?,?,?,?,'pending',?) "
        "ON CONFLICT(email) DO UPDATE SET name=excluded.name, quote=excluded.quote, "
        "source=excluded.source, kind=excluded.kind, confidence=excluded.confidence, "
        "status='pending', detected_at=excluded.detected_at, decided_at='', decided_by='', sent_at=''",
        (e, name or "", quote or "", source or "", kind or "general", float(confidence or 0), _now()))
    cx.commit()
    return cx.execute("SELECT id FROM testimonial_invite_candidates WHERE email=?", (e,)).fetchone()[0]


def get(cx, candidate_id):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    r = cur.execute("SELECT * FROM testimonial_invite_candidates WHERE id=?", (candidate_id,)).fetchone()
    return dict(r) if r else None


def pending_queue(cx, limit=200):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT * FROM testimonial_invite_candidates WHERE status='pending' "
        "ORDER BY confidence DESC, detected_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def pending_count(cx):
    init_table(cx)
    return cx.execute(
        "SELECT COUNT(*) FROM testimonial_invite_candidates WHERE status='pending'").fetchone()[0]


def set_status(cx, candidate_id, status, *, by="", sent=False):
    init_table(cx)
    if sent:
        cx.execute("UPDATE testimonial_invite_candidates SET status=?, decided_at=?, decided_by=?, "
                   "sent_at=? WHERE id=?", (status, _now(), by or "", _now(), candidate_id))
    else:
        cx.execute("UPDATE testimonial_invite_candidates SET status=?, decided_at=?, decided_by=? "
                   "WHERE id=?", (status, _now(), by or "", candidate_id))
    cx.commit()


def send_invite_email(email, name, *, quote="", send=None):
    """Email a client a testimonial invite linking to the in-house /results form.
    inbox.send_email is suppression-checked. Best-effort: returns False, never raises."""
    import os
    from dashboard import inbox as _inbox
    send = send or _inbox.send_email
    e = (email or "").strip()
    if not e:
        return False
    base = (os.environ.get("PUBLIC_BASE_URL") or "https://illtowell.com").rstrip("/")
    try:
        opener = (f'You mentioned: "{quote.strip()}"' if (quote or "").strip()
                  else "It sounds like you have been seeing some good results")
        body = (f"Aloha {name or 'there'},\n\n"
                f"{opener} - that means a lot, and your story could help someone else take their first "
                f"step. If you have a moment, would you share your experience? It only takes a minute, "
                f"and a short video means the most:\n\n{base}/results\n\n"
                f"With gratitude,\nDr. Glen & Rae\n")
        send(e, "Would you share your experience?", body, from_name="Dr. Glen Swartwout")
        return True
    except Exception as ex:  # noqa: BLE001
        print(f"[testimonial_invites] send failed for {e!r}: {ex!r}", flush=True)
        return False


def should_skip(cx, email, *, cooldown_days=COOLDOWN_DAYS):
    """True when we should NOT create a candidate for this email:
    already submitted a testimonial / suppressed / has a pending candidate /
    was sent-or-dismissed within the cooldown window."""
    init_table(cx)
    e = (email or "").strip().lower()
    if not e:
        return True
    try:
        if cx.execute("SELECT 1 FROM product_reviews WHERE kind='testimonial' AND lower(email)=? "
                      "LIMIT 1", (e,)).fetchone():
            return True
    except sqlite3.OperationalError:
        pass
    try:
        from dashboard import email_suppression as _es
        if _es.is_suppressed(cx, e):
            return True
    except Exception:
        pass
    row = cx.execute(
        "SELECT status, COALESCE(NULLIF(sent_at,''), NULLIF(decided_at,''), detected_at) "
        "FROM testimonial_invite_candidates WHERE email=?", (e,)).fetchone()
    if row:
        status, ts = row[0], row[1]
        if status == "pending":
            return True
        if status in ("sent", "dismissed") and _within_days(ts, cooldown_days):
            return True
    return False
