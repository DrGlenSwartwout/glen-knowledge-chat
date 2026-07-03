"""Free-tier analysis cadence: 1 request per calendar month per email.
Paid members bypass entirely (checked by caller). Atomic INSERT OR IGNORE claim."""
from datetime import datetime, timezone

def _month(month=None):
    return month or datetime.now(timezone.utc).strftime("%Y-%m")

def _norm(email):
    return (email or "").strip().lower()

def init_analysis_quota_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS analysis_quota (
        email TEXT NOT NULL, month TEXT NOT NULL, claimed_at TEXT NOT NULL,
        PRIMARY KEY (email, month))""")
    cx.commit()

def try_claim(cx, email, *, month=None):
    ok = cx.execute(
        "INSERT OR IGNORE INTO analysis_quota(email, month, claimed_at) VALUES (?,?,?)",
        (_norm(email), _month(month), datetime.now(timezone.utc).isoformat())
    ).rowcount == 1
    cx.commit()
    return ok

def claimed_this_month(cx, email, *, month=None):
    return cx.execute("SELECT 1 FROM analysis_quota WHERE email=? AND month=?",
                      (_norm(email), _month(month))).fetchone() is not None

def release(cx, email, *, month=None):
    """Undo a claim that turned out not to correspond to a completed request
    (e.g. the send/insert after the claim failed). No-op if no claim exists."""
    cx.execute("DELETE FROM analysis_quota WHERE email=? AND month=?",
              (_norm(email), _month(month)))
    cx.commit()
