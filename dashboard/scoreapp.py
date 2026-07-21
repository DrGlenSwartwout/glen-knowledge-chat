"""ScoreApp recent quiz signups — read from chat_log.db (populated by /webhook/scoreapp)."""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from . import db

LOG_DB = str(Path(__file__).parent.parent / "chat_log.db")


def recent_signups(limit=10):
    try:
        with db.connect(LOG_DB) as cx:
            # Reuse the inbound_leads table — scoreapp signups land here as source='scoreapp'.
            # `id` lets the dashboard reuse the per-lead action endpoints
            # (/api/leads/<id>/draft-reply, /tag, /dismiss, /send-reply).
            rows = cx.execute("""
                SELECT id, received_at, first_name, last_name, email, raw_json,
                       COALESCE(status, 'pending') AS status
                FROM inbound_leads
                WHERE source = 'scoreapp'
                  AND (status IS NULL OR status != 'dismissed')
                ORDER BY received_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        signups = [{"id":    r[0],
                    "date":  r[1],
                    "name":  f"{r[2] or ''} {r[3] or ''}".strip(),
                    "email": r[4],
                    "status": r[6]} for r in rows]
        return {"signups": signups,
                "count": len(signups),
                "as_of": datetime.now(timezone.utc).isoformat()}
    except sqlite3.OperationalError:
        return {"signups": [],
                "count": 0,
                "as_of": datetime.now(timezone.utc).isoformat()}
