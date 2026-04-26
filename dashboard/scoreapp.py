"""ScoreApp recent quiz signups — read from chat_log.db (populated by /webhook/scoreapp)."""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

LOG_DB = str(Path(__file__).parent.parent / "chat_log.db")


def recent_signups(limit=10):
    try:
        with sqlite3.connect(LOG_DB) as cx:
            # Reuse the inbound_leads table — scoreapp signups land here as source='scoreapp'
            # Schema: received_at, source, email, first_name, last_name, phone, raw_json
            rows = cx.execute("""
                SELECT received_at, first_name, last_name, email, raw_json
                FROM inbound_leads
                WHERE source = 'scoreapp'
                ORDER BY received_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        signups = [{"date": r[0],
                    "name": f"{r[1] or ''} {r[2] or ''}".strip(),
                    "email": r[3]} for r in rows]
        return {"signups": signups,
                "count": len(signups),
                "as_of": datetime.now(timezone.utc).isoformat()}
    except sqlite3.OperationalError:
        return {"signups": [],
                "count": 0,
                "as_of": datetime.now(timezone.utc).isoformat()}
