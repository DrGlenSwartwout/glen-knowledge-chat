"""Backfill: give every portal a current_scan_date (newest) + auto_advance default,
without disturbing clients who already pinned/opted out. Idempotent.

The real implementation lives in dashboard/portal_scan_backfill.py so the
console admin endpoint (app.py) and this CLI share one tested function."""
import sqlite3
from dashboard.portal_scan_backfill import backfill

if __name__ == "__main__":
    import os
    with sqlite3.connect(os.environ.get("LOG_DB", "chat_log.db")) as cx:
        print(backfill(cx))
