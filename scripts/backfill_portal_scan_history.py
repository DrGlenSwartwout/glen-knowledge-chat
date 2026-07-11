"""Backfill: give every portal a current_scan_date (newest) + auto_advance default,
without disturbing clients who already pinned/opted out. Idempotent."""
import sqlite3, sys
from dashboard import client_portal as cp, portal_biofield_reports as pbr

def backfill(cx) -> dict:
    cp.init_client_portal_table(cx); pbr.init_table(cx)
    emails = [r[0] for r in cx.execute("SELECT DISTINCT email FROM client_portals WHERE email IS NOT NULL")]
    filled = 0
    for email in emails:
        content = cp._read_content(cx, email) or {}
        changed = False
        if "auto_advance" not in content:
            content["auto_advance"] = True; changed = True
        if not content.get("current_scan_date"):
            dates = pbr.list_report_dates(cx, email)
            if dates:
                content["current_scan_date"] = dates[0]; changed = True
        if changed:
            cp._write_content(cx, email, content); filled += 1
    return {"portals": len(emails), "updated": filled}

if __name__ == "__main__":
    import os
    with sqlite3.connect(os.environ.get("LOG_DB", "chat_log.db")) as cx:
        print(backfill(cx))
