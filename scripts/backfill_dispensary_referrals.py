"""One-time backfill: materialize a kind='dispensary_portal' referral row for each
existing dispensary client, so current clients are durably attributed and L2-eligible.

Attribution only — writes NO reward, stamps NO rewarded_at. Idempotent (referee PK).

Usage:
    doppler run -p remedy-match -c prd -- env DATA_DIR=/path python3 \\
        scripts/backfill_dispensary_referrals.py [--dry-run]
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard import referrals as rf


def backfill(db_path, email_for_pid, *, dry_run=False):
    written = skipped = unresolved = 0
    with sqlite3.connect(db_path) as cx:
        rf.init_tables(cx)
        pairs = cx.execute(
            "SELECT practitioner_id, lower(customer_email) AS email, MIN(invoice_id) AS ref "
            "FROM dispensary_orders "
            "WHERE customer_email IS NOT NULL AND customer_email != '' "
            "GROUP BY practitioner_id, lower(customer_email)").fetchall()
        cache = {}
        for pid, email, ref in pairs:
            if pid not in cache:
                cache[pid] = (email_for_pid(pid) or "").strip().lower()
            owner = cache[pid]
            if not owner:
                unresolved += 1
                continue
            if dry_run:
                written += 1
                continue
            wrote = rf.record_redemption(cx, "", owner, email, ref, kind="dispensary_portal")
            if wrote:
                written += 1
            else:
                skipped += 1
    return {"written": written, "skipped": skipped, "unresolved": unresolved}


def _email_for_pid(pid):
    from dashboard.practitioner_portal import practitioner_email_by_id
    return practitioner_email_by_id(pid)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    base = os.environ.get("DATA_DIR") or "."
    path = os.path.join(base, "chat_log.db")
    result = backfill(path, _email_for_pid, dry_run=dry)
    print(f"backfill {'(dry-run) ' if dry else ''}complete: {result}", flush=True)
