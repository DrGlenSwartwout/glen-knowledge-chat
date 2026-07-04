"""Backfill logic for materializing kind='dispensary_portal' referral rows from
existing dispensary_orders. Pure over an injected email resolver so it is testable
without Supabase and callable both from the CLI script and an in-web console endpoint
(the prod chat_log.db is only mounted on the web container).

Attribution only — writes NO reward, stamps NO rewarded_at. Idempotent (referee PK).
"""
import sqlite3

from dashboard import referrals as rf


def backfill(db_path, email_for_pid, *, dry_run=False):
    """For each distinct (practitioner_id, lower(customer_email)) in dispensary_orders,
    resolve the practitioner email via email_for_pid(pid) and INSERT OR IGNORE a
    kind='dispensary_portal' redemption (order_ref = MIN invoice_id for the pair).
    Returns {written, skipped, unresolved}. dry_run counts would-writes, writes nothing."""
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
