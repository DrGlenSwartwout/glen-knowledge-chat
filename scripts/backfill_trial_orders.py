#!/usr/bin/env python3
"""
One-pass backfill — give every historical $1 biofield trial a captured-charge
order so it shows in the Payments console (/console/payments). Going-forward
trials already get an order at fulfillment; this covers trials completed before
that shipped.

Reads `biofield_trial_grants` from chat_log.db, re-fetches each Stripe checkout
session for its PaymentIntent + amount, and upserts a `biofield_trial` order
(status 'done'). Idempotent on (source, external_ref) — safe to re-run; only
missing orders are created. The heavy lifting + its tests live in
dashboard.payments.backfill_trial_orders.

Usage:
  doppler run --project remedy-match --config prd -- \\
    python3 scripts/backfill_trial_orders.py [--dry-run]
"""
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    dry = "--dry-run" in sys.argv
    from dashboard import payments as P, stripe_pay as SP

    log_db = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))) / "chat_log.db"
    cx = sqlite3.connect(str(log_db))
    cx.row_factory = sqlite3.Row
    try:
        if dry:
            # Count what WOULD be created without writing, by re-fetching sessions
            # and checking for an existing order — no upserts.
            grants = cx.execute("SELECT session_id, email FROM biofield_trial_grants").fetchall()
            would = {"created": 0, "skipped": 0, "unpaid": 0, "failed": 0}
            for g in grants:
                try:
                    sess = SP.get_session(g["session_id"]) or {}
                    pi = (sess.get("payment_intent") or "").strip()
                    if not pi:
                        would["unpaid"] += 1; continue
                    exists = cx.execute(
                        "SELECT 1 FROM orders WHERE source='biofield_trial' AND external_ref=?",
                        (pi,)).fetchone()
                    would["skipped" if exists else "created"] += 1
                except Exception as e:
                    print(f"[trial-backfill] {g['session_id']}: {e!r}", flush=True)
                    would["failed"] += 1
            print(f"[DRY RUN] {len(grants)} grants -> {would} (no writes)")
            return

        res = P.backfill_trial_orders(cx, SP.get_session)
        print(f"Backfill complete: {res}")
    finally:
        cx.close()


if __name__ == "__main__":
    main()
