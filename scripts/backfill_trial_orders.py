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
        res = P.backfill_trial_orders(cx, SP.get_session, dry_run=dry)
        label = "[DRY RUN] would apply" if dry else "Backfill complete"
        print(f"{label}: {res}")
    finally:
        cx.close()


if __name__ == "__main__":
    main()
