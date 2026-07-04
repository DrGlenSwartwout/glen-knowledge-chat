"""One-time backfill CLI: materialize a kind='dispensary_portal' referral row for each
existing dispensary client, so current clients are durably attributed and L2-eligible.

Core logic lives in dashboard.referral_backfill.backfill (shared with the in-web console
endpoint). Attribution only — writes NO reward, stamps NO rewarded_at. Idempotent.

Usage:
    doppler run -p remedy-match -c prd -- env DATA_DIR=/path python3 \\
        scripts/backfill_dispensary_referrals.py [--dry-run]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard.referral_backfill import backfill  # re-exported for callers/tests


def _email_for_pid(pid):
    from dashboard.practitioner_portal import practitioner_email_by_id
    return practitioner_email_by_id(pid)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    base = os.environ.get("DATA_DIR") or "."
    path = os.path.join(base, "chat_log.db")
    result = backfill(path, _email_for_pid, dry_run=dry)
    print(f"backfill {'(dry-run) ' if dry else ''}complete: {result}", flush=True)
