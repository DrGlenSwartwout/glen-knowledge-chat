#!/usr/bin/env python3
"""
Beta cohort backfill — tags the inner-circle contacts in GHL with
'beta-personal-email' so the incentive engine includes them in the
beta-test sends.

Inner circle (manually maintained):
  - Glen Swartwout
  - Rae Luscombe
  - Kauilani Bright Perdomo (primary email — most-used; she has a
    secondary Kbrightperdomo@gmail.com used on biofield sends)
  - Keikilani Bright (Perdomo)
  - Active certification participants (separate sub-script — see
    backfill_active_cert_participants.py — pulls Practice Better
    attendance + Zoom replays in last 90 days)

Usage (run once at beta launch, idempotent for re-runs):
  doppler run --project remedy-match --config prd -- \\
    python3 scripts/backfill_beta_cohort.py
"""
import os
import sys
from pathlib import Path

# Make app.py importable as a sibling
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── Inner circle (verified emails from Glen's Gmail thread search) ─────────
INNER_CIRCLE = [
    # (display_name, email)
    ("Glen Swartwout",          "this.elf@gmail.com"),
    ("Rae Luscombe",            "suerae1111@gmail.com"),
    ("Kauilani Bright Perdomo", "restorealoha@gmail.com"),
    ("Keikilani Bright",        "keikibright@gmail.com"),
]


def ghl_tag_contact(email: str, tag: str) -> None:
    """Upsert + ensure tag is present. Wraps existing ghl_upsert_contact
    so this script can be tested without hitting GHL.

    NOTE: Imported lazily inside the function so test mocks via
    `patch("scripts.backfill_beta_cohort.ghl_tag_contact")` work cleanly.
    The real path imports app.ghl_upsert_contact only at runtime.
    """
    from app import ghl_upsert_contact
    cid, _, err = ghl_upsert_contact(email, "", "", source_tag=tag,
                                      extra_tags=[tag])
    if err:
        print(f"  WARN: {email} tag={tag}: {err}")


def tag_inner_circle(inner_circle):
    """Tag each non-placeholder email with 'beta-personal-email'."""
    for name, email in inner_circle:
        if not email or "<" in email:
            print(f"  SKIP: {name} (no email yet)")
            continue
        ghl_tag_contact(email, "beta-personal-email")
        print(f"  ✓ tagged {email}")


def main():
    print(f"Tagging {len(INNER_CIRCLE)} inner-circle contacts with "
          f"'beta-personal-email'...")
    tag_inner_circle(INNER_CIRCLE)
    print("\nNext step: run backfill_active_cert_participants.py "
          "to expand the beta cohort to certification attendees.")


if __name__ == "__main__":
    main()
