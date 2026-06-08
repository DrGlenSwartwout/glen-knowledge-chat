#!/usr/bin/env python3
"""Re-authorize Glen's Gmail token with the FULL scope superset.

The current token (~/.config/google/token.json) only has readonly + compose, so
archive (gmail.modify) and filter creation (gmail.settings.basic) return 403.
This re-runs the browser consent with a SUPERSET of scopes: everything the
existing tools already use (console_push_cron + tracking watcher) PLUS modify +
settings.basic. Run once, interactively, on Glen's Mac:

    python3 scripts/gmail_reauth_full.py

A browser opens; log in as drglenswartwout@gmail.com and approve. Done.
"""

import os
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

CREDENTIALS = Path(os.path.expanduser("~/.config/gmail-triage/credentials.json"))
TOKEN_OUT = Path(os.path.expanduser("~/.config/google/token.json"))

# Superset: keep every scope the existing token/tools rely on, and ADD
# gmail.modify (archive/label) + gmail.settings.basic (create filters).
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",          # NEW — archive/label
    "https://www.googleapis.com/auth/gmail.settings.basic",  # NEW — create filters
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/calendar.readonly",
]


def main():
    if not CREDENTIALS.exists():
        raise SystemExit(f"Missing OAuth client at {CREDENTIALS}")
    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_OUT.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_OUT.write_text(creds.to_json())
    print(f"\n✓ Token written to {TOKEN_OUT}")
    try:
        from googleapiclient.discovery import build
        svc = build("gmail", "v1", credentials=creds)
        who = svc.users().getProfile(userId="me").execute().get("emailAddress")
        print(f"✓ Authorized as: {who}")
        print("✓ Scopes granted:")
        for s in (creds.scopes or []):
            print("    -", s)
    except Exception as e:
        print(f"(verify skipped: {e})")


if __name__ == "__main__":
    main()
