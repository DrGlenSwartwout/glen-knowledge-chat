#!/usr/bin/env python3
"""Mint (or re-mint) Glen's Gmail OAuth token.

The existing token at ~/.config/google/token.json was revoked
(invalid_grant), which broke BOTH the tracking watcher and console_push_cron.py.
This runs the one-time browser consent and writes a fresh token with the full
scope set both tools expect.

Run it once, interactively, on Glen's Mac:
    python3 scripts/gmail_tracking_auth.py
A browser window opens; log in as the account that sends the "tracking number"
emails (drglenswartwout@gmail.com) and approve. That's it.
"""

import os
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

CREDENTIALS = Path(os.path.expanduser(
    "~/.config/gmail-triage/credentials.json"))            # OAuth client (installed)
TOKEN_OUT = Path(os.path.expanduser("~/.config/google/token.json"))

# Full set so the new token also restores console_push_cron.py. The watcher only
# needs gmail.readonly + gmail.compose (a subset).
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
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
    except Exception as e:
        print(f"(verify skipped: {e})")


if __name__ == "__main__":
    main()
