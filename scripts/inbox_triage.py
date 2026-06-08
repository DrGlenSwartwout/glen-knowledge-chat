#!/usr/bin/env python3
"""One-time (re-runnable) inbox triage for drglenswartwout@gmail.com.

Archives the Promotions/Updates/Social firehose out of the Inbox (reversible —
messages stay in All Mail) and optionally creates filters so future mail in
those categories auto-skips the Inbox. Primary/personal, starred, and the last
30 days are always protected.

Run from ~/deploy-chat (reuses dashboard.inbox for the Gmail service):

    python3 scripts/inbox_triage.py count       # label stats (read-only)
    python3 scripts/inbox_triage.py dry-run      # exact count + 25 sample (read-only)
    python3 scripts/inbox_triage.py apply --confirm   # archive (needs gmail.modify)
    python3 scripts/inbox_triage.py filters --confirm # create auto-archive filters

Archive is reversible: nothing is deleted/trashed; restore by re-adding INBOX.
Requires the FULL-scope token (run scripts/gmail_reauth_full.py first).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import inbox as _inbox  # noqa: E402

# Standard sweep: Promotions/Updates/Social, older than 30 days, never starred.
ARCHIVE_QUERY = ("in:inbox -is:starred older_than:30d "
                 "(category:promotions OR category:updates OR category:social)")

# Future auto-archive filters.
FILTER_CATEGORIES = ["promotions", "updates", "social"]
# Belt-and-suspenders sender filters for the worst bulk-marketing domains seen
# in the inbox sample (category filters are best-effort due to Gmail ordering).
FILTER_SENDER_DOMAINS = [
    "gotowebinar.com", "alibaba.com", "promotelabs.email", "sendfoxmail.com",
    "digitaltriggers.io", "speakertunity.com", "industryrockstar.com",
    "johnthornhill.name",
]


def svc():
    return _inbox._get_gmail_service()


def label_stats(s):
    out = {}
    for lid in ("INBOX", "CATEGORY_PERSONAL", "STARRED", "UNREAD"):
        try:
            d = s.users().labels().get(userId="me", id=lid).execute()
            out[lid] = (d.get("threadsTotal"), d.get("threadsUnread"))
        except Exception as e:
            out[lid] = ("err", str(e)[:30])
    return out


def matching_ids(s, query):
    ids, tok = [], None
    while True:
        r = s.users().messages().list(userId="me", q=query, maxResults=500, pageToken=tok).execute()
        ids += [m["id"] for m in r.get("messages", [])]
        tok = r.get("nextPageToken")
        if not tok:
            return ids


def cmd_count(s):
    for k, (t, u) in label_stats(s).items():
        print(f"  {k:18} total={t}  unread={u}")


def cmd_dry_run(s):
    print(f"Query: {ARCHIVE_QUERY}\n")
    ids = matching_ids(s, ARCHIVE_QUERY)
    print(f"WOULD ARCHIVE: {len(ids)} messages\n\nSample (newest 25):")
    for mid in ids[:25]:
        h = s.users().messages().get(userId="me", id=mid, format="metadata",
                                     metadataHeaders=["From", "Subject", "Date"]).execute()
        hd = {x["name"]: x["value"] for x in h.get("payload", {}).get("headers", [])}
        frm = (hd.get("From", ""))[:34]
        subj = (hd.get("Subject", ""))[:46]
        print(f"  {frm:34}  {subj}")
    print("\n(no changes made — this is a dry run)")


def cmd_apply(s):
    ids = matching_ids(s, ARCHIVE_QUERY)
    print(f"Archiving {len(ids)} messages (removing INBOX label)...")
    done = 0
    for i in range(0, len(ids), 1000):
        chunk = ids[i:i + 1000]
        s.users().messages().batchModify(
            userId="me", body={"ids": chunk, "removeLabelIds": ["INBOX"]}).execute()
        done += len(chunk)
        print(f"  ...{done}/{len(ids)}")
    print(f"✓ Archived {done} messages (still in All Mail; reversible).")


def cmd_filters(s):
    created = []
    for cat in FILTER_CATEGORIES:
        body = {"criteria": {"query": f"category:{cat}"},
                "action": {"removeLabelIds": ["INBOX"]}}
        try:
            f = s.users().settings().filters().create(userId="me", body=body).execute()
            created.append(f"category:{cat} -> archive  (id {f.get('id')})")
        except Exception as e:
            created.append(f"category:{cat} FAILED: {str(e)[:60]}")
    for dom in FILTER_SENDER_DOMAINS:
        body = {"criteria": {"from": dom},
                "action": {"removeLabelIds": ["INBOX"]}}
        try:
            f = s.users().settings().filters().create(userId="me", body=body).execute()
            created.append(f"from:{dom} -> archive  (id {f.get('id')})")
        except Exception as e:
            created.append(f"from:{dom} FAILED: {str(e)[:60]}")
    print("Filters:")
    for c in created:
        print("  -", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["count", "dry-run", "apply", "filters"])
    ap.add_argument("--confirm", action="store_true", help="required for apply/filters")
    a = ap.parse_args()
    s = svc()
    if a.mode == "count":
        cmd_count(s)
    elif a.mode == "dry-run":
        cmd_dry_run(s)
    elif a.mode == "apply":
        if not a.confirm:
            raise SystemExit("Refusing to archive without --confirm. Run dry-run first.")
        cmd_apply(s)
    elif a.mode == "filters":
        if not a.confirm:
            raise SystemExit("Refusing to create filters without --confirm.")
        cmd_filters(s)


if __name__ == "__main__":
    main()
