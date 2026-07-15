#!/usr/bin/env python3
"""Bulk-sync FMP-exported client photos into the client_photos store.

Reads a folder of image files named by client id_pk OR email, resolves each to an
email, writes to the store (source='fmp', precedence-respecting) and pushes to prod.
FMP snapshot + photos are Mac-only, so this runs locally and pushes the finished
bytes to prod. Run via `bash ~/deploy-chat/sync_client_photos.sh <folder>` (DRY=1
previews). See docs/superpowers/specs/2026-07-14-client-photos-slice3-*.md.
"""
import os, sys, sqlite3, json, base64, urllib.request
from collections import Counter
from dashboard import client_photos as cph

DB = os.path.join(os.environ.get("DATA_DIR", os.path.expanduser("~/deploy-chat")), "chat_log.db")
KEY = os.environ.get("CONSOLE_SECRET", "")
BASE = "https://illtowell.com"
_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def sniff_content_type(blob):
    if blob[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    if blob[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return None


def resolve_email(cx, key):
    key = (key or "").strip()
    if not key:
        return None
    if "@" in key:
        return key.lower()
    row = cx.execute("SELECT email FROM fmp_clients WHERE id_pk=?", (key,)).fetchone()
    return (row[0] or "").strip().lower() if row and row[0] else None


def push_prod(email, blob, ctype):
    if not BASE:
        return False
    body = json.dumps({"email": email, "content_type": ctype, "source": "fmp",
                       "image": base64.b64encode(blob).decode()}).encode()
    req = urllib.request.Request(BASE.rstrip("/") + "/api/console/client-photo", data=body,
                                 method="POST", headers={"X-Console-Key": KEY,
                                 "Content-Type": "application/json"})
    return bool(json.load(urllib.request.urlopen(req, timeout=30)).get("ok"))


def sync_folder(cx, folder, push_fn=None, write=True):
    results = []
    for fn in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in _EXTS:
            continue
        rec = {"file": fn}
        try:
            with open(os.path.join(folder, fn), "rb") as f:
                blob = f.read()
            ctype = sniff_content_type(blob)
            if not blob or not ctype:
                rec["action"] = "skip:not-an-image"; results.append(rec); continue
            email = resolve_email(cx, stem)
            if not email:
                rec["action"] = "skip:no-email"; results.append(rec); continue
            rec["email"] = email
            if not write:
                rec["action"] = "would-sync"; results.append(rec); continue
            if cph.put(cx, email, blob, ctype, source="fmp", force=False) is None:
                rec["action"] = "skip:precedence"; results.append(rec); continue
            pushed = push_fn(email, blob, ctype) if push_fn else False
            rec["action"] = "synced" if pushed else "synced(local-only)"
        except Exception as e:
            rec["action"] = "error:" + str(e)[:60]
        results.append(rec)
    return results


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/fmp-photos")
    dry = os.environ.get("DRY", "0") == "1"
    if not os.path.isdir(folder):
        print(f"folder not found: {folder}"); sys.exit(1)
    cx = sqlite3.connect(DB)
    results = sync_folder(cx, folder, push_fn=(None if dry else push_prod), write=not dry)
    cx.close()
    print(f"MODE={'DRY' if dry else 'LIVE'} folder={folder} files={len(results)}")
    for r in results:
        print(f"  {r['file']:30} {r.get('email','-'):32} {r['action']}")
    print("SUMMARY:", dict(Counter(r["action"].split(":")[0].split("(")[0] for r in results)))


if __name__ == "__main__":
    main()
