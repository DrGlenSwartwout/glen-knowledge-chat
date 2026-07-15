# Client Photos — Slice 3 (FMP Folder Bulk Sync) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bulk-import FMP-exported client photos: a local script reads a folder of image files named by client id_pk (or email), resolves each to an email, and writes them into the shared `client_photos` store (respecting a client's own portal upload) + pushes to prod.

**Architecture:** Two units. (1) `client_photos.put` gains an optional `force` param and a source-precedence check so a bulk `fmp` write does not clobber a higher-precedence photo. (2) `scripts/sync_client_photos.py` — pure, testable functions (`sniff_content_type`, `resolve_email`, `sync_folder`) plus a thin `main`; runs locally (FMP snapshot + photos are Mac-only), pushes to prod via the Slice 1 endpoint. No new prod endpoint.

**Tech Stack:** Python 3 (stdlib only — `sqlite3`, `os`, `base64`, `urllib`), pytest, the Slice 1 store `dashboard/client_photos.py`, the prod push `POST /api/console/client-photo` (Slice 1).

## Global Constraints

- Source precedence: `portal-self` (4) > operator manual `console`/`fmp-intake-upload` (3) > `fmp` bulk (2) > `ghl` (1) > unknown (0).
- `put` stays **backward compatible**: default `force=True` (all existing Slice 1/2 callers keep last-write-wins). Only the Slice 3 sync passes `force=False`, which skips the write when an existing photo's source outranks the incoming source.
- Photo bytes' true type is detected from magic bytes, NOT the filename extension (`.jpg` may hold a PNG). Types: JPEG `FF D8 FF`, PNG `89 50 4E 47 0D 0A 1A 0A`, WEBP `RIFF....WEBP`, GIF `GIF87a`/`GIF89a`.
- Filename stem is the key: if it contains `@` → treat as email; else → resolve via `fmp_clients.id_pk = stem` → email. No email resolvable → skip that file.
- The sync runs LOCALLY against `~/deploy-chat/chat_log.db` and pushes to prod `https://illtowell.com` with `X-Console-Key: $CONSOLE_SECRET`. Invoke: `doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat PYTHONPATH=$HOME/deploy-chat python3 scripts/sync_client_photos.py <folder>` (same pattern as `fulfill_requests.sh`). `DRY=1` previews without writing local or prod.
- Store API (Slice 1): `put(cx, email, blob, content_type, source="upload", force=True) -> email|None`; `get(cx, email) -> {"blob","content_type"}|None`.

---

### Task 1: Precedence-aware `client_photos.put`

**Files:**
- Modify: `dashboard/client_photos.py`
- Test: `tests/test_client_photos.py` (extend — it already exists from Slice 1)

**Interfaces:**
- Produces: `put(cx, email, blob, content_type, source="upload", force=True) -> email|None` — with `force=False`, returns `None` (no write) when an existing row's source outranks `source`; a module-level `_rank(source) -> int`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_client_photos.py`:

```python
def test_precedence_fmp_does_not_overwrite_portal_self():
    cx = _cx()
    cp.put(cx, "a@b.com", b"client-chosen", "image/png", source="portal-self")
    # bulk fmp write must NOT clobber the client's own photo
    assert cp.put(cx, "a@b.com", b"from-fmp", "image/jpeg", source="fmp", force=False) is None
    got = cp.get(cx, "a@b.com")
    assert got["blob"] == b"client-chosen" and got["content_type"] == "image/png"


def test_precedence_fmp_overwrites_lower_and_equal():
    cx = _cx()
    cp.put(cx, "a@b.com", b"ghl-img", "image/png", source="ghl")
    assert cp.put(cx, "a@b.com", b"fmp-img", "image/jpeg", source="fmp", force=False) == "a@b.com"
    assert cp.get(cx, "a@b.com")["blob"] == b"fmp-img"          # fmp(2) > ghl(1)
    assert cp.put(cx, "a@b.com", b"fmp-2", "image/jpeg", source="fmp", force=False) == "a@b.com"
    assert cp.get(cx, "a@b.com")["blob"] == b"fmp-2"            # fmp == fmp, still writes


def test_precedence_fmp_writes_when_absent():
    cx = _cx()
    assert cp.put(cx, "a@b.com", b"fmp-img", "image/jpeg", source="fmp", force=False) == "a@b.com"


def test_force_true_default_always_writes():
    cx = _cx()
    cp.put(cx, "a@b.com", b"client", "image/png", source="portal-self")
    # a deliberate operator upload (default force=True) overwrites even portal-self
    assert cp.put(cx, "a@b.com", b"operator", "image/png", source="console") == "a@b.com"
    assert cp.get(cx, "a@b.com")["blob"] == b"operator"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_client_photos.py -k precedence -v`
Expected: FAIL — `put()` has no `force` kwarg (TypeError) / no precedence logic.

- [ ] **Step 3: Implement precedence in `dashboard/client_photos.py`**

Add after `_norm`:

```python
_RANK = {"portal-self": 4, "console": 3, "fmp-intake-upload": 3, "fmp": 2, "ghl": 1}


def _rank(source):
    return _RANK.get((source or "").strip().lower(), 0)
```

Replace `put` with:

```python
def put(cx, email, blob, content_type, source="upload", force=True):
    """Upsert a client's photo. force=True (default) = last-write-wins (all existing
    callers). force=False = skip when an existing photo's source outranks `source`
    (a bulk 'fmp' write must not clobber a client's own 'portal-self' upload).
    Returns the normalized email written, or None (no email/blob, or precedence skip)."""
    e = _norm(email)
    if not e or not blob:
        return None
    init_table(cx)
    if not force:
        row = cx.execute("SELECT source FROM client_photos WHERE email=?", (e,)).fetchone()
        if row and _rank(source) < _rank(row[0]):
            return None
    cx.execute(
        "INSERT INTO client_photos(email, image_blob, content_type, source, updated_at) "
        "VALUES(?,?,?,?,?) ON CONFLICT(email) DO UPDATE SET "
        "image_blob=excluded.image_blob, content_type=excluded.content_type, "
        "source=excluded.source, updated_at=excluded.updated_at",
        (e, blob, (content_type or "image/jpeg"), source, _now()))
    cx.commit()
    return e
```

- [ ] **Step 4: Run all store tests to verify pass (new + Slice 1 regressions)**

Run: `python3 -m pytest tests/test_client_photos.py -v`
Expected: PASS (Slice 1's 4 tests still pass under the `force=True` default + the 4 new precedence tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/client_photos.py tests/test_client_photos.py
git commit -m "Client photos Slice 3: precedence-aware put (force flag + source rank)"
```

---

### Task 2: `sync_client_photos.py` folder sync

**Files:**
- Create: `scripts/sync_client_photos.py`
- Create: `sync_client_photos.sh` (repo root wrapper)
- Test: `tests/test_sync_client_photos.py`

**Interfaces:**
- Consumes: `dashboard.client_photos.put(..., source="fmp", force=False)`; `fmp_clients(id_pk, email)`.
- Produces: `sniff_content_type(blob) -> str|None`; `resolve_email(cx, key) -> str|None`; `sync_folder(cx, folder, push_fn=None, write=True) -> list[dict]`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_sync_client_photos.py`:

```python
import base64, sqlite3, sys
from pathlib import Path
import pytest

repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
from scripts import sync_client_photos as s
from dashboard import client_photos as cp

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 8


def _db_with_client(id_pk="21459", email="jc@x.com"):
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE fmp_clients (id_pk TEXT, email TEXT)")
    cx.execute("INSERT INTO fmp_clients (id_pk, email) VALUES (?,?)", (id_pk, email))
    cx.commit()
    return cx


def test_sniff_content_type():
    assert s.sniff_content_type(PNG) == "image/png"
    assert s.sniff_content_type(JPEG) == "image/jpeg"
    assert s.sniff_content_type(b"RIFF\x00\x00\x00\x00WEBPblah") == "image/webp"
    assert s.sniff_content_type(b"not an image") is None


def test_resolve_email_id_and_email_and_unknown():
    cx = _db_with_client("21459", "jc@x.com")
    assert s.resolve_email(cx, "21459") == "jc@x.com"
    assert s.resolve_email(cx, "Someone@X.com") == "someone@x.com"   # email stem passes through
    assert s.resolve_email(cx, "99999") is None                       # unknown id


def test_sync_folder_writes_resolves_and_pushes(tmp_path):
    cx = _db_with_client("21459", "jc@x.com")
    (tmp_path / "21459.jpg").write_bytes(PNG)          # png bytes, .jpg name -> sniffed png
    (tmp_path / "bob@x.com.png").write_bytes(JPEG)     # email-named
    (tmp_path / "99999.jpg").write_bytes(PNG)          # no matching client
    (tmp_path / "notes.txt").write_bytes(b"hi")        # ignored (not an image ext)
    (tmp_path / "junk.png").write_bytes(b"not-image")  # image ext, non-image bytes
    pushed = []
    res = s.sync_folder(cx, str(tmp_path), push_fn=lambda e, b, c: pushed.append(e) or True)
    by_file = {r["file"]: r for r in res}
    assert by_file["21459.jpg"]["action"].startswith("synced") and by_file["21459.jpg"]["email"] == "jc@x.com"
    assert cp.get(cx, "jc@x.com")["content_type"] == "image/png"      # sniffed, not ext
    assert by_file["bob@x.com.png"]["action"].startswith("synced")
    assert by_file["99999.jpg"]["action"] == "skip:no-email"
    assert by_file["junk.png"]["action"] == "skip:not-an-image"
    assert "notes.txt" not in by_file                                 # non-image ext skipped
    assert set(pushed) == {"jc@x.com", "bob@x.com"}


def test_sync_folder_respects_precedence(tmp_path):
    cx = _db_with_client("21459", "jc@x.com")
    cp.put(cx, "jc@x.com", b"client-own", "image/png", source="portal-self")
    (tmp_path / "21459.jpg").write_bytes(JPEG)
    res = s.sync_folder(cx, str(tmp_path), push_fn=lambda e, b, c: True)
    assert res[0]["action"] == "skip:precedence"
    assert cp.get(cx, "jc@x.com")["blob"] == b"client-own"           # untouched


def test_sync_folder_dry_does_not_write(tmp_path):
    cx = _db_with_client("21459", "jc@x.com")
    (tmp_path / "21459.jpg").write_bytes(PNG)
    res = s.sync_folder(cx, str(tmp_path), push_fn=None, write=False)
    assert res[0]["action"] == "would-sync"
    assert cp.get(cx, "jc@x.com") is None                            # nothing written
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_sync_client_photos.py -v`
Expected: FAIL — `scripts/sync_client_photos.py` / its functions don't exist (ImportError).

- [ ] **Step 3: Implement `scripts/sync_client_photos.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_sync_client_photos.py -v`
Expected: PASS (6 tests). (These are pure-Python; they do NOT import `app`, so bare pytest is fine.)

- [ ] **Step 5: Add the wrapper `sync_client_photos.sh`**

```bash
#!/bin/bash
# On-demand: bulk-sync FMP-exported client photos (folder arg, default ~/Desktop/fmp-photos).
# DRY=1 previews without writing.  Usage: bash ~/deploy-chat/sync_client_photos.sh [folder]
doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" PYTHONPATH="$HOME/deploy-chat" python3 "$HOME/deploy-chat/scripts/sync_client_photos.py" "$@"
```

- [ ] **Step 6: Commit**

```bash
chmod +x scripts/sync_client_photos.py sync_client_photos.sh
git add scripts/sync_client_photos.py sync_client_photos.sh tests/test_sync_client_photos.py
git commit -m "Client photos Slice 3: FMP folder sync script + wrapper"
```

---

## Self-Review

- **Spec coverage:** id→email + email passthrough (Task 2 `resolve_email`) ✓; skip no-email (✓); precedence — fmp does not overwrite portal-self (Task 1 + Task 2 test) ✓; type from bytes not extension (Task 2 `sniff_content_type`) ✓; local write + prod push (Task 2) ✓; idempotent + per-file log (Task 2 `sync_folder`) ✓; DRY preview (✓); runs locally under doppler (wrapper) ✓.
- **Placeholders:** none — full code + commands throughout.
- **Type consistency:** `put(..., source, force)` and `get` match Slice 1 signatures; `sync_folder`/`resolve_email`/`sniff_content_type` used consistently across script and tests.
- **Backward compatibility:** `put` default `force=True` leaves Slice 1/2 callers unchanged — Step 4 of Task 1 re-runs the Slice 1 tests to confirm.
- **Deferred:** GHL pull (Slice 4); the FileMaker-side container export (Glen's manual step); content-hash skip for unchanged files (only if the folder proves large).
