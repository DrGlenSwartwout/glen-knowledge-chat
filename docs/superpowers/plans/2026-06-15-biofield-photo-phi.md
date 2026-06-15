# Biofield Photo PHI — Viewer + Retention + Consent — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the PHI photo gaps before `BIOFIELD_CHECKOUT_ENABLED` goes live: (1) a **console-gated photo viewer** so Glen's team can actually see an uploaded client photo, (2) a **retention sweep** that auto-deletes photos older than N days (default 30) and clears the flag, (3) a **consent notice** on the gate's photo step. Photos remain private (never publicly served).

**Architecture:** `dashboard/biofield_store.py` gains `clear_photo(cx, email)` and a pure-ish `purge_expired_photos(cx, *, photo_root, retention_days, now_ts)` (scans the photo dir by file mtime, deletes expired files, clears the matching DB flag). `app.py` adds `GET /admin/biofield/photo?email=` (CONSOLE_SECRET-gated; returns the image bytes with a path-under-root guard) and `POST /api/cron/biofield-photo-purge` (X-Cron-Secret; `BIOFIELD_PHOTO_RETENTION_DAYS` default 30). The gate page shows the consent notice.

**Tech Stack:** Python 3.11, Flask, sqlite, pytest.

**Context:** photos are at `DATA_DIR/biofield-photos/<sha256(email)>.<ext>` (private, not under STATIC, never served); `biofield_store` has `photo_on_file`/`photo_path`; `set_photo_on_file` sets them. Console-auth pattern: `if CONSOLE_SECRET: key = request.headers.get("X-Console-Key","") or request.args.get("key",""); if key != CONSOLE_SECRET: 401`. Cron-secret pattern: `X-Cron-Secret` == `CRON_SECRET` or `CONSOLE_SECRET`. `_biofield_data_dir()` returns the photo root parent.

**Test invocation:** pure → bare venv; app → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -q` (worktree; ignore the 2 known pre-existing failures).

---

### Task 1: `biofield_store` — clear_photo + purge_expired_photos

**Files:** Modify `dashboard/biofield_store.py`; Test `tests/test_biofield_photo_purge.py`

- [ ] **Step 1: Failing test**

```python
import os, sqlite3, time
from pathlib import Path
from dashboard import biofield_store as bs

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    bs.init_table(cx); return cx

def test_clear_photo(tmp_path):
    cx = _cx()
    p = tmp_path / "a.jpg"; p.write_bytes(b"x")
    bs.set_photo_on_file(cx, "p@x.com", str(p))
    assert bs.get(cx, "p@x.com")["photo_on_file"]
    bs.clear_photo(cx, "p@x.com")
    r = bs.get(cx, "p@x.com")
    assert not r["photo_on_file"] and not r["photo_path"]

def test_purge_expired_photos(tmp_path):
    cx = _cx()
    root = tmp_path / "biofield-photos"; root.mkdir()
    old = root / "old.jpg"; old.write_bytes(b"x")
    new = root / "new.jpg"; new.write_bytes(b"y")
    bs.set_photo_on_file(cx, "old@x.com", str(old))
    bs.set_photo_on_file(cx, "new@x.com", str(new))
    now = time.time()
    os.utime(old, (now - 40 * 86400, now - 40 * 86400))   # 40 days old
    os.utime(new, (now - 2 * 86400, now - 2 * 86400))      # 2 days old
    n = bs.purge_expired_photos(cx, photo_root=str(root), retention_days=30, now_ts=now)
    assert n == 1
    assert not old.exists() and new.exists()
    assert not bs.get(cx, "old@x.com")["photo_on_file"]    # flag cleared
    assert bs.get(cx, "new@x.com")["photo_on_file"]         # kept

def test_purge_missing_root_is_noop(tmp_path):
    cx = _cx()
    assert bs.purge_expired_photos(cx, photo_root=str(tmp_path / "nope"),
                                   retention_days=30, now_ts=time.time()) == 0
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** in `dashboard/biofield_store.py`:
```python
def clear_photo(cx, email):
    cx.execute("UPDATE biofield_readiness SET photo_on_file=0, photo_path=NULL, updated_at=? "
               "WHERE lower(email)=lower(?)", (_now(), str(email).strip()))
    cx.commit()

def purge_expired_photos(cx, *, photo_root, retention_days, now_ts):
    """Delete photo files older than retention_days (by file mtime) + clear their DB flag.
    Returns the count deleted. Safe if the dir is missing."""
    import os
    from pathlib import Path as _P
    root = _P(photo_root)
    if not root.is_dir():
        return 0
    cutoff = now_ts - retention_days * 86400
    n = 0
    for f in root.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                path_str = str(f)
                f.unlink()
                cx.execute("UPDATE biofield_readiness SET photo_on_file=0, photo_path=NULL, "
                           "updated_at=? WHERE photo_path=?", (_now(), path_str))
                n += 1
        except OSError:
            continue
    cx.commit()
    return n
```
(`_now()` already exists in the module.)

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(biofield-phi): clear_photo + purge_expired_photos`

---

### Task 2: admin photo viewer + purge cron

**Files:** Modify `app.py`; Test `tests/test_biofield_photo_routes.py`

- [ ] **Step 1: Failing test** —
  - `GET /admin/biofield/photo?email=p@x.com` without console key → 401; with key but no photo on file → 404; with key + a photo on file → 200, `Content-Type` image/*, body == the stored bytes. (Seed via `biofield_store.set_photo_on_file` pointing at a real tmp file under the biofield-photos dir.)
  - **Path guard:** if `photo_path` somehow points outside the biofield-photos root, the route returns 404 (does not serve it).
  - `POST /api/cron/biofield-photo-purge` with `X-Cron-Secret` → 200 `{ok, purged: N}`; deletes expired files (seed an old file). Wrong/no secret → 401.

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** in `app.py`:
  - `GET /admin/biofield/photo`: CONSOLE_SECRET gate; `email = request.args.get("email","")`; open LOG_DB; `row = biofield_store.get(cx, email)`; if no `photo_path` → 404. **Path guard:** resolve the photo dir `pd = (_biofield_data_dir()/"biofield-photos").resolve()`; `fp = Path(row["photo_path"]).resolve()`; if `pd not in fp.parents` or not `fp.is_file()` → 404. Else return the bytes: `return Response(fp.read_bytes(), mimetype=mimetypes.guess_type(str(fp))[0] or "application/octet-stream")`. (Add `import mimetypes` if needed; `Response` is already imported.)
  - `POST /api/cron/biofield-photo-purge`: X-Cron-Secret auth; `days = int(os.environ.get("BIOFIELD_PHOTO_RETENTION_DAYS", "30"))`; open LOG_DB; `biofield_store.init_table(cx)`; `n = biofield_store.purge_expired_photos(cx, photo_root=str(_biofield_data_dir()/"biofield-photos"), retention_days=days, now_ts=time.time())`; return `{ok:True, purged:n}`. (`import time` if needed.)

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(biofield-phi): console photo viewer + retention purge cron`

---

### Task 3: consent notice on the gate + doc

**Files:** Modify `static/biofield-ready.html`, `docs/biofield-gate.md`

- [ ] **Step 1:** In `static/biofield-ready.html`, near the photo upload control, add the consent notice (small, muted text):
  > "To prepare your Biofield Analysis, Dr. Glen's team uses a clear photo of your face to visualize you during remote biofield testing. By uploading it you consent to us storing it securely and using it only for your analysis and program design. We keep it private, never share or display it, and delete it after your analysis is complete. You can ask us to delete it any time."
  No em dashes, no ALL CAPS. Verify the page still parses (`html.parser`).
- [ ] **Step 2:** Append a "PHI / photo handling" section to `docs/biofield-gate.md`: private storage (already), the **console-gated viewer** (`/admin/biofield/photo?email=&key=`) as the team's only retrieval path, the **retention sweep** (`/api/cron/biofield-photo-purge`, `BIOFIELD_PHOTO_RETENTION_DAYS` default 30 — schedule a periodic hit), the **consent notice** on the gate + the matching ToS clause, and the open HIPAA-posture note (Glen/Rae/counsel; Render encrypts at rest, no BAA by default).
- [ ] **Step 3:** Suite green: `… -m pytest tests/test_biofield_photo_purge.py tests/test_biofield_photo_routes.py tests/test_biofield_gate_routes.py -q`.
- [ ] **Step 4:** Commit — `feat(biofield-phi): consent notice + doc`

---

## Self-review
- **Coverage:** viewer (console-gated, path-guarded) closes the retrieval gap; retention purge (mtime-based, configurable, cron) closes the deletion gap; consent notice + ToS clause closes the consent gap. HIPAA posture is flagged as Glen's call (not a code item).
- **Type consistency:** `clear_photo(cx,email)`, `purge_expired_photos(cx,*,photo_root,retention_days,now_ts)->int`; routes `/admin/biofield/photo`, `/api/cron/biofield-photo-purge`; env `BIOFIELD_PHOTO_RETENTION_DAYS`.
- **Risk:** PHI access — the viewer is CONSOLE_SECRET-gated + path-guarded to the photo dir (no traversal, no arbitrary-file read); purge only deletes inside the photo dir by mtime. Both low-risk; the viewer is the one new way to read a photo, and it requires the console key.

## Done
Team can view an uploaded biofield photo (console-gated), photos auto-delete after the retention window, and the gate carries a clear consent notice — clearing the PHI review for `BIOFIELD_CHECKOUT_ENABLED`.
