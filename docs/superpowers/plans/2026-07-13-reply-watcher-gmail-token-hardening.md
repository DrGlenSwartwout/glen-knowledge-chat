# Reply-watcher Gmail Token Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the reply-watcher (and the shared console-inbox Gmail loader) load their OAuth token from the durable `oauth_tokens` DB store with file fallback and refresh write-back, and fire exactly one deduped SMTP alert when the token can't be loaded at all — so the token can never silently drop into a 15-min failure flood.

**Architecture:** A new standalone module `dashboard/gmail_token.py` owns token resolution (DB row → file fallback → self-heal backfill), refresh write-back, and a small health/alert-dedup state row. `reply_watcher.py` and `dashboard/inbox.py` call it instead of their private `_resolve_token_path`/`_get_gmail_service`. `app.py`'s `cron_reply_watch` maps a typed `GmailTokenMissing` to a deduped SMTP alert. Reuses the exact `oauth_tokens` table + write-back pattern already in `app.py`'s `_run_cron`.

**Tech Stack:** Python 3, Flask (`app.py`), `google-auth`/`google-api-python-client`, `sqlite3` (`chat_log.db`), `smtplib` (existing transactional path), pytest.

## Global Constraints

- Durable store is the existing `oauth_tokens` table (`name TEXT PRIMARY KEY, token_json TEXT NOT NULL, updated_at TEXT NOT NULL`) — no new table, no migration.
- Token name for this path is exactly `inbox_gmail`; its health row is `inbox_gmail_health`.
- DB writes use a module-level `threading.Lock` and `sqlite3.connect(..., timeout=10)`; upsert is `INSERT ... ON CONFLICT(name) DO UPDATE`.
- `dashboard/gmail_token.py` MUST NOT import `app.py` (avoid circular import); it takes `db_path` as a parameter.
- Change is additive and backward-compatible: with no `inbox_gmail` DB row it falls back to the file exactly as today, then self-heals the DB.
- App-importing tests SILENTLY SKIP under bare pytest (no `PINECONE_API_KEY`); run everything under `doppler run -p remedy-match -c dev -- python3 -m pytest`. The `tests/test_gmail_token.py` module does NOT import `app` and runs without doppler.
- No em dashes / no ALL CAPS in code comments and copy.

---

### Task 1: `dashboard/gmail_token.py` — durable loader + self-heal

**Files:**
- Create: `dashboard/gmail_token.py`
- Test: `tests/test_gmail_token.py`

**Interfaces:**
- Produces: `GmailTokenMissing(RuntimeError)`; `LoadedGmail = namedtuple("LoadedGmail", ["creds", "source", "original_json", "name"])`; `default_db_path() -> str`; `load_gmail_credentials(db_path, name="inbox_gmail", scopes=None) -> LoadedGmail`. Internal helpers `_read_db_token`, `_write_db_token`, `_read_file_token`, `_build_creds`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gmail_token.py
import json
import sqlite3
import pytest
from pathlib import Path

from dashboard import gmail_token as gt

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def _token_json(access="ya29.access", refresh="1//refresh"):
    return json.dumps({
        "token": access,
        "refresh_token": refresh,
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "cid.apps.googleusercontent.com",
        "client_secret": "secret",
        "scopes": SCOPES,
    })

def _db(tmp_path):
    p = tmp_path / "chat_log.db"
    with sqlite3.connect(p) as cx:
        cx.execute("CREATE TABLE oauth_tokens (name TEXT PRIMARY KEY, "
                   "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        cx.commit()
    return str(p)

def test_loads_from_db_when_present(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO oauth_tokens VALUES (?,?,?)",
                   ("inbox_gmail", _token_json(), "2026-07-13T00:00:00Z"))
        cx.commit()
    loaded = gt.load_gmail_credentials(db, name="inbox_gmail", scopes=SCOPES)
    assert loaded.source == "db"
    assert loaded.name == "inbox_gmail"
    assert loaded.creds.refresh_token == "1//refresh"

def test_falls_back_to_file_and_self_heals_db(tmp_path, monkeypatch):
    db = _db(tmp_path)
    tokfile = tmp_path / "google-token.json"
    tokfile.write_text(_token_json(access="from-file"))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tokfile))
    loaded = gt.load_gmail_credentials(db, name="inbox_gmail", scopes=SCOPES)
    assert loaded.source == "file"
    # self-heal: DB row now exists
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT token_json FROM oauth_tokens WHERE name=?",
                         ("inbox_gmail",)).fetchone()
    assert row is not None
    assert json.loads(row[0])["token"] == "from-file"

def test_raises_when_nowhere(tmp_path, monkeypatch):
    db = _db(tmp_path)
    monkeypatch.delenv("GMAIL_TOKEN_PATH", raising=False)
    monkeypatch.setattr(gt, "_FILE_CANDIDATES", [str(tmp_path / "nope.json")])
    with pytest.raises(gt.GmailTokenMissing):
        gt.load_gmail_credentials(db, name="inbox_gmail", scopes=SCOPES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_gmail_token.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.gmail_token'`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/gmail_token.py
"""Durable Gmail OAuth token loading for the reply-watcher and console inbox.

Source of truth is the oauth_tokens DB row (name="inbox_gmail"), matching the
pattern app.py:_run_cron already uses for glen_gmail/rae_gmail. Falls back to the
token file on the Render persistent disk and self-heals the DB from it. Standalone:
takes db_path, never imports app.py.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import namedtuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Token-file fallback, same order as reply_watcher/dashboard.inbox used before.
_FILE_CANDIDATES = [
    "/data/google-token.json",                                  # Render persistent disk
    str(Path.home() / ".config" / "google" / "token.json"),     # local dev
]

_lock = threading.Lock()

LoadedGmail = namedtuple("LoadedGmail", ["creds", "source", "original_json", "name"])


class GmailTokenMissing(RuntimeError):
    """Raised when no usable Gmail token exists in the DB or on disk."""


def default_db_path() -> str:
    # dashboard/ is one level below the repo root where chat_log.db lives.
    return str(Path(__file__).resolve().parent.parent / "chat_log.db")


def _read_db_token(db_path: str, name: str) -> Optional[str]:
    with sqlite3.connect(db_path, timeout=10) as cx:
        row = cx.execute(
            "SELECT token_json FROM oauth_tokens WHERE name=?", (name,)
        ).fetchone()
    return row[0] if row else None


def _write_db_token(db_path: str, name: str, token_json: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _lock, sqlite3.connect(db_path, timeout=10) as cx:
        cx.execute(
            "CREATE TABLE IF NOT EXISTS oauth_tokens (name TEXT PRIMARY KEY, "
            "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        cx.execute(
            "INSERT INTO oauth_tokens (name, token_json, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET token_json=excluded.token_json, "
            "updated_at=excluded.updated_at",
            (name, token_json, ts),
        )
        cx.commit()


def _read_file_token() -> Optional[str]:
    env = os.environ.get("GMAIL_TOKEN_PATH")
    for c in ([env] if env else []) + _FILE_CANDIDATES:
        if c and Path(c).exists():
            return Path(c).read_text()
    return None


def _build_creds(token_json: str, scopes: Sequence[str]):
    from google.oauth2.credentials import Credentials
    info = json.loads(token_json)
    granted = set(info.get("scopes") or [])
    requested = set(scopes or DEFAULT_SCOPES)
    # Pass the intersection so a refresh never asks for a scope the token lacks
    # (which Google rejects as invalid_scope). Mirrors the old inbox.py logic.
    effective = list(requested & granted) if granted else list(requested)
    if not effective:
        effective = list(granted) or list(requested)
    return Credentials.from_authorized_user_info(info, scopes=effective)


def load_gmail_credentials(db_path: str, name: str = "inbox_gmail",
                           scopes: Optional[Sequence[str]] = None) -> LoadedGmail:
    scopes = list(scopes or DEFAULT_SCOPES)
    token_json = _read_db_token(db_path, name)
    source = "db"
    if not token_json:
        token_json = _read_file_token()
        source = "file"
    if not token_json:
        raise GmailTokenMissing(
            f"No Gmail token for '{name}' in the oauth_tokens DB or on disk. "
            f"Re-run '~/AI-Training/02 Skills/google-auth.py' and "
            f"PUT it to /api/tokens/{name}."
        )
    creds = _build_creds(token_json, scopes)
    normalized = creds.to_json()  # canonical baseline for refresh comparison
    if source == "file":
        _write_db_token(db_path, name, normalized)  # self-heal the durable store
    return LoadedGmail(creds=creds, source=source, original_json=normalized, name=name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_gmail_token.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/gmail_token.py tests/test_gmail_token.py
git commit -m "feat: durable Gmail token loader with file fallback + self-heal"
```

---

### Task 2: refresh write-back

**Files:**
- Modify: `dashboard/gmail_token.py`
- Test: `tests/test_gmail_token.py`

**Interfaces:**
- Consumes: `LoadedGmail`, `_write_db_token`.
- Produces: `persist_refreshed_credentials(db_path, loaded: LoadedGmail) -> bool` — writes back only if `loaded.creds.to_json()` differs from `loaded.original_json`; returns whether it wrote.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_gmail_token.py
def test_persist_writes_only_when_changed(tmp_path):
    db = _db(tmp_path)
    creds = gt._build_creds(_token_json(access="old"), SCOPES)
    baseline = creds.to_json()
    # unchanged -> no write, returns False
    loaded_same = gt.LoadedGmail(creds=creds, source="db",
                                 original_json=baseline, name="inbox_gmail")
    assert gt.persist_refreshed_credentials(db, loaded_same) is False
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM oauth_tokens").fetchone()[0] == 0
    # changed (simulate refresh) -> writes, returns True
    creds.token = "new-access"
    loaded_changed = gt.LoadedGmail(creds=creds, source="db",
                                    original_json=baseline, name="inbox_gmail")
    assert gt.persist_refreshed_credentials(db, loaded_changed) is True
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT token_json FROM oauth_tokens WHERE name=?",
                         ("inbox_gmail",)).fetchone()
    assert json.loads(row[0])["token"] == "new-access"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_gmail_token.py::test_persist_writes_only_when_changed -v`
Expected: FAIL with `AttributeError: module 'dashboard.gmail_token' has no attribute 'persist_refreshed_credentials'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to dashboard/gmail_token.py
def persist_refreshed_credentials(db_path: str, loaded: LoadedGmail) -> bool:
    """Write the token back to the DB if google-auth refreshed it during the run.
    Best-effort: comparison is against the normalized baseline captured at load."""
    current = loaded.creds.to_json()
    if current == loaded.original_json:
        return False
    _write_db_token(db_path, loaded.name, current)
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_gmail_token.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/gmail_token.py tests/test_gmail_token.py
git commit -m "feat: persist refreshed Gmail creds back to the durable store"
```

---

### Task 3: health + alert-dedup state

**Files:**
- Modify: `dashboard/gmail_token.py`
- Test: `tests/test_gmail_token.py`

**Interfaces:**
- Consumes: `_read_db_token`, `_write_db_token`.
- Produces: `record_ok(db_path, name, now_iso=None)`; `should_send_alert(db_path, name, now_iso, window_hours=6) -> bool`; `record_alert(db_path, name, now_iso)`. State lives in the `oauth_tokens` row named `f"{name}_health"`, `token_json` = `{"healthy": bool, "last_ok": iso|None, "last_alert": iso|None}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_gmail_token.py
def test_alert_dedup_within_window(tmp_path):
    db = _db(tmp_path)
    t0 = "2026-07-13T00:00:00+00:00"
    t_soon = "2026-07-13T02:00:00+00:00"   # +2h, inside 6h window
    t_later = "2026-07-13T07:00:00+00:00"  # +7h, outside window
    # first time: no health row -> should alert
    assert gt.should_send_alert(db, "inbox_gmail", t0) is True
    gt.record_alert(db, "inbox_gmail", t0)
    # inside window -> suppressed
    assert gt.should_send_alert(db, "inbox_gmail", t_soon) is False
    # outside window -> alert again
    assert gt.should_send_alert(db, "inbox_gmail", t_later) is True

def test_record_ok_clears_alert_and_marks_healthy(tmp_path):
    db = _db(tmp_path)
    t0 = "2026-07-13T00:00:00+00:00"
    gt.record_alert(db, "inbox_gmail", t0)
    gt.record_ok(db, "inbox_gmail", now_iso="2026-07-13T01:00:00+00:00")
    raw = gt._read_db_token(db, "inbox_gmail_health")
    state = json.loads(raw)
    assert state["healthy"] is True
    assert state["last_alert"] is None
    # after an OK, a later failure alerts again (window cleared)
    assert gt.should_send_alert(db, "inbox_gmail", "2026-07-13T01:30:00+00:00") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_gmail_token.py -k "alert or record_ok" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'should_send_alert'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to dashboard/gmail_token.py
def _health_name(name: str) -> str:
    return f"{name}_health"


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def record_ok(db_path: str, name: str, now_iso: Optional[str] = None) -> None:
    now = now_iso or datetime.now(timezone.utc).isoformat()
    _write_db_token(db_path, _health_name(name),
                    json.dumps({"healthy": True, "last_ok": now, "last_alert": None}))


def record_alert(db_path: str, name: str, now_iso: str) -> None:
    raw = _read_db_token(db_path, _health_name(name))
    state = json.loads(raw) if raw else {}
    state["healthy"] = False
    state["last_alert"] = now_iso
    _write_db_token(db_path, _health_name(name), json.dumps(state))


def should_send_alert(db_path: str, name: str, now_iso: str,
                      window_hours: int = 6) -> bool:
    raw = _read_db_token(db_path, _health_name(name))
    if not raw:
        return True
    last_alert = (json.loads(raw) or {}).get("last_alert")
    if not last_alert:
        return True
    return _parse_iso(now_iso) - _parse_iso(last_alert) >= timedelta(hours=window_hours)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_gmail_token.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/gmail_token.py tests/test_gmail_token.py
git commit -m "feat: Gmail token health + alert-dedup state row"
```

---

### Task 4: wire `reply_watcher.py` onto the durable loader

**Files:**
- Modify: `reply_watcher.py` (`_get_gmail_service` at line 51, `process_inbox_replies` at line 180)
- Test: `tests/test_reply_watcher_token_wiring.py` (new)

**Interfaces:**
- Consumes: `gmail_token.load_gmail_credentials`, `persist_refreshed_credentials`, `record_ok`, `default_db_path`, `GmailTokenMissing`, `LoadedGmail`.
- Produces: `_get_gmail_service(db_path=None) -> (svc, LoadedGmail)`; `process_inbox_replies` unchanged signature, but when it loads its own service it persists refreshed creds and records health at the end.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reply_watcher_token_wiring.py
import sqlite3
import types
import pytest
import reply_watcher as rw
from dashboard import gmail_token as gt

def _db(tmp_path):
    p = tmp_path / "chat_log.db"
    with sqlite3.connect(p) as cx:
        cx.execute("CREATE TABLE oauth_tokens (name TEXT PRIMARY KEY, "
                   "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        cx.commit()
    return str(p)

def test_process_persists_and_records_ok(tmp_path, monkeypatch):
    db = _db(tmp_path)
    # Fake loader returns a fake creds + LoadedGmail; avoid real Gmail/Google.
    fake_creds = types.SimpleNamespace(to_json=lambda: '{"token":"new"}')
    loaded = gt.LoadedGmail(creds=fake_creds, source="db",
                            original_json='{"token":"old"}', name="inbox_gmail")
    monkeypatch.setattr(rw, "_build_service_from_creds", lambda creds: object())
    monkeypatch.setattr(gt, "load_gmail_credentials", lambda *a, **k: loaded)
    persisted = {}
    monkeypatch.setattr(gt, "persist_refreshed_credentials",
                        lambda dbp, l: persisted.setdefault("hit", True))
    recorded = {}
    monkeypatch.setattr(gt, "record_ok",
                        lambda dbp, name: recorded.setdefault("hit", True))
    # Stub out the actual Gmail work: patch the label/list calls to no-op counts.
    monkeypatch.setattr(rw, "_ensure_label", lambda svc, name: "LBL")
    monkeypatch.setattr(rw, "_scan_and_process",
                        lambda svc, db_path, dry_run, max_messages,
                               processed_label_id, nonuser_label_id:
                        {"processed": 0, "skipped_nonuser": 0, "errored": 0, "details": []})
    counts = rw.process_inbox_replies(db_path=db, dry_run=True, max_messages=1)
    assert counts["processed"] == 0
    assert persisted.get("hit") is True
    assert recorded.get("hit") is True

def test_process_uses_injected_svc_without_token_load(tmp_path, monkeypatch):
    called = {"load": False}
    monkeypatch.setattr(gt, "load_gmail_credentials",
                        lambda *a, **k: called.__setitem__("load", True))
    monkeypatch.setattr(rw, "_ensure_label", lambda svc, name: "LBL")
    monkeypatch.setattr(rw, "_scan_and_process",
                        lambda *a, **k: {"processed": 0, "skipped_nonuser": 0,
                                          "errored": 0, "details": []})
    rw.process_inbox_replies(svc=object(), db_path=str(tmp_path / "x.db"))
    assert called["load"] is False  # injected svc bypasses token load
```

Note: this test assumes Task 4 refactors the message-scanning body of `process_inbox_replies` (lines 201 onward) into a helper `_scan_and_process(svc, db_path, dry_run, max_messages, processed_label_id, nonuser_label_id) -> counts`, and adds `_build_service_from_creds(creds) -> svc`. Do that refactor as part of Step 3 (behavior-preserving extraction).

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_reply_watcher_token_wiring.py -v`
Expected: FAIL (`_build_service_from_creds`/`_scan_and_process` not defined, and `process_inbox_replies` does not yet call the loader/persist/record_ok).

- [ ] **Step 3: Write minimal implementation**

Replace `reply_watcher.py` lines 51-64 (`_get_gmail_service`) and refactor `process_inbox_replies`:

```python
# reply_watcher.py — replace the old file-based _get_gmail_service
from dashboard import gmail_token as _gmail_token

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]  # keep existing


def _build_service_from_creds(creds):
    from googleapiclient.discovery import build
    return build("gmail", "v1", credentials=creds)


def _get_gmail_service(db_path=None):
    """Load durable creds (DB-first, file fallback, self-heal) and build the
    Gmail service. Returns (svc, LoadedGmail) so the caller can persist a refresh."""
    loaded = _gmail_token.load_gmail_credentials(
        db_path or _gmail_token.default_db_path(),
        name="inbox_gmail", scopes=GMAIL_SCOPES,
    )
    return _build_service_from_creds(loaded.creds), loaded
```

Refactor `process_inbox_replies` (keep the existing scanning body, just extract it and add the load/persist/record bookends):

```python
def process_inbox_replies(svc=None, db_path=None, dry_run=False, max_messages=50) -> dict:
    if db_path is None:
        db_path = _gmail_token.default_db_path()
    loaded = None
    if svc is None:
        svc, loaded = _get_gmail_service(db_path)

    processed_label_id = _ensure_label(svc, PROCESSED_LABEL)
    nonuser_label_id = _ensure_label(svc, NONUSER_LABEL)
    counts = _scan_and_process(svc, db_path, dry_run, max_messages,
                               processed_label_id, nonuser_label_id)

    if loaded is not None:
        try:
            _gmail_token.persist_refreshed_credentials(db_path, loaded)
            _gmail_token.record_ok(db_path, "inbox_gmail")
        except Exception as e:  # best-effort; never fail the run on write-back
            print(f"[reply-watcher] token write-back failed: {e!r}", flush=True)
    return counts


def _scan_and_process(svc, db_path, dry_run, max_messages,
                      processed_label_id, nonuser_label_id) -> dict:
    # MOVE the existing body from old process_inbox_replies (the query build,
    # messages().list(...), the per-message loop, and the counts dict) here
    # verbatim. Behavior-preserving extraction, no logic change.
    ...
```

Delete the old `_resolve_token_path`, `_TOKEN_PATH_CANDIDATES`, `DEFAULT_TOKEN_PATH`, and the old `_get_gmail_service` file-reading body (now superseded). Keep `_ensure_label`, `_strip_quoted_reply`, and the rest.

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_reply_watcher_token_wiring.py tests/test_reply_watcher_token_path.py -v`
Expected: the wiring tests PASS. `tests/test_reply_watcher_token_path.py` asserts the OLD `_resolve_token_path` behavior — update or delete it in this step since that function is removed (replaced by `dashboard/gmail_token.py`'s resolution, already covered by `tests/test_gmail_token.py`). If updating, point it at `gt.load_gmail_credentials` raising `GmailTokenMissing`.

- [ ] **Step 5: Commit**

```bash
git add reply_watcher.py tests/test_reply_watcher_token_wiring.py tests/test_reply_watcher_token_path.py
git commit -m "feat: reply-watcher uses durable Gmail token loader + write-back"
```

---

### Task 5: wire `dashboard/inbox.py` onto the durable loader

**Files:**
- Modify: `dashboard/inbox.py` (`_resolve_token_path` line 48, `_get_gmail_service` line 62)
- Test: `tests/test_inbox_token_wiring.py` (new)

**Interfaces:**
- Consumes: `gmail_token.load_gmail_credentials`, `default_db_path`.
- Produces: `dashboard/inbox.py:_get_gmail_service()` builds the service from durable creds (DB-first, file fallback, self-heal). Same public behavior for `list_threads`/`get_thread`/`send_reply`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inbox_token_wiring.py
import types
import pytest
from dashboard import inbox
from dashboard import gmail_token as gt

def test_inbox_service_uses_durable_loader(monkeypatch):
    fake_creds = object()
    loaded = gt.LoadedGmail(creds=fake_creds, source="db",
                            original_json="{}", name="inbox_gmail")
    seen = {}
    monkeypatch.setattr(gt, "load_gmail_credentials",
                        lambda db_path, name="inbox_gmail", scopes=None:
                        seen.setdefault("name", name) or loaded)
    monkeypatch.setattr(inbox, "_build_service",
                        lambda creds: types.SimpleNamespace(creds=creds))
    svc = inbox._get_gmail_service()
    assert seen["name"] == "inbox_gmail"
    assert svc.creds is fake_creds
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_inbox_token_wiring.py -v`
Expected: FAIL (`inbox._build_service` not defined; `_get_gmail_service` still reads a file).

- [ ] **Step 3: Write minimal implementation**

Replace `dashboard/inbox.py` lines 40-83 (`_TOKEN_PATH_CANDIDATES`, `_resolve_token_path`, `_get_gmail_service`) with:

```python
# dashboard/inbox.py
from dashboard import gmail_token as _gmail_token

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
]


def _build_service(creds):
    from googleapiclient.discovery import build
    return build("gmail", "v1", credentials=creds)


def _get_gmail_service():
    """Durable Gmail service for the console inbox: DB-first token (name
    'inbox_gmail'), file fallback, self-heal. Scope intersection is handled
    inside gmail_token._build_creds."""
    loaded = _gmail_token.load_gmail_credentials(
        _gmail_token.default_db_path(), name="inbox_gmail", scopes=GMAIL_SCOPES,
    )
    return _build_service(loaded.creds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_inbox_token_wiring.py tests/test_gmail_token.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/inbox.py tests/test_inbox_token_wiring.py
git commit -m "feat: console inbox uses durable Gmail token loader"
```

---

### Task 6: `cron_reply_watch` deduped SMTP alert on token loss

**Files:**
- Modify: `app.py` (`cron_reply_watch` at line 23319; add `_send_token_alert` helper near the SMTP block ~line 393)
- Test: `tests/test_reply_watch_alert.py` (new; imports `app`, runs under doppler dev)

**Interfaces:**
- Consumes: `gmail_token.GmailTokenMissing`, `should_send_alert`, `record_alert`; existing `LOG_DB`, `smtplib`.
- Produces: `_send_token_alert(subject, body) -> bool` (SMTP, returns sent?); `cron_reply_watch` returns 500 with `{"ok": False, "error": ...}` on `GmailTokenMissing`, firing at most one alert per 6h.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reply_watch_alert.py
import importlib, sys
from pathlib import Path
import pytest

def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")

def test_token_missing_alerts_once(monkeypatch, tmp_path):
    app_module = _app()
    from dashboard import gmail_token as gt
    db = str(tmp_path / "chat_log.db")
    import sqlite3
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE oauth_tokens (name TEXT PRIMARY KEY, "
                   "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        cx.commit()
    monkeypatch.setattr(app_module, "LOG_DB", db)
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    # process_inbox_replies raises token-missing
    def _raise(*a, **k):
        raise gt.GmailTokenMissing("no token")
    monkeypatch.setattr("reply_watcher.process_inbox_replies", _raise)
    sent = []
    monkeypatch.setattr(app_module, "_send_token_alert",
                        lambda subject, body: sent.append(subject) or True)
    client = app_module.app.test_client()
    r1 = client.post("/api/cron/reply-watch", headers={"X-Cron-Secret": "s3cret"})
    r2 = client.post("/api/cron/reply-watch", headers={"X-Cron-Secret": "s3cret"})
    assert r1.status_code == 500 and r2.status_code == 500
    assert len(sent) == 1  # deduped within the 6h window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_reply_watch_alert.py -v`
Expected: FAIL (`_send_token_alert` missing; `cron_reply_watch` does not dedup/alert on `GmailTokenMissing`).

- [ ] **Step 3: Write minimal implementation**

Add the helper near the existing SMTP code in `app.py`:

```python
def _send_token_alert(subject: str, body: str) -> bool:
    """One-off operational alert over SMTP (independent of the Gmail token).
    Recipient = ALERT_EMAIL env, else SMTP_USER. Returns whether it sent."""
    to_email = os.environ.get("ALERT_EMAIL") or os.environ.get("SMTP_USER")
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    frm = os.environ.get("SMTP_FROM", user)
    if not (to_email and host and user and pw):
        print(f"[token-alert] SMTP not configured; would send: {subject}", flush=True)
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = frm
        msg["To"] = to_email
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=10) as s:
            s.starttls()
            s.login(user, pw)
            s.sendmail(frm, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[token-alert] SMTP send failed: {e}", flush=True)
        return False
```

Change the `cron_reply_watch` try/except (lines 23341-23345) to special-case `GmailTokenMissing`:

```python
    from reply_watcher import process_inbox_replies
    from dashboard import gmail_token as _gt
    try:
        counts = process_inbox_replies(db_path=str(LOG_DB), dry_run=dry_run,
                                       max_messages=max_messages)
    except _gt.GmailTokenMissing as e:
        now_iso = datetime.now(timezone.utc).isoformat()
        if _gt.should_send_alert(str(LOG_DB), "inbox_gmail", now_iso):
            _send_token_alert(
                "Gmail token needs re-auth (reply-watcher down)",
                "The reply-watcher could not load its Gmail token from the DB or "
                "disk. Re-run '~/AI-Training/02 Skills/google-auth.py' and PUT it to "
                f"/api/tokens/inbox_gmail.\n\nDetail: {e}",
            )
            _gt.record_alert(str(LOG_DB), "inbox_gmail", now_iso)
        return jsonify({"ok": False, "error": str(e), "token_missing": True}), 500
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(e)}), 500
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_reply_watch_alert.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reply_watch_alert.py
git commit -m "feat: deduped SMTP alert when reply-watcher Gmail token is missing"
```

---

### Task 7: full-suite check + rollout note

**Files:**
- Modify: `docs/superpowers/plans/2026-07-13-reply-watcher-gmail-token-hardening.md` (check off) — no code.

- [ ] **Step 1: Run the affected suite green under doppler**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_gmail_token.py tests/test_reply_watcher_token_wiring.py tests/test_inbox_token_wiring.py tests/test_reply_watch_alert.py -v`
Expected: all PASS. Also run any pre-existing `tests/test_reply_watcher*.py` and `tests/test_inbox*.py` to confirm no regression.

- [ ] **Step 2: Confirm no bare-pytest email/side-effects**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_reply_watch_alert.py -v` and confirm the SMTP path is monkeypatched (no real send). Never run the full suite bare (unmocked fulfillment tests send real email — see repo memory).

- [ ] **Step 3: Rollout note (no code)**

Seeding is automatic: on the first reply-watch run after deploy, the loader falls back to the present `/data/google-token.json`, and self-heal writes it into `oauth_tokens[inbox_gmail]`. From then on the DB is source of truth and refreshes write back. Optional belt-and-suspenders: `PUT /api/tokens/inbox_gmail` (header `X-Console-Key: $CONSOLE_SECRET`) with the current token JSON. Confirm live after deploy: a reply-watch run logs source and the `oauth_tokens[inbox_gmail]` + `inbox_gmail_health` rows exist. Ensure `ALERT_EMAIL` (or `SMTP_USER`) is set in doppler prd so the alert has a recipient.

- [ ] **Step 4: Open PR**

```bash
git push -u origin sess/9da18059
gh pr create --title "Harden reply-watcher Gmail token (durable DB store + write-back + SMTP alert)" --body "Implements docs/superpowers/specs/2026-07-13-reply-watcher-gmail-token-hardening-design.md"
```
