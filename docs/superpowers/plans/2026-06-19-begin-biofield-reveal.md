# Begin #4a — Biofield Analysis Reveal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A locally-produced ranked-remedy draft is ingested, reviewed/approved in a console, and revealed to the verified owner via a magic link as a free top match plus blurred deeper matches, setting the `biofield` gate (Find step 2).

**Architecture:** New `dashboard/biofield_reveals.py` store (table `biofield_reveals`, ai_draft -> confirmed). A draft-ingest endpoint stores pushed drafts. A console actions module (dispatch spine) edits/approves; approve mints a magic-link `auth_tokens` row and emails the owner. A token-verified reveal route serves the top-free / rest-blurred page and sets the `biofield` gate. No Stripe, no FMP, no server-side scan interpretation.

**Tech Stack:** Flask (Python 3.11), SQLite, the `auth_tokens` magic-link pattern, the Business-OS dispatch spine (`dashboard/actions.py`/`rbac.py`/`dispatch.py`), pytest + Flask test client.

## Global Constraints

- No emoji, no em dashes. Live page, no feature flag for the reveal route; the console is staff-gated (`_check_console_auth`). `main` auto-deploys; manual visual pass before relying on it.
- Auth on the ingest endpoint uses the EXISTING pattern, NOT a widened one: `expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")`; header `X-Cron-Secret` or `X-Console-Key` or `?key=`. (Per the #3 lesson - do not modify auth to satisfy a test; set `CRON_SECRET` in the test.)
- The reveal magic link is NOT consumed on view (reopenable; 30-day TTL). Stored reveals are shown ONLY via the token (verified ownership) - never from a typed email.
- A `confirmed` reveal is never overwritten by a re-pushed draft. Webhooks/ingest always return 200 on a valid push; helpers wrapped, never raise into the caller.
- The `biofield` gate is set via the existing `_record_entry_unlock("biofield", email)` (#3) - idempotent, wrapped. No new journey gate (it already exists in `VALID_TRIGGERS`).
- XSS-safe front-end (`textContent`/escaped injection; no raw user/AI text into innerHTML). All copy provisional (BNSN later).
- Test harness: `importlib` app load; tmp `LOG_DB`; init the new table + `auth_tokens` + `begin_funnel.init_journey_tables`; Flask test client; mock the send fn; mock GHL on any free-tier transition. Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Spec: `docs/superpowers/specs/2026-06-19-begin-biofield-reveal-design.md`.

---

### Task 1: `biofield_reveals` store module

**Files:** Create `dashboard/biofield_reveals.py`; Create `tests/test_biofield_reveals.py`.

**Interfaces produced:** `init_table(cx)`; `upsert_draft(cx, email, scan_date, top, blurred, source) -> int` (returns row id; no-op overwrite if already `confirmed`); `list_drafts(cx) -> [dict]`; `get(cx, id) -> dict|None`; `get_by_token_hash(cx, th) -> dict|None`; `set_top(cx, id, top)`; `approve(cx, id, by, token_hash) -> bool`. Row dict keys: `id, email, scan_date, top, blurred, status, approved_at, approved_by, created_at, updated_at` (top/blurred returned as parsed objects).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_biofield_reveals.py
"""Begin #4a - biofield_reveals store: ai_draft -> confirmed, idempotent draft."""
import sqlite3
import sys
from pathlib import Path
import pytest


def _mod():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import biofield_reveals
        return biofield_reveals
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    _mod().init_table(cx)
    return cx


def test_upsert_creates_draft(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19",
                         {"name": "Cistus Shield", "slug": "cistus-shield", "meaning": "Calm the terrain."},
                         [{"kind": "binder"}, {"kind": "mineral"}], "e4l-matcher")
    row = m.get(cx, rid)
    assert row["status"] == "ai_draft"
    assert row["top"]["name"] == "Cistus Shield"
    assert len(row["blurred"]) == 2


def test_upsert_updates_while_draft(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "One"}, [], "s")
    rid2 = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Two"}, [], "s")
    assert rid == rid2
    assert m.get(cx, rid)["top"]["name"] == "Two"


def test_confirmed_not_overwritten(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "One"}, [], "s")
    m.approve(cx, rid, "glen", "hash123")
    m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Two"}, [], "s")
    row = m.get(cx, rid)
    assert row["status"] == "confirmed"
    assert row["top"]["name"] == "One"


def test_set_top_stays_draft(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "One"}, [], "s")
    m.set_top(cx, rid, {"name": "Edited", "meaning": "new"})
    row = m.get(cx, rid)
    assert row["status"] == "ai_draft" and row["top"]["name"] == "Edited"


def test_approve_and_token_lookup(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "One"}, [], "s")
    assert m.approve(cx, rid, "glen", "hashABC") is True
    row = m.get_by_token_hash(cx, "hashABC")
    assert row["id"] == rid and row["status"] == "confirmed" and row["approved_by"] == "glen"


def test_list_drafts_only_drafts(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    r1 = m.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "A"}, [], "s")
    r2 = m.upsert_draft(cx, "b@x.com", "2026-06-19", {"name": "B"}, [], "s")
    m.approve(cx, r1, "glen", "h1")
    drafts = m.list_drafts(cx)
    assert [d["id"] for d in drafts] == [r2]
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveals.py -v`
Expected: FAIL (module not importable).

- [ ] **Step 3: Implement `dashboard/biofield_reveals.py`**

```python
"""Begin #4a store: per-scan funnel Biofield reveal drafts. ai_draft -> confirmed.
Distinct from portal_biofield_reports (the $300-service portal report)."""
import json
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS biofield_reveals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            top_json TEXT NOT NULL,
            blurred_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'ai_draft',
            token_hash TEXT,
            approved_at TEXT, approved_by TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(email, scan_date)
        )
    """)
    cx.commit()


def _row(r):
    if r is None:
        return None
    d = dict(r)
    d["top"] = json.loads(d.pop("top_json") or "{}")
    d["blurred"] = json.loads(d.pop("blurred_json") or "[]")
    return d


def upsert_draft(cx, email, scan_date, top, blurred, source):
    cx.row_factory = None
    email = (email or "").strip().lower()
    now = _now()
    existing = cx.execute(
        "SELECT id, status FROM biofield_reveals WHERE email=? AND scan_date=?",
        (email, scan_date)).fetchone()
    if existing is not None:
        rid, status = existing[0], existing[1]
        if status == "confirmed":
            return rid  # never overwrite a confirmed reveal
        cx.execute(
            "UPDATE biofield_reveals SET top_json=?, blurred_json=?, updated_at=? WHERE id=?",
            (json.dumps(top or {}), json.dumps(blurred or []), now, rid))
        cx.commit()
        return rid
    cur = cx.execute(
        "INSERT INTO biofield_reveals (email, scan_date, top_json, blurred_json, status, created_at, updated_at) "
        "VALUES (?,?,?,?, 'ai_draft', ?, ?)",
        (email, scan_date, json.dumps(top or {}), json.dumps(blurred or []), now, now))
    cx.commit()
    return cur.lastrowid


def list_drafts(cx):
    cx.row_factory = __import__("sqlite3").Row
    rows = cx.execute(
        "SELECT * FROM biofield_reveals WHERE status='ai_draft' ORDER BY id DESC").fetchall()
    return [_row(r) for r in rows]


def get(cx, rid):
    cx.row_factory = __import__("sqlite3").Row
    return _row(cx.execute("SELECT * FROM biofield_reveals WHERE id=?", (rid,)).fetchone())


def get_by_token_hash(cx, th):
    cx.row_factory = __import__("sqlite3").Row
    return _row(cx.execute(
        "SELECT * FROM biofield_reveals WHERE token_hash=? AND status='confirmed'", (th,)).fetchone())


def set_top(cx, rid, top):
    cx.execute("UPDATE biofield_reveals SET top_json=?, updated_at=? WHERE id=? AND status='ai_draft'",
               (json.dumps(top or {}), _now(), rid))
    cx.commit()


def approve(cx, rid, by, token_hash):
    now = _now()
    cur = cx.execute(
        "UPDATE biofield_reveals SET status='confirmed', approved_at=?, approved_by=?, token_hash=?, updated_at=? "
        "WHERE id=? AND status='ai_draft'",
        (now, by, token_hash, now, rid))
    cx.commit()
    return cur.rowcount == 1
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveals.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_reveals.py tests/test_biofield_reveals.py
git commit -m "feat: begin #4a biofield_reveals store (ai_draft -> confirmed)"
```

---

### Task 2: Draft ingest endpoint

**Files:** Modify `app.py`; Create `tests/test_biofield_reveal_routes.py`.

**Interfaces:** Consumes `dashboard.biofield_reveals` (Task 1). Produces route `POST /api/e4l/reveal-draft`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_biofield_reveal_routes.py
"""Begin #4a - reveal ingest + reveal route."""
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals
    import begin_funnel
    with sqlite3.connect(db) as cx:
        biofield_reveals.init_table(cx)
        begin_funnel.init_journey_tables(cx)
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)
    return db


def test_ingest_stores_draft(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    client = app_module.app.test_client()
    r = client.post("/api/e4l/reveal-draft",
                    json={"email": "a@x.com", "scan_date": "2026-06-19",
                          "top_match": {"name": "Cistus", "slug": "cistus", "meaning": "Calm."},
                          "blurred": [{"kind": "binder"}], "source": "e4l-matcher"},
                    headers={"X-Cron-Secret": "k"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    from dashboard import biofield_reveals
    with sqlite3.connect(db) as cx:
        d = biofield_reveals.list_drafts(cx)
    assert len(d) == 1 and d[0]["top"]["name"] == "Cistus"


def test_ingest_requires_auth(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    client = app_module.app.test_client()
    r = client.post("/api/e4l/reveal-draft", json={"email": "a@x.com", "scan_date": "d",
                    "top_match": {"name": "X"}}, headers={"X-Cron-Secret": "wrong"})
    assert r.status_code == 401


def test_ingest_missing_email_400(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    client = app_module.app.test_client()
    r = client.post("/api/e4l/reveal-draft", json={"scan_date": "d", "top_match": {"name": "X"}},
                    headers={"X-Cron-Secret": "k"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_routes.py::test_ingest_stores_draft -v`
Expected: FAIL (404 / route missing).

- [ ] **Step 3: Add the ingest route**

In `app.py`, near `api_e4l_scan_freshness` (grep `def api_e4l_scan_freshness`), add:

```python
@app.route("/api/e4l/reveal-draft", methods=["POST"])
def api_e4l_reveal_draft():
    """Ingest a locally-produced Biofield reveal draft (top match + blurred list)
    for console review. Auth: X-Cron-Secret (== CRON_SECRET, falls back to
    CONSOLE_SECRET)."""
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    scan_date = (data.get("scan_date") or "").strip()
    top = data.get("top_match") or {}
    if not email or not scan_date or not (top.get("name") or "").strip():
        return jsonify({"error": "email, scan_date, top_match.name required"}), 400
    from dashboard import biofield_reveals as _br
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _br.init_table(cx)
            rid = _br.upsert_draft(cx, email, scan_date, top,
                                   data.get("blurred") or [], (data.get("source") or "").strip())
        return jsonify({"ok": True, "id": rid})
    except Exception as e:
        print(f"[reveal-draft] {e!r}", flush=True)
        return jsonify({"ok": False, "error": "store failed"}), 500
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_routes.py -v`
Expected: the three ingest tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_biofield_reveal_routes.py
git commit -m "feat: begin #4a reveal-draft ingest endpoint"
```

---

### Task 3: Console actions (edit / approve + magic link + email)

**Files:** Create `dashboard/biofield_reveal_actions.py`; Modify `app.py` (register + configure at startup); Create `tests/test_biofield_reveal_actions.py`.

**Interfaces:** Consumes `dashboard.biofield_reveals`, `dashboard.actions`, `dashboard.rbac`. Produces actions `biofield_reveal.edit` / `biofield_reveal.approve`; `configure(**kw)` injecting `base_url`, `send` (fn `(to, subject, body)->bool`), `hash_token` (fn), `mint_token` (fn `()->str`). The approve executor mints a token, stamps it on the row, INSERTs an `auth_tokens` row (purpose `biofield_reveal`), and emails the owner.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_biofield_reveal_actions.py
import sqlite3, sys
from pathlib import Path
import pytest


def _mods():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import biofield_reveals, biofield_reveal_actions
        return biofield_reveals, biofield_reveal_actions
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
    from dashboard import biofield_reveals
    biofield_reveals.init_table(cx)
    return cx


class _Actor:
    name = "glen"


def test_approve_confirms_mints_token_and_sends(tmp_path):
    br, acts = _mods()
    cx = _cx(tmp_path)
    rid = br.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Cistus"}, [], "s")
    sent = []
    acts.configure(base_url="https://x.test",
                   send=lambda to, subject, body: sent.append((to, subject, body)) or True,
                   hash_token=lambda t: "H:" + t, mint_token=lambda: "TOK123")
    acts._exec_approve({"id": rid}, {"cx": cx, "actor": _Actor()})
    row = br.get(cx, rid)
    assert row["status"] == "confirmed"
    at = cx.execute("SELECT email, purpose FROM auth_tokens WHERE token_hash=?", ("H:TOK123",)).fetchone()
    assert at == ("a@x.com", "biofield_reveal")
    assert len(sent) == 1 and "/begin/biofield/TOK123" in sent[0][2]


def test_approve_never_fails_on_send_error(tmp_path):
    br, acts = _mods()
    cx = _cx(tmp_path)
    rid = br.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "C"}, [], "s")
    def _boom(*a, **k):
        raise RuntimeError("smtp down")
    acts.configure(base_url="https://x.test", send=_boom,
                   hash_token=lambda t: "H:" + t, mint_token=lambda: "TOK")
    acts._exec_approve({"id": rid}, {"cx": cx, "actor": _Actor()})  # must not raise
    assert br.get(cx, rid)["status"] == "confirmed"


def test_edit_updates_top_stays_draft(tmp_path):
    br, acts = _mods()
    cx = _cx(tmp_path)
    rid = br.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Old"}, [], "s")
    acts._exec_edit({"id": rid, "name": "New", "meaning": "warm"}, {"cx": cx, "actor": _Actor()})
    row = br.get(cx, rid)
    assert row["status"] == "ai_draft" and row["top"]["name"] == "New" and row["top"]["meaning"] == "warm"
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_actions.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `dashboard/biofield_reveal_actions.py`**

```python
"""Begin #4a console actions: edit / approve a Biofield reveal. On approve,
mint a magic-link token, stamp it, write an auth_tokens row, email the owner.
Registered on the Business-OS dispatch spine. app.py injects deps via configure()."""
from datetime import datetime, timezone, timedelta

from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import biofield_reveals as _br

_DEPS = {}  # base_url, send, hash_token, mint_token - set by app.py


def configure(**kw):
    _DEPS.update(kw)


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_edit(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    cur = _br.get(ctx["cx"], rid)
    if not cur:
        raise ValueError("not found")
    top = dict(cur["top"])
    if "name" in params:
        top["name"] = (params.get("name") or "").strip()
    if "meaning" in params:
        top["meaning"] = (params.get("meaning") or "").strip()
    if "slug" in params:
        top["slug"] = (params.get("slug") or "").strip()
    _br.set_top(ctx["cx"], rid, top)
    return {"ok": True}


def _exec_approve(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    row = _br.get(ctx["cx"], rid)
    if not row:
        raise ValueError("not found")
    mint = _DEPS.get("mint_token") or (lambda: "tok")
    hash_token = _DEPS.get("hash_token") or (lambda t: t)
    token = mint()
    th = hash_token(token)
    ok = _br.approve(ctx["cx"], rid, _actor_name(ctx.get("actor")), th)
    if not ok:
        return {"ok": False, "note": "already approved"}
    now = datetime.now(timezone.utc)
    exp = (now + timedelta(days=30)).isoformat()
    ctx["cx"].execute(
        "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
        (th, row["email"], "biofield_reveal", now.isoformat(), exp))
    ctx["cx"].commit()
    # Best-effort notify; approval must never fail if the email fails.
    try:
        send = _DEPS.get("send")
        base = _DEPS.get("base_url", "")
        if send:
            url = f"{base}/begin/biofield/{token}"
            body = ("Aloha,\n\nYour Biofield Analysis is ready. View your top remedy match here:\n"
                    f"{url}\n\nIn wellness,\nDr. Glen and Rae\n")
            send(row["email"], "Your Biofield Analysis is ready", body)
    except Exception as e:  # noqa: BLE001 - notify must never fail the approve
        print(f"[biofield-reveal-approve] notify failed: {e!r}", flush=True)
    return {"ok": True}


def register():
    if get_action("biofield_reveal.approve"):
        return
    register_action(Action(
        key="biofield_reveal.edit", module="biofield_reveal", title="Edit Biofield reveal",
        description="Edit the top-match name/meaning (stays draft).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="biofield_reveal.approve", module="biofield_reveal", title="Approve Biofield reveal",
        description="Approve the top reveal, mint the magic link, and email the owner.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
```

- [ ] **Step 4: Register + configure at app startup**

Find where `sales_pages_actions` is registered/configured in `app.py` (grep `sales_pages_actions`). Beside it, add:

```python
    from dashboard import biofield_reveal_actions as _bra
    _bra.configure(base_url=PUBLIC_BASE_URL, send=_send_inquiry_email,
                   hash_token=_hash_token, mint_token=lambda: secrets.token_urlsafe(32))
    _bra.register()
```

(`PUBLIC_BASE_URL`, `_send_inquiry_email`, `_hash_token`, `secrets` already exist in app.py.)

- [ ] **Step 5: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_actions.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add dashboard/biofield_reveal_actions.py app.py tests/test_biofield_reveal_actions.py
git commit -m "feat: begin #4a console actions edit/approve + magic link + ready email"
```

---

### Task 4: Console API + page + nav

**Files:** Modify `app.py` (list endpoint + serve route); Create `static/console-biofield-reveals.html`; Modify the console nav (`static/op-nav.js` or the console nav include); add a serve test to `tests/test_biofield_reveal_routes.py`.

**Interfaces:** Consumes `dashboard.biofield_reveals.list_drafts`. Produces `GET /api/console/biofield-reveals` (JSON list) and `GET /console/biofield-reveals` (page). The page posts to the existing `/api/action/<key>` route with `biofield_reveal.edit`/`biofield_reveal.approve`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_biofield_reveal_routes.py`:

```python
def test_console_list_drafts(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "ck", raising=False)
    from dashboard import biofield_reveals
    with sqlite3.connect(db) as cx:
        biofield_reveals.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Cistus"}, [], "s")
    client = app_module.app.test_client()
    r = client.get("/api/console/biofield-reveals?key=ck")
    assert r.status_code == 200
    body = r.get_json()
    assert body["drafts"][0]["top"]["name"] == "Cistus"


def test_console_page_served(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "ck", raising=False)
    client = app_module.app.test_client()
    r = client.get("/console/biofield-reveals?key=ck")
    assert r.status_code == 200
    assert b"biofield" in r.data.lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_routes.py -k console -v`
Expected: FAIL (routes missing).

- [ ] **Step 3: Add the list endpoint + serve route**

In `app.py`, model the auth on the existing `/console/sales-pages` (grep `console_biofield_portal_page` or `console-sales-pages` for the `_check_console_auth` helper). Add:

```python
@app.route("/api/console/biofield-reveals", methods=["GET"])
def api_console_biofield_reveals():
    if not _check_console_auth(request):
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import biofield_reveals as _br
    with sqlite3.connect(LOG_DB) as cx:
        _br.init_table(cx)
        drafts = _br.list_drafts(cx)
    return jsonify({"drafts": drafts})


@app.route("/console/biofield-reveals", methods=["GET"])
def console_biofield_reveals_page():
    if not _check_console_auth(request):
        return jsonify({"error": "unauthorized"}), 401
    return send_from_directory(STATIC, "console-biofield-reveals.html")
```

(Use the SAME `_check_console_auth(request)` helper the other console routes use; grep it to confirm its exact name/signature and match it.)

- [ ] **Step 4: Create `static/console-biofield-reveals.html`**

Model on `static/console-sales-pages.html` (same gate/`key()`/`api()` helpers, same styling). It must: read the console key from the URL; `GET /api/console/biofield-reveals` to list drafts; render each draft showing the email, scan_date, an editable top-match name + meaning textarea, and the blurred count; a "Save edit" button posting `POST /api/action/biofield_reveal.edit` with `{id, name, meaning}` and an "Approve and send" button posting `POST /api/action/biofield_reveal.approve` with `{id}` (the existing `/api/action/<key>` route + console key header). All text rendered via `textContent`. Include the literal string `biofield` in the page (heading "Biofield Reveals"). Keep it under ~200 lines; no emoji, no em dashes.

- [ ] **Step 5: Add the nav sub-tab**

In the console nav (grep `op-nav.js` for the "Sales Pages" sub-tab added in Phase 5), add a "Biofield Reveals" entry linking to `/console/biofield-reveals` under the same Business-OS group. Match the existing entry's format exactly.

- [ ] **Step 6: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_routes.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app.py static/console-biofield-reveals.html static/op-nav.js tests/test_biofield_reveal_routes.py
git commit -m "feat: begin #4a console list + review page + nav"
```

---

### Task 5: Token-verified reveal page + gate

**Files:** Modify `app.py` (reveal route); Create `static/begin-biofield.html`; add a reveal test to `tests/test_biofield_reveal_routes.py`.

**Interfaces:** Consumes `dashboard.biofield_reveals.get_by_token_hash`, `_hash_token`, `_record_entry_unlock` (#3), the `auth_tokens` table. Produces `GET /begin/biofield/<token>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_biofield_reveal_routes.py`:

```python
def _approve_a_reveal(app_module, db, email="a@x.com"):
    """Create+approve a reveal directly, returning the plaintext token."""
    import secrets as _s
    from dashboard import biofield_reveals
    token = "tkn_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        rid = biofield_reveals.upsert_draft(cx, email, "2026-06-19",
              {"name": "Cistus Shield", "slug": "cistus-shield", "meaning": "Calm the terrain."},
              [{"kind": "binder"}, {"kind": "mineral"}], "s")
        biofield_reveals.approve(cx, rid, "glen", th)
        from datetime import datetime, timezone, timedelta
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def test_reveal_valid_token_renders_and_sets_gate(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    token = _approve_a_reveal(app_module, db)
    client = app_module.app.test_client()
    r = client.get(f"/begin/biofield/{token}")
    assert r.status_code == 200
    assert b"Cistus Shield" in r.data
    assert "no-store" in r.headers.get("Cache-Control", "")
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="a@x.com")
    assert "biofield" in st["unlocked_gates"]


def test_reveal_invalid_token_friendly_no_gate(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    r = client.get("/begin/biofield/bogus")
    assert r.status_code == 200  # friendly page, not a 500
    assert b"Cistus" not in r.data
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="a@x.com")
    assert "biofield" not in (st["unlocked_gates"] or [])
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_routes.py -k reveal_ -v`
Expected: FAIL (route missing).

- [ ] **Step 3: Add the reveal route**

In `app.py`, near the other `/begin/...` routes, add:

```python
@app.route("/begin/biofield/<token>", methods=["GET"])
def begin_biofield_reveal(token):
    """Token-verified Biofield reveal: top match free + blurred depth. Sets the
    biofield gate. The token is NOT consumed (reopenable, 30-day TTL)."""
    from dashboard import biofield_reveals as _br
    th = _hash_token((token or "").strip())
    row = None
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            at = cx.execute(
                "SELECT email, expires_at FROM auth_tokens WHERE token_hash=? AND purpose='biofield_reveal'",
                (th,)).fetchone()
            valid = False
            if at is not None:
                try:
                    valid = datetime.fromisoformat(at["expires_at"]) >= datetime.now(timezone.utc)
                except Exception:
                    valid = False
            if valid:
                _br.init_table(cx)
                row = _br.get_by_token_hash(cx, th)
    except Exception as e:
        print(f"[biofield-reveal] {e!r}", flush=True)
        row = None

    if row is None:
        resp = send_from_directory(STATIC, "begin-biofield.html")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["X-Reveal-State"] = "invalid"
        return resp

    # set the biofield gate (idempotent, wrapped) -> Find step 2 fills
    _record_entry_unlock("biofield", row["email"])

    top = row["top"] or {}
    blurred_n = len(row["blurred"] or [])
    slug = (top.get("slug") or "").strip()
    buy_url = f"/begin/buy/{slug}" if slug else "/begin/match"
    payload = {"name": top.get("name", ""), "meaning": top.get("meaning", ""),
               "buy_url": buy_url, "blurred_count": blurred_n}
    html = render_template_string(_BIOFIELD_REVEAL_HTML, data_json=json.dumps(payload))
    resp = app.make_response(html)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp
```

NOTE: the implementer may instead serve `static/begin-biofield.html` and inject `payload` as a `window.__REVEAL__` JSON `<script>` (mirroring the `/begin/explore` `window.__EXPLORE__` injection pattern, grep `__EXPLORE__`) rather than `render_template_string`. Either is fine; the test only requires the top name to appear in the served HTML and the `no-store` header. If using the static-file + injection pattern, read the file, inject the JSON before `</head>`, and return it with the no-store headers. Pick the injection pattern to match `/begin/explore` for consistency, and remove the `_BIOFIELD_REVEAL_HTML`/`render_template_string` approach.

- [ ] **Step 4: Create `static/begin-biofield.html`**

A standalone page modeled on the `begin.html` styling (same `:root` palette / fonts). It must: read `window.__REVEAL__` (the injected JSON), and if present render the **top match free** (the `name` as a heading, the `meaning` line, and a buy button to `buy_url`), then a **blurred stack** of `blurred_count` placeholder cards (CSS `filter: blur(6px)` + a translucent overlay) with the teaser text "and {blurred_count} more in your full analysis" and a disabled/"Unlocking soon" CTA button labeled "Unlock your full Biofield Analysis". If `window.__REVEAL__` is absent (invalid-token case), show a calm "This link is no longer valid. Please request a fresh one." message and NOTHING personal. All dynamic text via `textContent`. No emoji, no em dashes. Include the literal `Unlock your full Biofield Analysis` string.

- [ ] **Step 5: Run to verify pass + the begin sweep**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_routes.py tests/test_biofield_reveals.py tests/test_biofield_reveal_actions.py -v`
Then: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k "begin or biofield" -v`
Expected: all PASS; no regressions.

- [ ] **Step 6: Commit**

```bash
git add app.py static/begin-biofield.html tests/test_biofield_reveal_routes.py
git commit -m "feat: begin #4a token-verified reveal page + biofield gate"
```

Note for the reviewer: manual visual pass required (the blurred stack, the stub CTA, the invalid-token page).

---

## Self-Review

**1. Spec coverage:** ingest (T2); store ai_draft->confirmed, no-overwrite (T1); console edit/approve + magic link + ready email (T3); console list/page/nav (T4); token-verified reveal + top-free/blurred + stub CTA + biofield gate + no-store + friendly invalid page (T5). Privacy (token = ownership) -> T5 verify. ToS-membership: the reveal is reached only via the emailed magic link to the activated email; the gate sets by that email. Deferred 4b ($1 unblur) -> stub CTA in T5. No FMP / no Stripe / no server interpretation -> none added.

**2. Placeholder scan:** No TBD/handle-edge-cases. The two HTML tasks (T4 console, T5 reveal) describe required elements + exact assertions rather than full markup (modeled on existing pages) - acceptable for front-end, with concrete serve-test assertions pinning the contract.

**3. Type consistency:** `upsert_draft/get/list_drafts/get_by_token_hash/set_top/approve` signatures consistent across T1/T2/T3/T4/T5. Row dict `{top, blurred, status, email, ...}` consistent. `configure(base_url, send, hash_token, mint_token)` injected in T3 Step 4 matches the executor's `_DEPS` use. `_record_entry_unlock("biofield", email)` matches #3. `auth_tokens` purpose `biofield_reveal` consistent across T3 (write) and T5 (verify). The reveal never consumes the token (no `consumed_at` write) - consistent with the reopenable-link requirement.
