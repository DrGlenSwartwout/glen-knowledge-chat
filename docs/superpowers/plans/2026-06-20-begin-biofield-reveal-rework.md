# Begin #4a Rework — interpretation auto-shows; only remedies gated — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. This REWORKS the already-built #4a (commits 9d3367d..54f717a on branch sess/47eb2a23-biofield-reveal) to the rev-2 spec.

**Goal:** Rework the Biofield reveal so the interpretation shows automatically and only the matched remedies are gated: the draft carries interpretation + ranked remedies; the "ready" email + magic link are sent at ingest; the reveal shows the interpretation always with all remedies blurred; Glen's console approve un-blurs the top remedy; the rest wait for 4b.

**Architecture:** Reshape the `biofield_reveals` store (interpretation_json + remedies_json + first_approved + token_hash minted at ingest). Move token-mint + ready-email from approve to the ingest endpoint (first insert only). Console approve flips `first_approved` (no email). The reveal route shows the row regardless of approval; the page shows the interpretation always and un-blurs remedies[0] only when approved.

**Tech Stack:** Flask, SQLite, `auth_tokens` magic link, the dispatch spine, pytest.

## Global Constraints

- No emoji, no em dashes. Live reveal route, no flag; console staff-gated. `main` auto-deploys.
- Un-widened ingest auth: `expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET","")`. Do not modify any auth check to satisfy a test (set `CRON_SECRET`).
- Token NOT consumed on view (reopenable, 30-day TTL). Stored reveal shown ONLY via the token. Invalid token -> friendly page, NO personal data (`window.__REVEAL__ = null`).
- `window.__REVEAL__` injection escapes `<`,`>`,`&` (keep the rev-1 fix).
- Ingest always 200 on a valid push; token-mint + email are best-effort and never fail the 200; content updates only while `first_approved=0`; never overwrites after approval; token minted + email sent EXACTLY once (first insert).
- Store getters use a per-cursor Row factory (no connection-state leak) - keep this pattern.
- `biofield` gate via `_record_entry_unlock("biofield", email)` on view; no new gate.
- XSS-safe: dynamic text via `textContent`. Copy provisional.
- Test harness as in the prior plan (tmp LOG_DB; init table + auth_tokens + journey tables; mock send; mock GHL). Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Spec: `docs/superpowers/specs/2026-06-19-begin-biofield-reveal-design.md`.

---

### Task 1: Reshape the store + rework ingest (token + email at ingest)

**Files:** Modify `dashboard/biofield_reveals.py`; Modify `app.py` (`api_e4l_reveal_draft`); Rewrite `tests/test_biofield_reveals.py`; update the ingest tests in `tests/test_biofield_reveal_routes.py`.

**Interfaces produced:** `init_table(cx)`; `upsert(cx, email, scan_date, interpretation, remedies, source) -> (id, is_new)`; `set_token(cx, id, token_hash)`; `set_interpretation(cx, id, interpretation)`; `set_remedies(cx, id, remedies)`; `approve_first(cx, id, by) -> bool`; `list_pending(cx) -> [dict]`; `get(cx, id)`; `get_by_token_hash(cx, th)`. Row dict keys: `id, email, scan_date, interpretation, remedies, first_approved (bool), approved_at, approved_by, created_at, updated_at`.

- [ ] **Step 1: Replace `tests/test_biofield_reveals.py`** with the rev-2 store tests:

```python
"""Begin #4a rev - biofield_reveals store: interpretation + remedies + first_approved."""
import sqlite3, sys
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


def _interp():
    return {"greeting": "Aloha", "body": "Your terrain reading."}


def _remedies():
    return [{"name": "Cistus Shield", "slug": "cistus-shield", "meaning": "Calm the terrain."},
            {"name": "Binder", "slug": "binder", "meaning": "Bind and clear."}]


def test_upsert_new_then_update(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid, is_new = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    assert is_new is True
    row = m.get(cx, rid)
    assert row["interpretation"]["greeting"] == "Aloha"
    assert len(row["remedies"]) == 2 and row["first_approved"] is False
    rid2, is_new2 = m.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Hi", "body": "v2"}, _remedies(), "s")
    assert rid2 == rid and is_new2 is False
    assert m.get(cx, rid)["interpretation"]["greeting"] == "Hi"


def test_no_overwrite_after_approval(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid, _ = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    m.approve_first(cx, rid, "glen")
    m.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "X", "body": "X"}, [], "s")
    row = m.get(cx, rid)
    assert row["first_approved"] is True
    assert row["interpretation"]["greeting"] == "Aloha"  # unchanged after approval


def test_approve_first_and_list_pending(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    r1, _ = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    r2, _ = m.upsert(cx, "b@x.com", "2026-06-19", _interp(), _remedies(), "s")
    assert m.approve_first(cx, r1, "glen") is True
    pending = m.list_pending(cx)
    assert [p["id"] for p in pending] == [r2]
    assert m.get(cx, r1)["approved_by"] == "glen"


def test_token_lookup(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid, _ = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    m.set_token(cx, rid, "H:tok")
    assert m.get_by_token_hash(cx, "H:tok")["id"] == rid


def test_edit_interpretation_and_remedies(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    rid, _ = m.upsert(cx, "a@x.com", "2026-06-19", _interp(), _remedies(), "s")
    m.set_interpretation(cx, rid, {"greeting": "Edited", "body": "new"})
    m.set_remedies(cx, rid, [{"name": "Only", "slug": "only", "meaning": "m"}])
    row = m.get(cx, rid)
    assert row["interpretation"]["greeting"] == "Edited" and len(row["remedies"]) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveals.py -v`
Expected: FAIL (new API).

- [ ] **Step 3: Rewrite `dashboard/biofield_reveals.py`**

```python
"""Begin #4a store: per-scan funnel Biofield reveal (interpretation auto-shown +
ranked remedies, blurred until the top is approved). Distinct from
portal_biofield_reports."""
import json
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _rows_cursor(cx):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    return cur


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS biofield_reveals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            interpretation_json TEXT NOT NULL DEFAULT '{}',
            remedies_json TEXT NOT NULL DEFAULT '[]',
            first_approved INTEGER NOT NULL DEFAULT 0,
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
    d["interpretation"] = json.loads(d.pop("interpretation_json") or "{}")
    d["remedies"] = json.loads(d.pop("remedies_json") or "[]")
    d["first_approved"] = bool(d.get("first_approved"))
    return d


def upsert(cx, email, scan_date, interpretation, remedies, source):
    """Insert or update a reveal. Content updates only while not yet approved.
    Returns (id, is_new) - is_new True only on first insert (caller mints token +
    emails exactly once)."""
    email = (email or "").strip().lower()
    now = _now()
    existing = cx.execute(
        "SELECT id, first_approved FROM biofield_reveals WHERE email=? AND scan_date=?",
        (email, scan_date)).fetchone()
    if existing is not None:
        rid, approved = existing[0], existing[1]
        if not approved:
            cx.execute(
                "UPDATE biofield_reveals SET interpretation_json=?, remedies_json=?, updated_at=? WHERE id=?",
                (json.dumps(interpretation or {}), json.dumps(remedies or []), now, rid))
            cx.commit()
        return rid, False
    cur = cx.execute(
        "INSERT INTO biofield_reveals (email, scan_date, interpretation_json, remedies_json, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        (email, scan_date, json.dumps(interpretation or {}), json.dumps(remedies or []), now, now))
    cx.commit()
    return cur.lastrowid, True


def set_token(cx, rid, token_hash):
    cx.execute("UPDATE biofield_reveals SET token_hash=?, updated_at=? WHERE id=?",
               (token_hash, _now(), rid))
    cx.commit()


def set_interpretation(cx, rid, interpretation):
    cx.execute("UPDATE biofield_reveals SET interpretation_json=?, updated_at=? WHERE id=? AND first_approved=0",
               (json.dumps(interpretation or {}), _now(), rid))
    cx.commit()


def set_remedies(cx, rid, remedies):
    cx.execute("UPDATE biofield_reveals SET remedies_json=?, updated_at=? WHERE id=? AND first_approved=0",
               (json.dumps(remedies or []), _now(), rid))
    cx.commit()


def approve_first(cx, rid, by):
    now = _now()
    cur = cx.execute(
        "UPDATE biofield_reveals SET first_approved=1, approved_at=?, approved_by=?, updated_at=? WHERE id=?",
        (now, by, now, rid))
    cx.commit()
    return cur.rowcount == 1


def list_pending(cx):
    rows = _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE first_approved=0 ORDER BY id DESC").fetchall()
    return [_row(r) for r in rows]


def get(cx, rid):
    return _row(_rows_cursor(cx).execute("SELECT * FROM biofield_reveals WHERE id=?", (rid,)).fetchone())


def get_by_token_hash(cx, th):
    return _row(_rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE token_hash=?", (th,)).fetchone())
```

- [ ] **Step 4: Rework the ingest endpoint** in `app.py` (`api_e4l_reveal_draft`). Replace the body so it accepts interpretation+remedies, upserts, and on a NEW row mints the token + auth_tokens row + sends the ready email:

```python
@app.route("/api/e4l/reveal-draft", methods=["POST"])
def api_e4l_reveal_draft():
    """Ingest a Biofield reveal draft (interpretation + ranked remedies). On the
    first insert, mint the magic link, store the auth_tokens row, and email the
    owner. Auth: X-Cron-Secret (== CRON_SECRET, falls back to CONSOLE_SECRET)."""
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    scan_date = (data.get("scan_date") or "").strip()
    interp = data.get("interpretation") or {}
    remedies = data.get("remedies") or []
    if not email or not scan_date or not (interp or remedies):
        return jsonify({"error": "email, scan_date, and interpretation or remedies required"}), 400
    from dashboard import biofield_reveals as _br
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _br.init_table(cx)
            rid, is_new = _br.upsert(cx, email, scan_date, interp, remedies, (data.get("source") or "").strip())
            if is_new:
                token = secrets.token_urlsafe(32)
                _br.set_token(cx, rid, _hash_token(token))
                now = datetime.now(timezone.utc)
                cx.execute(
                    "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                    (_hash_token(token), email, "biofield_reveal", now.isoformat(),
                     (now + timedelta(days=30)).isoformat()))
                cx.commit()
        if is_new:
            try:
                url = f"{PUBLIC_BASE_URL}/begin/biofield/{token}"
                body = ("Aloha,\n\nYour Biofield Analysis is ready. View your reading here:\n"
                        f"{url}\n\nIn wellness,\nDr. Glen and Rae\n")
                _send_inquiry_email(email, "Your Biofield Analysis is ready", body)
            except Exception as e:
                print(f"[reveal-draft] notify failed: {e!r}", flush=True)
        return jsonify({"ok": True, "id": rid})
    except Exception as e:
        print(f"[reveal-draft] {e!r}", flush=True)
        return jsonify({"ok": False, "error": "store failed"}), 500
```

- [ ] **Step 5: Update the ingest tests** in `tests/test_biofield_reveal_routes.py`. Replace `test_ingest_stores_draft` body to push `{interpretation, remedies}` and assert the row is stored via `list_pending`, an `auth_tokens` row with purpose `biofield_reveal` exists, and a re-push does NOT add a second auth_tokens row. Keep `test_ingest_requires_auth` and update `test_ingest_missing_email_400` to send `{scan_date, interpretation:{...}}` without email -> 400. (Mock `_send_inquiry_email` via `monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)`.)

```python
def test_ingest_stores_draft_and_mints_token_once(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    client = app_module.app.test_client()
    payload = {"email": "a@x.com", "scan_date": "2026-06-19",
               "interpretation": {"greeting": "Aloha", "body": "reading"},
               "remedies": [{"name": "Cistus", "slug": "cistus", "meaning": "calm"}], "source": "m"}
    r = client.post("/api/e4l/reveal-draft", json=payload, headers={"X-Cron-Secret": "k"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    r2 = client.post("/api/e4l/reveal-draft", json=payload, headers={"X-Cron-Secret": "k"})
    assert r2.status_code == 200
    from dashboard import biofield_reveals
    with sqlite3.connect(db) as cx:
        assert len(biofield_reveals.list_pending(cx)) == 1
        n = cx.execute("SELECT COUNT(*) FROM auth_tokens WHERE email='a@x.com' AND purpose='biofield_reveal'").fetchone()[0]
    assert n == 1  # token minted once
```

(Ensure `_fresh` inits `auth_tokens` - add `cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")` to the `_fresh` helper if not present; the real app also creates it at import.)

- [ ] **Step 6: Run + commit**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveals.py tests/test_biofield_reveal_routes.py::test_ingest_stores_draft_and_mints_token_once tests/test_biofield_reveal_routes.py::test_ingest_requires_auth tests/test_biofield_reveal_routes.py::test_ingest_missing_email_400 -v`
Expected: PASS.

```bash
git add dashboard/biofield_reveals.py app.py tests/test_biofield_reveals.py tests/test_biofield_reveal_routes.py
git commit -m "feat: begin #4a rework store+ingest (interpretation+remedies, token+email at ingest)"
```

---

### Task 2: Rework console actions + page

**Files:** Modify `dashboard/biofield_reveal_actions.py`; Modify `static/console-biofield-reveals.html`; Rewrite `tests/test_biofield_reveal_actions.py`.

**Interfaces:** Consumes the Task-1 store. `biofield_reveal.approve` -> `approve_first` (no email); `biofield_reveal.edit` -> `set_interpretation` + `set_remedies`.

- [ ] **Step 1: Replace `tests/test_biofield_reveal_actions.py`**

```python
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
    from dashboard import biofield_reveals
    biofield_reveals.init_table(cx)
    return cx


class _Actor:
    name = "glen"


def test_approve_flips_first_approved_no_email(tmp_path):
    br, acts = _mods(); cx = _cx(tmp_path)
    rid, _ = br.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Hi"}, [{"name": "C"}], "s")
    sent = []
    acts.configure(send=lambda *a, **k: sent.append(a))
    acts._exec_approve({"id": rid}, {"cx": cx, "actor": _Actor()})
    assert br.get(cx, rid)["first_approved"] is True
    assert sent == []  # no email on approve (it went out at ingest)


def test_edit_updates_interpretation_and_remedies(tmp_path):
    br, acts = _mods(); cx = _cx(tmp_path)
    rid, _ = br.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Old"}, [{"name": "A"}], "s")
    acts._exec_edit({"id": rid, "greeting": "New", "body": "b",
                     "remedies": [{"name": "B", "slug": "b", "meaning": "m"}]},
                    {"cx": cx, "actor": _Actor()})
    row = br.get(cx, rid)
    assert row["interpretation"]["greeting"] == "New" and row["remedies"][0]["name"] == "B"
    assert row["first_approved"] is False
```

- [ ] **Step 2: Run to verify fail.** `... -m pytest tests/test_biofield_reveal_actions.py -v` -> FAIL.

- [ ] **Step 3: Rewrite `dashboard/biofield_reveal_actions.py`**

```python
"""Begin #4a console actions: edit interpretation/remedies; approve = un-blur the
top remedy (first_approved=1). The ready email already went out at ingest, so
approve sends nothing. Registered on the Business-OS dispatch spine."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import biofield_reveals as _br

_DEPS = {}


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
    interp = dict(cur["interpretation"])
    if "greeting" in params:
        interp["greeting"] = (params.get("greeting") or "").strip()
    if "body" in params:
        interp["body"] = (params.get("body") or "").strip()
    _br.set_interpretation(ctx["cx"], rid, interp)
    if isinstance(params.get("remedies"), list):
        _br.set_remedies(ctx["cx"], rid, params["remedies"])
    return {"ok": True}


def _exec_approve(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    ok = _br.approve_first(ctx["cx"], rid, _actor_name(ctx.get("actor")))
    return {"ok": bool(ok)}


def register():
    if get_action("biofield_reveal.approve"):
        return
    register_action(Action(
        key="biofield_reveal.edit", module="biofield_reveal", title="Edit Biofield reveal",
        description="Edit the interpretation and/or ranked remedies (stays pending).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="biofield_reveal.approve", module="biofield_reveal", title="Approve top remedy",
        description="Un-blur the top remedy for the visitor (the rest unlock via the $1 trial).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
```

- [ ] **Step 4: Update the app.py configure call.** The `_bra.configure(...)` in app.py (grep `_bra.configure`) passed `base_url`/`send`/`hash_token`/`mint_token` (rev-1, used by approve). Approve no longer needs them; simplify to `_bra.configure()` or leave the call (the deps are now unused by the executors - harmless). Minimal: change it to `_bra.configure()` and keep `_bra.register()`.

- [ ] **Step 5: Update `static/console-biofield-reveals.html`.** It must now render, per pending reveal: the editable interpretation (greeting + body) and the editable ranked remedies (the FIRST is labeled as the one that un-blurs on approve); a "Save edit" button posting `/api/action/biofield_reveal.edit` `{id, greeting, body, remedies}`; an "Approve top remedy" button posting `/api/action/biofield_reveal.approve` `{id}`. List from `GET /api/console/biofield-reveals` (which now returns `list_pending`). All text via textContent. Keep the heading "Biofield Reveals". No emoji/em dash.

- [ ] **Step 6: Run + commit**

Run: `... -m pytest tests/test_biofield_reveal_actions.py tests/test_biofield_reveal_routes.py -v` -> PASS (note: the console list test now lists pending rows; ensure `/api/console/biofield-reveals` uses `list_pending` - update that route in app.py if it still calls `list_drafts`, which was renamed to `list_pending`).

```bash
git add dashboard/biofield_reveal_actions.py app.py static/console-biofield-reveals.html tests/test_biofield_reveal_actions.py tests/test_biofield_reveal_routes.py
git commit -m "feat: begin #4a rework console (edit interpretation/remedies; approve un-blurs top)"
```

---

### Task 3: Rework reveal route + page

**Files:** Modify `app.py` (`begin_biofield_reveal`); Modify `static/begin-biofield.html`; update reveal tests in `tests/test_biofield_reveal_routes.py`.

**Interfaces:** Consumes `get_by_token_hash`. The route resolves the row regardless of `first_approved`; the page shows the interpretation always and un-blurs remedies[0] only when `first_approved`.

- [ ] **Step 1: Update the reveal tests** in `tests/test_biofield_reveal_routes.py`. Rework `_approve_a_reveal` -> a helper that creates a reveal via `upsert` + sets a token + (optionally) `approve_first`, returning the plaintext token. Tests:

```python
def _make_reveal(app_module, db, email="a@x.com", approved=False):
    import secrets as _s
    from dashboard import biofield_reveals
    token = "tkn_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        rid, _ = biofield_reveals.upsert(cx, email, "2026-06-19",
                 {"greeting": "Aloha", "body": "Your terrain reading."},
                 [{"name": "Cistus Shield", "slug": "cistus-shield", "meaning": "Calm."},
                  {"name": "Binder", "slug": "binder", "meaning": "Clear."}], "s")
        biofield_reveals.set_token(cx, rid, th)
        if approved:
            biofield_reveals.approve_first(cx, rid, "glen")
        from datetime import datetime, timezone, timedelta
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def test_reveal_shows_interpretation_pending_all_blurred(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    token = _make_reveal(app_module, db, approved=False)
    r = app_module.app.test_client().get(f"/begin/biofield/{token}")
    assert r.status_code == 200
    assert b"terrain reading" in r.data              # interpretation auto-shows
    assert b"\\u0022first_approved\\u0022: false" in r.data or b'"first_approved": false' in r.data
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="a@x.com")
    assert "biofield" in st["unlocked_gates"]


def test_reveal_approved_top_unblurred(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    token = _make_reveal(app_module, db, approved=True)
    r = app_module.app.test_client().get(f"/begin/biofield/{token}")
    assert r.status_code == 200
    assert b"Cistus Shield" in r.data
    assert "no-store" in r.headers.get("Cache-Control", "")


def test_reveal_invalid_token_friendly_no_pii(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    r = app_module.app.test_client().get("/begin/biofield/bogus")
    assert r.status_code == 200
    assert b"terrain reading" not in r.data and b"Cistus" not in r.data
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="a@x.com")
    assert "biofield" not in (st["unlocked_gates"] or [])
```

(The `first_approved` JSON assertion: the route injects the full reveal payload incl. `first_approved`; assert the boolean is present. Adjust the exact bytes to match the escaped injection - the implementer should make the test match the real output, e.g. assert `b"first_approved" in r.data` plus `b"Cistus Shield" not in r.data` for the pending case to prove the top is NOT revealed when pending.)

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Rework the reveal route** so the payload carries the interpretation + the remedies + `first_approved`, and resolves the row for ANY status (not only confirmed). Replace the payload-building part of `begin_biofield_reveal`:

```python
    top = (row["remedies"] or [{}])[0] if row["remedies"] else {}
    slug = (top.get("slug") or "").strip()
    payload = {
        "interpretation": row["interpretation"] or {},
        "remedies": row["remedies"] or [],
        "first_approved": bool(row["first_approved"]),
        "top_buy_url": (f"/begin/buy/{slug}" if slug else "/begin/match"),
    }
    _safe = (json.dumps(payload).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
    injection = f"<script>window.__REVEAL__ = {_safe};</script>"
```

Keep the token verify (purpose `biofield_reveal`, not expired, NOT consumed), the `get_by_token_hash` resolution, the `_record_entry_unlock("biofield", row["email"])` gate-set, the no-store headers, and the invalid-token `window.__REVEAL__ = null` branch - all unchanged from the current route except the payload shape and that the row no longer needs `status='confirmed'` (the store's `get_by_token_hash` already returns the row regardless).

- [ ] **Step 4: Rework `static/begin-biofield.html`.** Read `window.__REVEAL__`. If present: render the **interpretation** (greeting heading + body) ALWAYS; then the remedies: if `first_approved`, show `remedies[0]` un-blurred (name + meaning + a buy button to `top_buy_url`) and the remaining remedies blurred (CSS blur) with "+N more in your full analysis" + the disabled "Unlock your full Biofield Analysis" CTA; if NOT `first_approved`, render ALL remedies blurred with a calm "Your top match is being finalized." note (still show the blurred stack + the stub CTA). If `window.__REVEAL__` is null: the friendly "this link is no longer valid" message, nothing personal. All dynamic text via textContent. Keep the literal "Unlock your full Biofield Analysis". No emoji/em dash.

- [ ] **Step 5: Run the focused tests + the full biofield + begin sweep**

Run: `... -m pytest tests/test_biofield_reveal_routes.py tests/test_biofield_reveals.py tests/test_biofield_reveal_actions.py -v`
Then: `... -m pytest tests/ -k "begin or biofield" -v`
Expected: all PASS; no regressions.

- [ ] **Step 6: Commit**

```bash
git add app.py static/begin-biofield.html tests/test_biofield_reveal_routes.py
git commit -m "feat: begin #4a rework reveal (interpretation always shown; top un-blurs when approved)"
```

---

## Self-Review

**1. Spec coverage (rev 2):** interpretation auto-shows -> T3 (route payload + page always renders it). Remedies blurred, top un-blurs on approve -> T2 (approve=approve_first) + T3 (page un-blurs remedies[0] iff first_approved). Email at ingest, once -> T1 (is_new mints token + sends once; re-push no re-send). No FMP/Stripe/server-interpretation -> none added. Privacy (token only) + no-PII invalid + token not consumed -> T3. Gate on view -> T3. Un-widened auth -> T1 (kept). `__REVEAL__` escaping -> T3 (kept).

**2. Placeholder scan:** No TBD. The two HTML reworks (T2 console, T3 reveal) describe required elements + pin them via the route/serve test assertions.

**3. Type consistency:** `upsert(... ) -> (id, is_new)`; `approve_first(cx,id,by)`; `set_interpretation`/`set_remedies`; `list_pending`; `get`/`get_by_token_hash` consistent across T1/T2/T3 and the routes. Payload keys `interpretation/remedies/first_approved/top_buy_url` consistent between the route (T3 Step 3) and the page (T3 Step 4). The console list route uses `list_pending` (renamed from `list_drafts` - T2 Step 6 fixes the route). `configure()` no longer needs send/token deps (T2 Step 4).
