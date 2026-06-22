# Reveal Outreach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Glen email a reviewed client their reveal link - a per-draft "Send reveal link" button + a "Send all approved un-notified" batch - so silently-seeded drafts can reach clients.

**Architecture:** A `notified_at` column on `biofield_reveals`; a self-contained `_send_reveal_link(rid)` (factored out of `_resend_biofield_reveal`) that mints a fresh token, emails the reveal link, and marks notified only on a successful send; two console actions; the ingest pre-marks `notify=true` drafts.

**Tech Stack:** Python 3.11, Flask, SQLite, vanilla-JS console.

## Global Constraints

- deploy-chat only; PR + merge. No emoji, no em dashes.
- `_send_reveal_link(rid)` does SMTP OUTSIDE the `_db_lock` (mint+commit under the lock, send after, mark-notified under the lock again only if sent).
- Outreach is **approved-only** (the action checks `first_approved`); the resend path is NOT approval-gated.
- `notified_at` is set only on a successful send (a failed send stays retryable).
- Test run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <t> -v`.

---

## File Structure

- `dashboard/biofield_reveals.py`: `notified_at` ALTER, `set_notified`, `list_approved_unnotified`.
- `app.py`: `_send_reveal_link(rid)` (+ refactor `_resend_biofield_reveal`); ingest pre-mark; wire `_bra.configure(send_reveal_link=_send_reveal_link)`.
- `dashboard/biofield_reveal_actions.py`: `_exec_send`, `_exec_send_all`, register both.
- `static/console-biofield-reveals.html`: Send + Send-all buttons.
- Tests: `tests/test_biofield_reveal_send.py`.

---

## Task 1: store - `notified_at` + helpers

**Files:**
- Modify: `dashboard/biofield_reveals.py`
- Test: `tests/test_biofield_reveal_send.py` (create)

**Interfaces:**
- `set_notified(cx, rid)` -> sets `notified_at` = now.
- `list_approved_unnotified(cx, limit=200) -> [row dicts]` (first_approved=1 AND notified_at null/empty).

- [ ] **Step 1: Write the failing test**

Create `tests/test_biofield_reveal_send.py`:

```python
# tests/test_biofield_reveal_send.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load(mod):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"{mod} not importable: {e}")


def test_set_notified_and_list_approved_unnotified(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        r1, _ = br.upsert(cx, "a@x.com", "2026-06-20", {"body": "x"}, [], "s")  # approved, unnotified
        r2, _ = br.upsert(cx, "b@x.com", "2026-06-20", {"body": "x"}, [], "s")  # approved, notified
        r3, _ = br.upsert(cx, "c@x.com", "2026-06-20", {"body": "x"}, [], "s")  # not approved
        br.approve_first(cx, r1, "glen")
        br.approve_first(cx, r2, "glen")
        br.set_notified(cx, r2)
        ids = [r["id"] for r in br.list_approved_unnotified(cx)]
    assert r1 in ids and r2 not in ids and r3 not in ids
    with sqlite3.connect(db) as cx:
        row = br.get(cx, r2)
    assert row["notified_at"]
```

- [ ] **Step 2: Run -> FAIL**

Run: `... -m pytest tests/test_biofield_reveal_send.py::test_set_notified_and_list_approved_unnotified -v`
Expected: FAIL (`set_notified` / `list_approved_unnotified` / `notified_at` absent).

- [ ] **Step 3: Implement** (`dashboard/biofield_reveals.py`)

In `init_table`, add the idempotent ALTER (next to the `dropped` / `layers_json` ones):

```python
    try:
        cx.execute("ALTER TABLE biofield_reveals ADD COLUMN notified_at TEXT")
    except Exception:
        pass
```

Add the helpers:

```python
def set_notified(cx, rid):
    cx.execute("UPDATE biofield_reveals SET notified_at=?, updated_at=? WHERE id=?",
               (_now(), _now(), rid))
    cx.commit()


def list_approved_unnotified(cx, limit=200):
    return [_row(r) for r in _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE first_approved=1 AND (notified_at IS NULL OR notified_at='') "
        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
```

(`_now`, `_row`, `_rows_cursor` already exist in the module.)

- [ ] **Step 4: Run -> PASS**

Run: `... -m pytest tests/test_biofield_reveal_send.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_reveals.py tests/test_biofield_reveal_send.py
git commit -m "feat: biofield_reveals notified_at + list_approved_unnotified"
```

---

## Task 2: `_send_reveal_link` + refactor resend + ingest pre-mark

**Files:**
- Modify: `app.py`
- Test: `tests/test_biofield_reveal_send.py`, plus keep `tests/test_link_resend.py` + `tests/test_biofield_layers.py` green

**Interfaces:** `_send_reveal_link(rid) -> bool` - mint a fresh reveal token, email the link, mark notified only on success. No approval gate.

- [ ] **Step 1: Write the failing tests**

```python
def _app_db(monkeypatch, tmp_path):
    app_module = _load("app")
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens "
                   "(token_hash TEXT, email TEXT, purpose TEXT, extra TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return app_module, db


def test_send_reveal_link_mints_sends_marks(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "s@x.com", "2026-06-20", {"body": "x"},
                           [{"name": "Top", "slug": "top", "meaning": "m"}], "src")
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body: sent.append((to, body)) or True)
    ok = app_module._send_reveal_link(rid)
    assert ok is True and len(sent) == 1 and "/begin/biofield/" in sent[0][1]
    with sqlite3.connect(db) as cx:
        assert br.get(cx, rid)["notified_at"]
        n = cx.execute("SELECT COUNT(*) FROM auth_tokens WHERE email='s@x.com' AND purpose='biofield_reveal'").fetchone()[0]
    assert n == 1


def test_send_reveal_link_failed_send_not_marked(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "f@x.com", "2026-06-20", {"body": "x"}, [], "src")
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: False)
    ok = app_module._send_reveal_link(rid)
    assert ok is False
    with sqlite3.connect(db) as cx:
        assert not br.get(cx, rid)["notified_at"]


def test_ingest_notify_true_marks_notified(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.os.environ.get("CRON_SECRET") or app_module.CONSOLE_SECRET or ""
    if not key: pytest.skip("no secret")
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    c = app_module.app.test_client()
    c.post("/api/e4l/reveal-draft", headers={"X-Console-Key": key},
           json={"email": "n@x.com", "scan_date": "2026-06-20", "interpretation": {"body": "x"},
                 "layers": [{"n": 1, "title": "L", "summary": "s", "patterns": [], "remedy": None}]})
    c.post("/api/e4l/reveal-draft", headers={"X-Console-Key": key},
           json={"email": "q@x.com", "scan_date": "2026-06-20", "interpretation": {"body": "x"},
                 "layers": [{"n": 1, "title": "L", "summary": "s", "patterns": [], "remedy": None}],
                 "notify": False})
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rows = {r["email"]: r["notified_at"] for r in br.list_pending(cx)}
    assert rows.get("n@x.com")          # notify true -> marked
    assert not rows.get("q@x.com")      # notify false -> unmarked
```

- [ ] **Step 2: Run -> FAIL**

Run: `... -m pytest tests/test_biofield_reveal_send.py -k "send_reveal_link or ingest_notify_true" -v`
Expected: FAIL (`_send_reveal_link` undefined; ingest does not mark).

- [ ] **Step 3: Add `_send_reveal_link` + refactor `_resend_biofield_reveal`** (`app.py`)

Replace `_resend_biofield_reveal` (app.py:334) with:

```python
def _send_reveal_link(rid):
    """Mint a fresh reveal token (biofield_reveals.token_hash + auth_tokens), email the
    'ready' link, and mark notified only on a successful send. Returns True if sent.
    SMTP runs outside the db lock. No approval gate (callers decide)."""
    from dashboard import biofield_reveals as _br
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _br.init_table(cx)
        row = cx.execute("SELECT email FROM biofield_reveals WHERE id=?", (rid,)).fetchone()
        if not row or not row[0]:
            return False
        email = row[0]
        tok = secrets.token_urlsafe(32)
        _br.set_token(cx, rid, _hash_token(tok))
        now = datetime.now(timezone.utc)
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
                   "VALUES (?,?,?,?,?)",
                   (_hash_token(tok), email, "biofield_reveal", now.isoformat(),
                    (now + timedelta(days=30)).isoformat()))
        cx.commit()
    url = f"{PUBLIC_BASE_URL}/begin/biofield/{tok}"
    body = ("Aloha,\n\nYour Biofield Analysis is ready. View your reading here:\n"
            f"{url}\n\nIn wellness,\nDr. Glen and Rae\n")
    sent = _send_inquiry_email(email, "Your Biofield Analysis is ready", body)
    if sent:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _br.set_notified(cx, rid)
    return bool(sent)


def _resend_biofield_reveal(email, extra):
    """Resend path (not approval-gated): find the latest reveal for the email and send it."""
    from dashboard import biofield_reveals as _br
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _br.init_table(cx)
        row = cx.execute("SELECT id FROM biofield_reveals WHERE email=? ORDER BY id DESC LIMIT 1",
                         (email,)).fetchone()
    if not row:
        return
    _send_reveal_link(row[0])
```

(`set_notified` commits internally.)

- [ ] **Step 4: Ingest pre-mark** (`app.py`, the `if is_new and notify:` block ~11024)

Change the send line so a successful send marks notified:

```python
        if is_new and notify:
            try:
                url = f"{PUBLIC_BASE_URL}/begin/biofield/{token}"
                body = ("Aloha,\n\nYour Biofield Analysis is ready. View your reading here:\n"
                        f"{url}\n\nIn wellness,\nDr. Glen and Rae\n")
                if _send_inquiry_email(email, "Your Biofield Analysis is ready", body):
                    with _db_lock, sqlite3.connect(LOG_DB) as cx:
                        _br.set_notified(cx, rid)
            except Exception as e:
                print(f"[reveal-draft] notify failed: {e!r}", flush=True)
```

(Use the existing `rid`, `token`, `email` from the surrounding ingest scope; keep the existing copy. `_br` is already imported in the route.)

- [ ] **Step 5: Run -> PASS** (new tests + `tests/test_link_resend.py` for the resend refactor)

Run: `... -m pytest tests/test_biofield_reveal_send.py tests/test_link_resend.py -v`
Expected: PASS (resend still mints + sends; ingest marks correctly).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_biofield_reveal_send.py
git commit -m "feat: _send_reveal_link (shared) + ingest pre-marks notify=true"
```

---

## Task 3: console actions - send + send_all

**Files:**
- Modify: `dashboard/biofield_reveal_actions.py`, `app.py` (wire `configure`)
- Test: `tests/test_biofield_reveal_send.py`

**Interfaces:** `biofield_reveal.send {id}` -> `{sent, reason?}`; `biofield_reveal.send_all` -> `{sent, of}`.

- [ ] **Step 1: Write the failing tests** (dispatch-spine style; reuse the events/CONSOLE_SECRET setup)

```python
def _spine_db(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sek", raising=False)
    from dashboard import events as _ev
    with sqlite3.connect(db) as cx:
        _ev.init_event_tables(cx)
        cx.commit()
    return app_module, db


def test_send_action_approved_only(monkeypatch, tmp_path):
    app_module, db = _spine_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "a@x.com", "2026-06-20", {"body": "x"}, [], "s")  # not approved
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    c = app_module.app.test_client()
    r = c.post("/api/action/biofield_reveal.send", json={"id": rid}, headers={"X-Console-Key": "sek"})
    assert r.get_json()["result"]["sent"] is False           # unapproved -> not sent
    with sqlite3.connect(db) as cx:
        br.approve_first(cx, rid, "glen")
    r2 = c.post("/api/action/biofield_reveal.send", json={"id": rid}, headers={"X-Console-Key": "sek"})
    assert r2.get_json()["result"]["sent"] is True
    with sqlite3.connect(db) as cx:
        assert br.get(cx, rid)["notified_at"]


def test_send_all_batches_approved_unnotified(monkeypatch, tmp_path):
    app_module, db = _spine_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        a, _ = br.upsert(cx, "a@x.com", "2026-06-20", {"body": "x"}, [], "s")
        b, _ = br.upsert(cx, "b@x.com", "2026-06-20", {"body": "x"}, [], "s")
        c_, _ = br.upsert(cx, "c@x.com", "2026-06-20", {"body": "x"}, [], "s")  # stays unapproved
        br.approve_first(cx, a, "glen"); br.approve_first(cx, b, "glen")
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda to, *a, **k: sent.append(to) or True)
    r = app_module.app.test_client().post("/api/action/biofield_reveal.send_all",
                                          json={}, headers={"X-Console-Key": "sek"})
    res = r.get_json()["result"]
    assert res["sent"] == 2 and res["of"] == 2
    assert set(sent) == {"a@x.com", "b@x.com"}
```

- [ ] **Step 2: Run -> FAIL**

Run: `... -m pytest tests/test_biofield_reveal_send.py -k "send_action or send_all" -v`
Expected: FAIL (actions not registered).

- [ ] **Step 3: Add the executors + registration** (`dashboard/biofield_reveal_actions.py`, after `_exec_approve`)

```python
def _exec_send(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    rev = _br.get(ctx["cx"], rid)
    if not rev or not rev.get("first_approved"):
        return {"sent": False, "reason": "not_approved"}
    send = _DEPS.get("send_reveal_link")
    return {"sent": bool(send and send(rid))}


def _exec_send_all(params, ctx):
    rows = _br.list_approved_unnotified(ctx["cx"], limit=50)
    send = _DEPS.get("send_reveal_link")
    n = 0
    for r in rows:
        try:
            if send and send(r["id"]):
                n += 1
        except Exception as e:
            print(f"[reveal-send-all] {r.get('id')}: {e!r}", flush=True)
    return {"sent": n, "of": len(rows)}
```

In `register()`, after the `biofield_reveal.delete` registration:

```python
    register_action(Action(
        key="biofield_reveal.send", module="biofield_reveal", title="Send reveal link",
        description="Email an approved reveal's client their magic link.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_send))
    register_action(Action(
        key="biofield_reveal.send_all", module="biofield_reveal", title="Send all approved un-notified",
        description="Email every approved, not-yet-notified client their reveal link.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_send_all))
```

- [ ] **Step 4: Wire `_send_reveal_link` into the action module** (`app.py`, where `biofield_reveal_actions` is imported/registered ~21150)

After `from dashboard import biofield_reveal_actions as _bra` and its `register()`, add:

```python
_bra.configure(send_reveal_link=_send_reveal_link)
```

(Place it after `_send_reveal_link` is defined and after `_bra` is imported; `configure` just stores it in `_DEPS`.)

- [ ] **Step 5: Run -> PASS**

Run: `... -m pytest tests/test_biofield_reveal_send.py -v`
Expected: PASS (all store + send + action tests).

- [ ] **Step 6: Commit**

```bash
git add dashboard/biofield_reveal_actions.py app.py tests/test_biofield_reveal_send.py
git commit -m "feat: biofield_reveal.send + send_all console actions"
```

---

## Task 4: console UI - Send + Send-all buttons

**Files:**
- Modify: `static/console-biofield-reveals.html`
- Test: `tests/test_biofield_reveal_send.py` (serve assertion)

- [ ] **Step 1: Add the failing serve test**

```python
def test_console_page_ships_send_controls(monkeypatch, tmp_path):
    app_module, _ = _spine_db(monkeypatch, tmp_path)
    html = app_module.app.test_client().get(
        "/console/biofield-reveals", headers={"X-Console-Key": "sek"}).data.decode()
    assert "biofield_reveal.send_all" in html
    assert "Send reveal link" in html
```

- [ ] **Step 2: Run -> FAIL**

Run: `... -m pytest tests/test_biofield_reveal_send.py::test_console_page_ships_send_controls -v`
Expected: FAIL.

- [ ] **Step 3: Add the Send-all button** (near the top of the list render / `loadList`, e.g. just before rendering the Approved section header)

```javascript
  if(approved.length){
    var sendAllBtn = document.createElement('button');
    sendAllBtn.className = 'btn';
    sendAllBtn.textContent = 'Send all approved un-notified';
    sendAllBtn.onclick = doSendAll;
    el.appendChild(sendAllBtn);
    var apprHead = document.createElement('div');
    ...
```

- [ ] **Step 4: Add the per-card Send button** (in `buildCard`, only for approved cards - i.e. when `d.first_approved`) next to the Save/Approve buttons:

```javascript
  if(d.first_approved){
    var sendBtn = document.createElement('button');
    sendBtn.className = 'btn';
    sendBtn.textContent = d.notified_at ? ('Sent ' + d.notified_at.slice(0,10) + ' (re-send)') : 'Send reveal link';
    sendBtn.onclick = function(){ doSend(wrap, d.id, sendBtn); };
    row.appendChild(sendBtn);
  }
```

- [ ] **Step 5: Add `doSend` / `doSendAll`** (next to `doApprove`)

```javascript
async function doSend(card, id, btn){
  var statusEl = card.querySelector('.status');
  statusEl.textContent = 'Sending...';
  btn.disabled = true;
  var r = await api('POST', '/api/action/biofield_reveal.send', { id: id });
  if(r.ok && r.json.status === 'done' && r.json.result && r.json.result.sent){
    statusEl.textContent = 'Reveal link sent.';
    setTimeout(loadList, 700);
  } else {
    btn.disabled = false;
    statusEl.textContent = 'Not sent' + (r.json && r.json.result && r.json.result.reason ? ' (' + r.json.result.reason + ')' : '') + '.';
  }
}

async function doSendAll(){
  if(!window.confirm('Email every approved, not-yet-notified client their reveal link?')) return;
  var r = await api('POST', '/api/action/biofield_reveal.send_all', {});
  if(r.ok && r.json.status === 'done'){
    alert('Sent ' + r.json.result.sent + ' of ' + r.json.result.of + '.');
    loadList();
  } else {
    alert('Send-all failed (HTTP ' + r.status + ').');
  }
}
```

- [ ] **Step 6: Run -> PASS**

Run: `... -m pytest tests/test_biofield_reveal_send.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add static/console-biofield-reveals.html tests/test_biofield_reveal_send.py
git commit -m "feat: console Send / Send-all reveal-link buttons"
```

---

## Verification

- Per task: the named pytest target passes; then `tests/test_biofield_reveal_send.py tests/test_link_resend.py tests/test_biofield_layers.py` for no regression.
- Final whole-branch review (most capable model). Focus: SMTP outside `_db_lock`; `notified_at` only on successful send; outreach approved-only while resend is not gated; `send_all` skips notified/unapproved + isolates errors + caps; ingest marks notify=true only; resend refactor behavior-preserved; XSS-safe; no emoji/em-dash.
- Ship via PR + merge to main (auto-deploys); gentle `/console/biofield-reveals` probe.
- **Operational:** after deploy, Send-all (or per-draft Send) the approved subset of the 15 seeded drafts to begin outreach.

## Build order

Task 1 (store) -> Task 2 (send fn + ingest) -> Task 3 (actions) -> Task 4 (UI). Backend fully testable before the UI.
