# Magic-Link Resend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "Request a fresh link" button on every invalid magic-link page, backed by one `POST /link/resend` endpoint that re-mints and re-sends the right link from the expired token (or inquiry/practitioner ids).

**Architecture:** `/link/resend` branches on the body: `{token}` -> an `auth_tokens` purpose registry (generic mint-and-send, reveal is custom); `{inquiry_id, practitioner_id}` -> the inquiry-reply branch (its own `inquiry_reply_tokens` table + practitioner email). Always a generic ok. The same button is wired onto each invalid surface.

**Tech Stack:** Python 3.11, Flask, SQLite, vanilla-JS / inline-HTML front-end.

## Global Constraints

- deploy-chat only; PR + merge. No emoji, no em dashes.
- **Generic-ok contract:** every path returns `200 {"ok": true}` with the same message - found-or-not, sent-or-not. No enumeration, never an error to the user.
- Best-effort: a sender or DB failure is caught and still returns ok (logged).
- Re-minted tokens **preserve the original `extra`** (practitioner tokens need `practitioner_id` in `extra`).
- `portal` is NOT included (it is `portal_identity`-managed, a separate token system, and has no dead-end page in scope). `membership_cancel` is registered but has no page to wire.
- Registry purposes + TTL + URL (pinned from mint sites):
  - `biofield_reveal` 30 d `/begin/biofield/{token}` (CUSTOM)
  - `reorder` 1440 min `/reorder/auth/{token}`
  - `magic_link` 1440 min `/auth/magic-link/verify?token={token}`
  - `affiliate_magic_link` 1440 min `/affiliate/login-verify?token={token}`
  - `practitioner_claim` 7 d `/practitioner-claim/{token}`
  - `practitioner_optout` 365 d `/practitioner-optout/{token}`
  - `practitioner_share` 30 d `/share-with-practitioner/{token}`
  - inquiry-reply 30 d `/inquiries/{inquiry_id}/{practitioner_id}/reply?token={token}` (CUSTOM branch)
- Test run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_link_resend.py -v`.

---

## File Structure

- `app.py`: `_link_resend_generic` factory, `_resend_biofield_reveal`, `RESEND_HANDLERS`, `_resend_inquiry_reply`, and the `POST /link/resend` route.
- `static/begin-biofield.html`, `static/coaching.html`, `static/practitioner-claim.html`, `static/practitioner-optout.html`, `static/practitioner-share.html`, `static/inquiry-reply.html`: the button.
- The inline-HTML invalid responses in the reorder route (~app.py:9562) and the sign-in route (~app.py:10976): the button markup.
- Test: `tests/test_link_resend.py`.

---

## Task 1: `/link/resend` endpoint + auth_tokens registry

**Files:**
- Modify: `app.py`
- Test: `tests/test_link_resend.py` (create)

**Interfaces:**
- `RESEND_HANDLERS: dict[str, callable]` - `purpose -> handler(email, extra)`.
- `POST /link/resend` - body `{token}` or `{inquiry_id, practitioner_id}`; returns `{"ok": true, "message": <generic>}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_link_resend.py`:

```python
# tests/test_link_resend.py
import importlib, sqlite3, sys
from datetime import datetime, timezone, timedelta
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
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens "
                   "(token_hash TEXT, email TEXT, purpose TEXT, extra TEXT, "
                   "created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return db


def _seed_token(app_module, db, purpose, email="u@x.com", extra=None, expired=True):
    import secrets, json
    tok = "tk_" + secrets.token_urlsafe(8)
    now = datetime.now(timezone.utc)
    exp = now - timedelta(days=1) if expired else now + timedelta(days=1)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
                   "VALUES (?,?,?,?,?,?)",
                   (app_module._hash_token(tok), email, purpose,
                    json.dumps(extra) if extra else None, now.isoformat(), exp.isoformat()))
        cx.commit()
    return tok


def test_resend_reorder_mints_fresh_and_sends(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    sent = []
    monkeypatch.setattr(app_module, "send_magic_link_email",
                        lambda to, name, url: sent.append((to, url)) or ("smtp", None))
    tok = _seed_token(app_module, db, "reorder", email="r@x.com")
    r = app_module.app.test_client().post("/link/resend", json={"token": tok})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert len(sent) == 1 and sent[0][0] == "r@x.com" and "/reorder/auth/" in sent[0][1]
    # a fresh, unexpired reorder token now exists for the email
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM auth_tokens WHERE email='r@x.com' AND purpose='reorder'").fetchone()[0]
    assert n == 2  # the expired one + the fresh one


def test_resend_preserves_extra_for_practitioner(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "send_magic_link_email", lambda *a, **k: ("smtp", None))
    tok = _seed_token(app_module, db, "practitioner_claim", email="p@x.com", extra={"practitioner_id": "P9"})
    app_module.app.test_client().post("/link/resend", json={"token": tok})
    import json as _j
    with sqlite3.connect(db) as cx:
        rows = cx.execute("SELECT extra FROM auth_tokens WHERE purpose='practitioner_claim' "
                          "AND expires_at > ?", (datetime.now(timezone.utc).isoformat(),)).fetchall()
    assert rows and _j.loads(rows[0][0]) == {"practitioner_id": "P9"}


def test_resend_bogus_token_ok_no_send(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    sent = []
    monkeypatch.setattr(app_module, "send_magic_link_email", lambda *a, **k: sent.append(1) or ("smtp", None))
    r = app_module.app.test_client().post("/link/resend", json={"token": "not-a-real-token"})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert sent == []


def test_resend_reveal_existing_sends_reveal_email(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        br.upsert(cx, "rev@x.com", "2026-06-20", {"body": "x"},
                  [{"name": "Top", "slug": "top", "meaning": "m"}], "s")
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body: sent.append((to, subj, body)) or True)
    tok = _seed_token(app_module, db, "biofield_reveal", email="rev@x.com")
    r = app_module.app.test_client().post("/link/resend", json={"token": tok})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert len(sent) == 1 and "/begin/biofield/" in sent[0][2]


def test_resend_reveal_missing_ok_no_send(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: sent.append(1) or True)
    tok = _seed_token(app_module, db, "biofield_reveal", email="nobody@x.com")
    r = app_module.app.test_client().post("/link/resend", json={"token": tok})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert sent == []
```

- [ ] **Step 2: Run -> FAIL**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_link_resend.py -v`
Expected: FAIL (route `/link/resend` does not exist -> 404).

- [ ] **Step 3: Add the factory, handlers, registry, and route** (`app.py`, near the other auth/magic-link helpers)

```python
def _link_resend_generic(purpose, url_template, ttl):
    """Factory: mint a fresh `purpose` token (preserving extra) and email its URL."""
    def handler(email, extra):
        tok = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens "
                       "(token_hash TEXT, email TEXT, purpose TEXT, extra TEXT, "
                       "created_at TEXT, expires_at TEXT, consumed_at TEXT)")
            cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
                       "VALUES (?,?,?,?,?,?)",
                       (_hash_token(tok), email, purpose, extra, now.isoformat(), (now + ttl).isoformat()))
            cx.commit()
        send_magic_link_email(email, "", f"{PUBLIC_BASE_URL}" + url_template.format(token=tok))
    return handler


def _resend_biofield_reveal(email, extra):
    """Re-mint the reveal token (both biofield_reveals.token_hash + auth_tokens) and
    send the reveal email - only if a reveal still exists for the email."""
    from dashboard import biofield_reveals as _br
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _br.init_table(cx)
        row = cx.execute("SELECT id FROM biofield_reveals WHERE email=? ORDER BY id DESC LIMIT 1",
                         (email,)).fetchone()
        if not row:
            return
        tok = secrets.token_urlsafe(32)
        _br.set_token(cx, row[0], _hash_token(tok))
        now = datetime.now(timezone.utc)
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
                   "VALUES (?,?,?,?,?)",
                   (_hash_token(tok), email, "biofield_reveal", now.isoformat(),
                    (now + timedelta(days=30)).isoformat()))
        cx.commit()
    url = f"{PUBLIC_BASE_URL}/begin/biofield/{tok}"
    body = ("Aloha,\n\nYour Biofield Analysis is ready. View your reading here:\n"
            f"{url}\n\nIn wellness,\nDr. Glen and Rae\n")
    _send_inquiry_email(email, "Your Biofield Analysis is ready", body)


_AUTH_TTL = timedelta(minutes=AUTH_TOKEN_TTL_MIN)
RESEND_HANDLERS = {
    "biofield_reveal": _resend_biofield_reveal,
    "reorder": _link_resend_generic("reorder", "/reorder/auth/{token}", _AUTH_TTL),
    "magic_link": _link_resend_generic("magic_link", "/auth/magic-link/verify?token={token}", _AUTH_TTL),
    "affiliate_magic_link": _link_resend_generic("affiliate_magic_link", "/affiliate/login-verify?token={token}", _AUTH_TTL),
    "practitioner_claim": _link_resend_generic("practitioner_claim", "/practitioner-claim/{token}", timedelta(days=7)),
    "practitioner_optout": _link_resend_generic("practitioner_optout", "/practitioner-optout/{token}", timedelta(days=365)),
    "practitioner_share": _link_resend_generic("practitioner_share", "/share-with-practitioner/{token}", timedelta(days=30)),
}

_RESEND_OK = {"ok": True, "message": "If that link was valid, a fresh one is on its way. Check your email."}


@app.route("/link/resend", methods=["POST"])
def link_resend():
    """Re-mint + re-send a magic link from its expired token (or inquiry ids). Always
    returns a generic ok (no enumeration)."""
    data = request.get_json(silent=True) or {}
    inquiry_id = (data.get("inquiry_id") or "").strip()
    practitioner_id = (data.get("practitioner_id") or "").strip()
    token = (data.get("token") or "").strip()
    try:
        if inquiry_id and practitioner_id:
            _resend_inquiry_reply(inquiry_id, practitioner_id)  # Task 2
        elif token:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                row = cx.execute("SELECT email, purpose, extra FROM auth_tokens WHERE token_hash=?",
                                 (_hash_token(token),)).fetchone()
            if row:
                email, purpose, extra = row[0], row[1], row[2]
                handler = RESEND_HANDLERS.get(purpose)
                if handler and email:
                    handler(email, extra)
    except Exception as e:
        print(f"[link-resend] {e!r}", flush=True)
    return jsonify(_RESEND_OK)
```

Note: Task 1 references `_resend_inquiry_reply` (added in Task 2). Define a temporary `def _resend_inquiry_reply(a, b): return` stub now so the module imports; Task 2 replaces it. (Or implement Task 2 first - either order works since the route catches errors.)

- [ ] **Step 4: Run -> PASS** (the reveal + reorder + practitioner + bogus tests; the inquiry test arrives in Task 2)

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_link_resend.py -v`
Expected: PASS for the five Task-1 tests.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_link_resend.py
git commit -m "feat: POST /link/resend + auth_tokens resend registry"
```

---

## Task 2: inquiry-reply branch

**Files:**
- Modify: `app.py` (replace the `_resend_inquiry_reply` stub)
- Test: `tests/test_link_resend.py`

**Interfaces:** `_resend_inquiry_reply(inquiry_id, practitioner_id) -> None` - mints a fresh `inquiry_reply_tokens` row + emails the practitioner the reply link.

- [ ] **Step 1: Add the failing test**

```python
def test_resend_inquiry_reply_mints_and_emails_practitioner(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS inquiry_reply_tokens "
                   "(token_hash TEXT PRIMARY KEY, inquiry_id TEXT, practitioner_id TEXT, "
                   "created_at TEXT, expires_at TEXT, UNIQUE(inquiry_id, practitioner_id))")
        cx.execute("CREATE TABLE IF NOT EXISTS inquiry_practitioners "
                   "(id TEXT, inquiry_id TEXT, practitioner_id TEXT, practitioner_email TEXT, status TEXT, email_sent_at TEXT)")
        cx.execute("INSERT INTO inquiry_practitioners (id, inquiry_id, practitioner_id, practitioner_email, status) "
                   "VALUES ('1','INQ','P9','doc@x.com','sent')")
        cx.commit()
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body, **k: sent.append((to, body)) or True)
    r = app_module.app.test_client().post("/link/resend",
                                          json={"inquiry_id": "INQ", "practitioner_id": "P9"})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert len(sent) == 1 and sent[0][0] == "doc@x.com"
    assert "/inquiries/INQ/P9/reply?token=" in sent[0][1]
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM inquiry_reply_tokens WHERE inquiry_id='INQ' AND practitioner_id='P9'").fetchone()[0]
    assert n == 1


def test_resend_inquiry_reply_unknown_practitioner_no_send(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS inquiry_reply_tokens "
                   "(token_hash TEXT PRIMARY KEY, inquiry_id TEXT, practitioner_id TEXT, created_at TEXT, expires_at TEXT, UNIQUE(inquiry_id, practitioner_id))")
        cx.execute("CREATE TABLE IF NOT EXISTS inquiry_practitioners "
                   "(id TEXT, inquiry_id TEXT, practitioner_id TEXT, practitioner_email TEXT, status TEXT, email_sent_at TEXT)")
        cx.commit()
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: sent.append(1) or True)
    r = app_module.app.test_client().post("/link/resend",
                                          json={"inquiry_id": "NOPE", "practitioner_id": "P0"})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert sent == []
```

- [ ] **Step 2: Run -> FAIL** (the stub sends nothing -> the first test fails on `len(sent) == 1`)

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_link_resend.py::test_resend_inquiry_reply_mints_and_emails_practitioner -v`
Expected: FAIL.

- [ ] **Step 3: Replace the stub** (`app.py`)

```python
def _resend_inquiry_reply(inquiry_id, practitioner_id):
    """Mint a fresh inquiry reply token for (inquiry_id, practitioner_id) and email the
    practitioner the secure reply link. No-op if the practitioner is unknown."""
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        prow = cx.execute(
            "SELECT practitioner_email FROM inquiry_practitioners "
            "WHERE inquiry_id=? AND practitioner_id=? AND practitioner_email IS NOT NULL "
            "AND practitioner_email <> '' ORDER BY email_sent_at DESC LIMIT 1",
            (inquiry_id, practitioner_id)).fetchone()
        if not prow:
            return
        pmail = prow[0]
        tok = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        cx.execute("INSERT OR REPLACE INTO inquiry_reply_tokens "
                   "(token_hash, inquiry_id, practitioner_id, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (_hash_token(tok), inquiry_id, practitioner_id,
                    now.isoformat() + "Z", (now + timedelta(days=30)).isoformat() + "Z"))
        cx.commit()
    url = f"{PUBLIC_BASE_URL}/inquiries/{inquiry_id}/{practitioner_id}/reply?token={tok}"
    body = ("Aloha,\n\nHere is your fresh secure reply link for this inquiry:\n"
            f"{url}\n\nIn wellness,\nDr. Glen\n")
    _send_inquiry_email(pmail, "Your fresh reply link", body)
```

- [ ] **Step 4: Run -> PASS**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_link_resend.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_link_resend.py
git commit -m "feat: inquiry-reply branch for /link/resend"
```

---

## Task 3: the "Request a fresh link" button on every invalid surface

**Files:**
- Modify: `static/begin-biofield.html`, `static/coaching.html`, `static/practitioner-claim.html`, `static/practitioner-optout.html`, `static/practitioner-share.html`, `static/inquiry-reply.html`
- Modify: `app.py` (the inline-HTML invalid responses in the reorder route ~9562 and the sign-in route ~10976)
- Test: `tests/test_link_resend.py` (serve assertions)

**Interfaces:** consumes `POST /link/resend`.

- [ ] **Step 1: Add the failing serve test**

```python
def test_reveal_page_ships_resend_button(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    html = app_module.app.test_client().get("/begin/biofield/any-token").data.decode()
    assert "/link/resend" in html
    assert "Request a fresh link" in html
```

- [ ] **Step 2: Run -> FAIL**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_link_resend.py::test_reveal_page_ships_resend_button -v`
Expected: FAIL (button absent).

- [ ] **Step 3: Wire the reveal page** (`static/begin-biofield.html`, State 1, the `reveal === null` block)

After `card.appendChild(p);` and before `root.appendChild(card);`, insert a button that POSTs the URL token:

```javascript
        var btn = document.createElement("button");
        btn.textContent = "Request a fresh link";
        btn.className = "resend-btn";
        btn.onclick = function(){
          btn.disabled = true;
          fetch("/link/resend", {method:"POST", headers:{"Content-Type":"application/json"},
                                 body: JSON.stringify({token: _token})})
            .then(function(){ p.textContent = "Check your email for a fresh link."; })
            .catch(function(){ btn.disabled = false; });
        };
        card.appendChild(btn);
```

- [ ] **Step 4: Wire the inquiry-reply page** (`static/inquiry-reply.html`, the error block) - the ids come from the URL path `/inquiries/<id>/<pid>/reply`:

```html
<button id="resend-link-btn">Request a fresh link</button>
<script>
  (function(){
    var b = document.getElementById("resend-link-btn");
    if(!b) return;
    var m = location.pathname.match(/\/inquiries\/([^\/]+)\/([^\/]+)\/reply/);
    b.onclick = function(){
      if(!m) return;
      b.disabled = true;
      fetch("/link/resend", {method:"POST", headers:{"Content-Type":"application/json"},
                             body: JSON.stringify({inquiry_id: m[1], practitioner_id: m[2]})})
        .then(function(){ b.textContent = "Check your email."; })
        .catch(function(){ b.disabled = false; });
    };
  })();
</script>
```

- [ ] **Step 5: Wire the token-in-path pages** (`static/coaching.html`, `static/practitioner-claim.html`, `static/practitioner-optout.html`, `static/practitioner-share.html`) - add the same button; read the token from the last path segment (`location.pathname`) for the practitioner pages, or the `?token=` query for coaching:

```html
<button id="resend-link-btn">Request a fresh link</button>
<script>
  (function(){
    var b = document.getElementById("resend-link-btn");
    if(!b) return;
    var parts = location.pathname.replace(/\/+$/,"").split("/");
    var qp = new URLSearchParams(location.search);
    var token = qp.get("token") || parts[parts.length-1] || "";
    b.onclick = function(){
      b.disabled = true;
      fetch("/link/resend", {method:"POST", headers:{"Content-Type":"application/json"},
                             body: JSON.stringify({token: token})})
        .then(function(){ b.textContent = "Check your email for a fresh link."; })
        .catch(function(){ b.disabled = false; });
    };
  })();
</script>
```

Place it inside each page's invalid-link block (next to the existing "This link is no longer valid" copy). Only add to the invalid/error state markup, not the success states.

- [ ] **Step 6: Wire the inline-HTML routes** (`app.py` reorder ~9562 and sign-in ~10976) - append the same button + script to the inline invalid-HTML string. The reorder token is in the path (`/reorder/auth/<token>`); the sign-in verify token is in `?token=`. Use the same path/query reader as Step 5.

- [ ] **Step 7: Run -> PASS**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_link_resend.py -v`
Expected: PASS (serve assertion + all backend tests).

- [ ] **Step 8: Commit**

```bash
git add static/ app.py tests/test_link_resend.py
git commit -m "feat: Request-a-fresh-link button on every invalid magic-link surface"
```

---

## Verification

- `pytest tests/test_link_resend.py -v` green; quick `-k "biofield or reorder or auth"` for no regressions in the touched routes.
- Final whole-branch review (most capable model). Focus: generic-ok contract (no enumeration, never errors); `extra` preserved on re-mint; reveal handler updates BOTH `biofield_reveals.token_hash` and `auth_tokens` + only when a reveal exists; inquiry-reply emails the practitioner (not the client); buttons only on invalid/error states; XSS-safe; no emoji/em-dash.
- Ship via PR + merge to main (auto-deploys); gentle probe of `/begin/biofield/<bad>` (button present) per the warm-up rule.

## Build order

Task 1 (endpoint + registry; stub the inquiry fn) -> Task 2 (inquiry branch) -> Task 3 (front-end). Backend is fully testable before any front-end work.
