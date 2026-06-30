# SP2b-3 — ASH ally practitioner client-in-focus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a practitioner search their own clients and bring one "into focus" so the practitioner chat reads/writes that client's ASH memory — with a hard authorization guard so a practitioner can only ever touch their own clients.

**Architecture:** Two pure, unit-tested functions in `dashboard/practitioner_portal.py` (`client_belongs_to_practitioner` = the ownership guard, `search_clients` = the practitioner-scoped client search), both keyed off the existing `dispensary_orders` table. A new `GET /api/practitioner/clients/search` endpoint and wiring in `/api/practitioner/chat` (read `client_email`, re-check ownership every turn, then `ash_ally.ally_overlay`/`record_turn` keyed on the client's email). Front-end: fix the broken token auth, add a client-search box + "in focus" badge to `static/practitioner-dropship.html`, and send `client_email` with each chat turn. Same `ASH_ALLY_ENABLED` flag, dark by default.

**Tech Stack:** Python 3, stdlib `sqlite3`, the existing `dashboard/ash_ally.py`/`ash_map.py`, Flask routes in `app.py`, vanilla JS in the dropship HTML. Backend tests: plain pytest (functions take `db_path`, no app import).

## Global Constraints

- Modify: `dashboard/practitioner_portal.py`, `app.py`, `static/practitioner-dropship.html`. New test: `tests/test_practitioner_clients.py`.
- **Authorization is mandatory and non-negotiable:** before ANY `ash_map`/`ash_ally` read or write keyed on a client email in the practitioner chat, verify the client belongs to the authenticated practitioner via `client_belongs_to_practitioner(pid, email)`. `client_email` arrives from the client and is never trusted. Re-checked server-side on every turn.
- `practitioner_id` is a Supabase UUID **string** — always `str(...)`, never `int()`.
- Client set = `dispensary_orders` clients (the only practitioner→client link). Both `dispensary_orders` and `people` live in LOG_DB.
- New backend functions mirror `dispensary_order_history` exactly: signature `(..., *, db_path=None)`, `p = db_path or _db_path()`, `with sqlite3.connect(p) as cx: _ensure_dispensary_table(cx)`.
- `app.py` aliases: `_pp` = `dashboard.practitioner_portal`, `_chat` = `dashboard.practitioner_chat`; `ash_ally`, `LOG_DB`, `_db_lock` all already imported (lines 85/153/154). `_practitioner_session_pid()` (app.py:10034) resolves the practitioner UUID from `?token=` or body `token`.
- Record dispatch uses the established idiom: `threading.Thread(target=ash_ally.record_turn, args=(LOG_DB, _db_lock, subject, message, result.get("reply","")), daemon=True).start()`, try/except-wrapped; only when an owned client is in focus.
- Same flag `ASH_ALLY_ENABLED`, dark by default (`ally_overlay`/`record_turn` are already gated + fail-open).
- Backend tests run with plain pytest, no doppler/network: `python3 -m pytest tests/test_practitioner_clients.py -v`.

---

### Task 1: `client_belongs_to_practitioner` — the ownership guard

**Files:**
- Modify: `dashboard/practitioner_portal.py` (add after `dispensary_order_history`, ~line 188)
- Test: `tests/test_practitioner_clients.py` (new)

**Interfaces:**
- Consumes: `_ensure_dispensary_table`, `_db_path`, `record_dispensary_order` (all existing).
- Produces: `client_belongs_to_practitioner(practitioner_id, email, *, db_path=None) -> bool` — True iff a `dispensary_orders` row exists with that practitioner_id and (case-insensitive) customer_email. Empty/None practitioner_id or email → False.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_practitioner_clients.py
import dashboard.practitioner_portal as pp


def _seed(db):
    pp.record_dispensary_order("prac-1", invoice_id="i1", customer_email="Karin@X.com", db_path=db)
    pp.record_dispensary_order("prac-2", invoice_id="i2", customer_email="bob@x.com", db_path=db)


def test_belongs_true_for_own_client_case_insensitive(tmp_path):
    db = str(tmp_path / "t.db"); _seed(db)
    assert pp.client_belongs_to_practitioner("prac-1", "karin@x.com", db_path=db) is True
    assert pp.client_belongs_to_practitioner("prac-1", "  KARIN@X.COM ", db_path=db) is True


def test_belongs_false_for_other_practitioners_client(tmp_path):
    db = str(tmp_path / "t.db"); _seed(db)
    # bob belongs to prac-2 — prac-1 must NOT be able to claim him
    assert pp.client_belongs_to_practitioner("prac-1", "bob@x.com", db_path=db) is False


def test_belongs_false_for_unknown_or_empty(tmp_path):
    db = str(tmp_path / "t.db"); _seed(db)
    assert pp.client_belongs_to_practitioner("prac-1", "nobody@x.com", db_path=db) is False
    assert pp.client_belongs_to_practitioner("prac-1", "", db_path=db) is False
    assert pp.client_belongs_to_practitioner("prac-1", None, db_path=db) is False
    assert pp.client_belongs_to_practitioner("", "karin@x.com", db_path=db) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_practitioner_clients.py -k belongs -v`
Expected: FAIL — `AttributeError: module 'dashboard.practitioner_portal' has no attribute 'client_belongs_to_practitioner'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to dashboard/practitioner_portal.py after dispensary_order_history (~line 188)
def client_belongs_to_practitioner(practitioner_id, email, *, db_path=None) -> bool:
    """True iff `email` is a client of `practitioner_id` (has a dispensary order under
    them). The authorization guard before any ASH read/write keyed on a client email —
    a practitioner may only act on their own clients. Case-insensitive on email."""
    em = (email or "").strip().lower()
    if not practitioner_id or not em:
        return False
    p = db_path or _db_path()
    with sqlite3.connect(p) as cx:
        _ensure_dispensary_table(cx)
        row = cx.execute(
            "SELECT 1 FROM dispensary_orders "
            "WHERE practitioner_id=? AND lower(customer_email)=? LIMIT 1",
            (str(practitioner_id), em),
        ).fetchone()
    return row is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_practitioner_clients.py -k belongs -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/practitioner_portal.py tests/test_practitioner_clients.py
git commit -m "feat(practitioner): client_belongs_to_practitioner ownership guard"
```

---

### Task 2: `search_clients` — practitioner-scoped client search

**Files:**
- Modify: `dashboard/practitioner_portal.py` (add after Task 1's function)
- Test: `tests/test_practitioner_clients.py` (append)

**Interfaces:**
- Consumes: `_ensure_dispensary_table`, `_db_path`, `record_dispensary_order`.
- Produces: `search_clients(practitioner_id, q, *, limit=8, db_path=None) -> list[dict]` — `[{"email","name"}]` of the practitioner's own dispensary clients matching `q` (by email substring or joined `people.name`), deduped by email (DISTINCT), capped at `limit`. Empty `q` → `[]`. Never returns another practitioner's client.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_practitioner_clients.py
import sqlite3


def _seed_full(db):
    pp.record_dispensary_order("prac-1", invoice_id="i1", customer_email="karin@x.com", db_path=db)
    pp.record_dispensary_order("prac-1", invoice_id="i1b", customer_email="karin@x.com", db_path=db)  # repeat
    pp.record_dispensary_order("prac-1", invoice_id="i3", customer_email="larry@x.com", db_path=db)
    pp.record_dispensary_order("prac-2", invoice_id="i2", customer_email="bob@x.com", db_path=db)
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE IF NOT EXISTS people (email TEXT, name TEXT)")
    cx.execute("INSERT INTO people (email, name) VALUES (?, ?)", ("karin@x.com", "Karin Doe"))
    cx.commit(); cx.close()


def test_search_empty_q_returns_empty(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    assert pp.search_clients("prac-1", "", db_path=db) == []
    assert pp.search_clients("prac-1", "   ", db_path=db) == []


def test_search_by_email_and_by_joined_name(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    by_email = pp.search_clients("prac-1", "karin", db_path=db)
    assert {"email": "karin@x.com", "name": "Karin Doe"} in by_email
    by_name = pp.search_clients("prac-1", "doe", db_path=db)
    assert any(c["email"] == "karin@x.com" for c in by_name)


def test_search_dedupes_repeat_orders(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    res = pp.search_clients("prac-1", "karin", db_path=db)
    assert sum(1 for c in res if c["email"] == "karin@x.com") == 1


def test_search_never_returns_other_practitioners_client(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    assert pp.search_clients("prac-1", "bob", db_path=db) == []          # bob is prac-2's
    broad = pp.search_clients("prac-1", "x.com", db_path=db)              # matches everyone's email
    assert all(c["email"] != "bob@x.com" for c in broad)                 # but bob never leaks to prac-1


def test_search_respects_limit(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    assert len(pp.search_clients("prac-1", "x.com", limit=1, db_path=db)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_practitioner_clients.py -k search -v`
Expected: FAIL — `AttributeError: ... has no attribute 'search_clients'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to dashboard/practitioner_portal.py after client_belongs_to_practitioner
def search_clients(practitioner_id, q, *, limit=8, db_path=None) -> List[dict]:
    """The practitioner's own dispensary clients matching `q` (email substring or joined
    people.name), for the chat client-focus picker. Deduped by email; scoped to the
    practitioner (never returns another practitioner's client). Empty q -> []."""
    qq = (q or "").strip().lower()
    if not practitioner_id or not qq:
        return []
    like = f"%{qq}%"
    p = db_path or _db_path()
    with sqlite3.connect(p) as cx:
        _ensure_dispensary_table(cx)
        rows = cx.execute(
            "SELECT DISTINCT d.customer_email AS email, COALESCE(pe.name,'') AS name "
            "FROM dispensary_orders d "
            "LEFT JOIN people pe ON lower(pe.email) = lower(d.customer_email) "
            "WHERE d.practitioner_id = ? "
            "  AND d.customer_email IS NOT NULL AND d.customer_email <> '' "
            "  AND (lower(d.customer_email) LIKE ? OR lower(COALESCE(pe.name,'')) LIKE ?) "
            "ORDER BY name, email LIMIT ?",
            (str(practitioner_id), like, like, int(limit)),
        ).fetchall()
    return [{"email": r[0], "name": r[1]} for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_practitioner_clients.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/practitioner_portal.py tests/test_practitioner_clients.py
git commit -m "feat(practitioner): search_clients (practitioner-scoped client search)"
```

---

### Task 3: `GET /api/practitioner/clients/search` endpoint

**Files:**
- Modify: `app.py` (insert between `_build_ff_catalog` end ~11573 and the `@app.route("/api/practitioner/chat"...)` decorator ~11576)

**Interfaces:**
- Consumes: `_practitioner_session_pid()`, `_pp.search_clients(pid, q)`.
- Produces: `GET /api/practitioner/clients/search?token=&q=` → `{"ok": True, "clients": [...]}` or `401`.

- [ ] **Step 1: Add the route**

Insert this new route immediately before the `@app.route("/api/practitioner/chat", methods=["POST"])` decorator (after the `return catalog` of `_build_ff_catalog`):

```python
@app.route("/api/practitioner/clients/search", methods=["GET"])
def api_practitioner_clients_search():
    """Search the authenticated practitioner's own clients (dispensary-order clients)
    by name/email for the chat client-focus picker. Scoped to the practitioner."""
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "authentication required"}), 401
    q = (request.args.get("q") or "").strip()
    return jsonify({"ok": True, "clients": _pp.search_clients(pid, q)})
```

- [ ] **Step 2: Verify no syntax break + route present**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

Run: `grep -n "api/practitioner/clients/search" app.py`
Expected: the new route present (once).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(practitioner): GET /api/practitioner/clients/search (scoped client picker)"
```

---

### Task 4: Wire `/api/practitioner/chat` — authorize client_email → overlay/record

**Files:**
- Modify: `app.py` (`api_practitioner_chat`, ~11576-11602)

**Interfaces:**
- Consumes: `_pp.client_belongs_to_practitioner(pid, client_email)`, `ash_ally.ally_overlay(LOG_DB, subject)`, `ash_ally.record_turn(...)`, the `scoped_reply` `overlay=` param.

- [ ] **Step 1: Read client_email from the body**

After `history = body.get("history") or []` (line 11586), add:

```python
    client_email = (body.get("client_email") or "").strip().lower()
```

- [ ] **Step 2: Authorize + overlay + record around the scoped_reply call**

Replace:

```python
    catalog = _build_ff_catalog()
    result = _chat.scoped_reply(message, history, catalog)
```

with:

```python
    # Only act on a client this practitioner actually owns (authorization re-checked
    # every turn — client_email from the request is never trusted). Empty/unowned -> "".
    _subject = client_email if (client_email and _pp.client_belongs_to_practitioner(pid, client_email)) else ""
    _ally_ov = ash_ally.ally_overlay(LOG_DB, _subject)

    catalog = _build_ff_catalog()
    result = _chat.scoped_reply(message, history, catalog, overlay=_ally_ov)

    if _subject:
        try:
            import threading as _t
            _t.Thread(target=ash_ally.record_turn,
                      args=(LOG_DB, _db_lock, _subject, message, result.get("reply", "")),
                      daemon=True).start()
        except Exception:
            pass
```

- [ ] **Step 3: Verify no syntax break + wiring present**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

Run: `grep -n "client_belongs_to_practitioner" app.py`
Expected: present inside `api_practitioner_chat` (the guard).

Run: `grep -c "ash_ally.ally_overlay" app.py` and `grep -c "ash_ally.record_turn" app.py`
Expected: 7 and 7 (4 SSE from SP2b-1 + 2 scoped_reply from SP2b-2 + 1 here).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(practitioner): authorize client_email + wire ASH ally into /api/practitioner/chat"
```

---

### Task 5: Front-end — token fix + client search UI + focus badge

**Files:**
- Modify: `static/practitioner-dropship.html`

No TDD (static HTML/JS, no route/JS harness in this repo). Gate = grep-assert the required additions; behavioral proof is the go-live render-verify.

**Interfaces:**
- Consumes: `GET /api/practitioner/clients/search?token=&q=` (Task 3); the chat POST now carries `token` + `client_email`.

- [ ] **Step 1: Add the client-search + focus-badge markup**

In the chat panel, between `<div id="scoped-chat-header">Product assistant</div>` (line 75) and `<div id="scoped-chat-log"></div>` (line 76), insert:

```html
      <div id="client-focus-row" style="display:flex;gap:8px;align-items:center;margin:6px 0;flex-wrap:wrap">
        <input id="client-search" placeholder="Search your clients by name or email&hellip;" autocomplete="off" style="flex:1;min-width:180px">
        <span id="client-focus-badge" style="display:none;font-size:13px;background:#eef;padding:3px 8px;border-radius:10px"></span>
        <button id="client-focus-clear" type="button" style="display:none" onclick="clientFocusClear()">Clear</button>
      </div>
      <div id="client-search-results" style="display:none;border:1px solid #ddd;border-radius:6px;max-height:160px;overflow:auto"></div>
```

- [ ] **Step 2: Add the client-focus JS inside the chat IIFE**

Inside the `(function () {` chat IIFE, right after `var chatEndpoint = '/api/practitioner/chat';` (line 399), add:

```javascript
  var currentClientEmail = '';
  var clientSearchTimer = null;

  function renderClientResults(clients){
    var box = document.getElementById('client-search-results');
    if(!clients || !clients.length){ box.style.display='none'; box.innerHTML=''; return; }
    box.innerHTML = clients.map(function(c){
      var label = (c.name ? c.name + ' — ' : '') + c.email;
      return '<div class="client-result" data-email="'+c.email+'" style="padding:6px 10px;cursor:pointer">'+label+'</div>';
    }).join('');
    box.style.display='block';
    Array.prototype.forEach.call(box.querySelectorAll('.client-result'), function(el){
      el.onclick = function(){ clientFocusSet(el.getAttribute('data-email'), el.textContent); };
    });
  }
  function clientFocusSet(email, label){
    currentClientEmail = email || '';
    var badge = document.getElementById('client-focus-badge');
    badge.textContent = 'Client in focus: ' + (label || email);
    badge.style.display = 'inline-block';
    document.getElementById('client-focus-clear').style.display = 'inline-block';
    var box = document.getElementById('client-search-results');
    box.style.display='none'; box.innerHTML='';
    document.getElementById('client-search').value = '';
  }
  window.clientFocusClear = function(){
    currentClientEmail = '';
    document.getElementById('client-focus-badge').style.display='none';
    document.getElementById('client-focus-clear').style.display='none';
  };
  function clientSearch(q){
    if(!q || !q.trim()){ renderClientResults([]); return; }
    fetch('/api/practitioner/clients/search?token=' + encodeURIComponent(TOKEN) + '&q=' + encodeURIComponent(q.trim()))
      .then(function(r){ return r.json(); })
      .then(function(d){ if(d && d.ok) renderClientResults(d.clients || []); })
      .catch(function(){ /* fail-soft: no results */ });
  }
  (function(){
    var inp = document.getElementById('client-search');
    if(inp){
      inp.addEventListener('input', function(){
        clearTimeout(clientSearchTimer);
        var v = inp.value;
        clientSearchTimer = setTimeout(function(){ clientSearch(v); }, 250);
      });
    }
  })();
```

- [ ] **Step 3: Fix auth + send client_email on the chat POST**

Change the chat body line (line 461) from:

```javascript
    var body = {message: msg, history: chatHistory.slice(-10)};
```

to (adds the `token` — fixing the 401 bug — and the in-focus `client_email`):

```javascript
    var body = {token: TOKEN, message: msg, history: chatHistory.slice(-10), client_email: currentClientEmail};
```

- [ ] **Step 4: Verify the additions are present**

Run each grep; each must print a match:

```bash
grep -n "id=\"client-search\"" static/practitioner-dropship.html
grep -n "var currentClientEmail" static/practitioner-dropship.html
grep -n "api/practitioner/clients/search?token=" static/practitioner-dropship.html
grep -n "client_email: currentClientEmail" static/practitioner-dropship.html
grep -n "token: TOKEN, message: msg" static/practitioner-dropship.html
```

Expected: all five match. (Behavioral correctness — the typeahead, focus badge, and that the chat now authenticates — is proven at the go-live render-verify.)

- [ ] **Step 5: Commit**

```bash
git add static/practitioner-dropship.html
git commit -m "feat(practitioner-ui): client search + focus badge + token-auth fix on scoped chat"
```

---

### Task 6: Full-suite green + verification

**Files:** run-only.

- [ ] **Step 1: Run the backend suites**

Run: `python3 -m pytest tests/test_practitioner_clients.py tests/test_ash_ally.py tests/test_ash_map.py -v`
Expected: ALL passing (Task 1+2's 8 tests + SP2b-1/2 helper tests). Report the exact count.

- [ ] **Step 2: Confirm app parses + wiring counts**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('app ok')"`
Expected: `app ok`.

Run: `grep -c "ash_ally.ally_overlay" app.py` and `grep -c "ash_ally.record_turn" app.py`
Expected: 7 and 7.

Run: `grep -n "client_belongs_to_practitioner" app.py`
Expected: present in `api_practitioner_chat` (the guard runs before any client-keyed ASH access).

- [ ] **Step 3: Confirm the search endpoint + front-end wiring**

Run: `grep -n "api/practitioner/clients/search" app.py`
Expected: the route definition present.

Run: `grep -c "api/practitioner/clients/search" static/practitioner-dropship.html`
Expected: 1 (the front-end search fetch).

- [ ] **Step 4: Commit (only if a doc/verification file changed; otherwise skip)**

No code changes in this task.

---

## Self-Review

**Spec coverage:**
- Ownership guard `client_belongs_to_practitioner` (security core) → Task 1 ✓
- Practitioner-scoped `search_clients` (incl. never-returns-other-practitioner's-client) → Task 2 ✓
- `GET /api/practitioner/clients/search` endpoint → Task 3 ✓
- `/api/practitioner/chat` wiring: read client_email, re-check ownership every turn, overlay + record on owned client only, plain assistant otherwise → Task 4 ✓
- Front-end: token-auth fix, client search box, focus badge, send client_email → Task 5 ✓
- Dispensary-only client set, `dispensary_orders` keyed, UUID-string pid → Global Constraints + Tasks 1-2 ✓
- Same flag dark, fail-open helper, go-live render-verify → spec Verification; suite/parse → Task 6 ✓
- Out of scope (portal-published source, practitioner-mode extract, scan-analysis, Glendalf) → not in any task ✓

**Placeholder scan:** none — every code/test step carries full content. Task 6 is run-only by design.

**Type consistency:** `client_belongs_to_practitioner(practitioner_id, email, *, db_path=None) -> bool`, `search_clients(practitioner_id, q, *, limit=8, db_path=None) -> list[dict]` (items `{"email","name"}`), endpoint returns `{"ok","clients"}`, chat reads `client_email` and computes `_subject`/`_ally_ov`. The front-end fetches `?token=&q=` and posts `{token, message, history, client_email}`. Consistent across tasks.

**Security note:** authorization (`client_belongs_to_practitioner`) is enforced server-side in Task 4 on EVERY turn, independent of the front-end — the scoped search (Task 2/3) is a convenience, not the security boundary. A practitioner posting an arbitrary `client_email` they don't own gets `_subject=""` → no overlay, no record. This is the IDOR guard.

**Front-end testing boundary:** Task 5 edits a static HTML/JS file with no JS harness in this repo; its gate is grep-assertion of the required additions, with behavioral proof at the go-live render-verify (per the project's render-verify lesson). The security-critical logic is server-side and unit-tested (Tasks 1-2) / app-verified (Task 4).
