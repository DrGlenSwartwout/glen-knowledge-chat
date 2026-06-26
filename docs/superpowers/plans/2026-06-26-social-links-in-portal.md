# Social-Links in Personal Portal (2b-2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Display the ambassador's social links + an add-form in the personal portal's enrolled Ambassador card, authed by portal identity. Reuse a shared insert helper.

**Architecture:** `affiliate_dashboard.add_social_links(cx, slug, email, urls)` (extract); `/affiliate/social-links` refactored to use it; new `POST /api/portal/<token>/social-links` (identity-authed); render list + form in `client-portal.html`.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest.

## Global Constraints

- `add_social_links` is self-contained: `CREATE TABLE IF NOT EXISTS affiliate_social_links (...)` (no-op on prod), then for each url in `urls[:10]`: strip+truncate 500, skip non-http(s), INSERT `(ts, slug, email, url)` with ISO ts; commit; return count. None-raising (non-list urls → 0).
- New portal endpoint: identity via `_pi.resolve_identity` (token/session); `ident.email` → approved `affiliate_signups` slug; 404 if no identity, 403 if not an approved ambassador. `ident.email` is the authed identity, never client-supplied.
- Don't build points-awarding logic; display existing values only. Client portal only.
- `app.py`/HTML can't import offline → Tasks 2-4 verified live. Task 1 offline-TDD.
- Offline test cmd: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`.

---

### Task 1: `add_social_links` helper

**Files:**
- Modify: `dashboard/affiliate_dashboard.py` (append)
- Test: `tests/test_affiliate_social_links.py`

**Interfaces:**
- Produces: `add_social_links(cx, slug, email, urls) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_affiliate_social_links.py
import sqlite3
from dashboard import affiliate_dashboard as ad

def test_inserts_only_http_and_returns_count():
    cx = sqlite3.connect(":memory:")  # no table pre-created
    n = ad.add_social_links(cx, "amy7", "amy@x.com",
                            ["https://a.com/p", "ftp://nope", "http://b.com/q", "  notaurl "])
    assert n == 2
    rows = cx.execute("SELECT slug, email, url FROM affiliate_social_links ORDER BY id").fetchall()
    assert rows == [("amy7", "amy@x.com", "https://a.com/p"),
                    ("amy7", "amy@x.com", "http://b.com/q")]

def test_caps_at_10():
    cx = sqlite3.connect(":memory:")
    n = ad.add_social_links(cx, "amy7", "amy@x.com", [f"https://a.com/{i}" for i in range(15)])
    assert n == 10

def test_non_list_is_zero():
    cx = sqlite3.connect(":memory:")
    assert ad.add_social_links(cx, "amy7", "amy@x.com", None) == 0
    assert ad.add_social_links(cx, "amy7", "amy@x.com", []) == 0

def test_truncates_to_500():
    cx = sqlite3.connect(":memory:")
    long = "https://a.com/" + ("x" * 600)
    ad.add_social_links(cx, "amy7", "amy@x.com", [long])
    stored = cx.execute("SELECT url FROM affiliate_social_links").fetchone()[0]
    assert len(stored) == 500
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_affiliate_social_links.py -v`
Expected: FAIL — `add_social_links` missing.

- [ ] **Step 3: Implement** — append to `dashboard/affiliate_dashboard.py` (it already imports nothing time-related; add `from datetime import datetime, timezone` at the top if not present):

```python
def add_social_links(cx, slug, email, urls):
    """Store an ambassador's social-share URLs (http/https only, <=500 chars, max 10).
    Self-contained (creates the table if absent). Returns the count inserted."""
    cx.execute(
        "CREATE TABLE IF NOT EXISTS affiliate_social_links ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, slug TEXT, email TEXT, url TEXT, "
        "points INTEGER DEFAULT 0, views INTEGER DEFAULT 0, likes INTEGER DEFAULT 0, "
        "shares INTEGER DEFAULT 0)")
    if not isinstance(urls, (list, tuple)):
        return 0
    ts = datetime.now(timezone.utc).isoformat()
    count = 0
    for u in list(urls)[:10]:
        u = (u or "").strip()[:500]
        if not u.startswith(("http://", "https://")):
            continue
        cx.execute("INSERT INTO affiliate_social_links (ts, slug, email, url) VALUES (?,?,?,?)",
                   (ts, slug, email, u))
        count += 1
    cx.commit()
    return count
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_affiliate_social_links.py -v`
Expected: PASS (4 tests). Also run the existing affiliate_dashboard suite to confirm no regression: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_affiliate_dashboard.py -q`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/affiliate_dashboard.py tests/test_affiliate_social_links.py
git commit -m "feat(2b2): affiliate_dashboard.add_social_links (shared social-link insert)"
```

---

### Task 2: Refactor `/affiliate/social-links` to use the helper

**Files:**
- Modify: `app.py` — `affiliate_social_links_submit` (route `/affiliate/social-links`, ~line 8636; the insert loop ~8652-8662).

- [ ] **Step 1: Refactor**

Keep the OPTIONS handler, token read, the `SELECT slug,email,status` lookup, the `if not row → 404`, and `if status != "approved" → 403`. Replace the block from `ts = datetime.now(...)` through `cx.commit()` (the inline loop) with:
```python
        count = affiliate_dashboard.add_social_links(cx, slug, email, urls)
```
(keep the surrounding `with _db_lock, sqlite3.connect(LOG_DB) as cx:` and the final `return jsonify({"ok": True, "count": count})`). Ensure `affiliate_dashboard` is importable here (add `from dashboard import affiliate_dashboard` at the top of the route or use the module-level import if present).

- [ ] **Step 2: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "refactor(2b2): /affiliate/social-links uses add_social_links helper"
```

- [ ] **Step 3: Live verification (post-deploy — record in report)**

`POST /affiliate/social-links` with an approved token + `{"urls":["https://x.com/test"]}` → `{ok:true, count:1}` (standalone submit unchanged).

---

### Task 3: New `POST /api/portal/<token>/social-links`

**Files:**
- Modify: `app.py` — add the route near the other `/api/portal/<token>/*` POST routes (e.g. after `/api/portal/<token>/biofield/request`, ~line 11190).

- [ ] **Step 1: Add the route**

```python
@app.route("/api/portal/<token>/social-links", methods=["POST"])
def api_portal_social_links(token):
    from dashboard import portal_identity as _pi
    from dashboard import client_portal as _cp
    from dashboard import affiliate_dashboard as _ad
    sess = request.cookies.get("rm_portal_session", "")
    urls = (request.get_json(silent=True) or {}).get("urls") or []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        _pi._ensure_people_table(cx)
        ident = _pi.resolve_identity(cx, token=token, session_token=sess,
                                     client_login_enabled=_client_login_enabled())
        if ident is None:
            return jsonify({"error": "not found"}), 404
        row = cx.execute(
            "SELECT slug FROM affiliate_signups WHERE lower(email)=? AND status='approved' LIMIT 1",
            (ident.email,)).fetchone()
        if not row:
            return jsonify({"error": "not an approved ambassador"}), 403
        count = _ad.add_social_links(cx, row[0], ident.email, urls)
    return jsonify({"ok": True, "count": count})
```
(`ident.email` is already lowercased by resolve_identity in this codebase; if not, lowercase it for the query. Mirror the import + `_db_lock` style of the adjacent `_biofield_transition`/biofield routes.)

- [ ] **Step 2: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "feat(2b2): POST /api/portal/<token>/social-links (identity-authed)"
```

- [ ] **Step 3: Live verification (post-deploy — record in report)**

For an enrolled portal user (token), `POST /api/portal/<token>/social-links {"urls":["https://x.com/test"]}` → `{ok:true, count:1}`; a non-enrolled portal token → 403; unknown token → 404. Then `/api/portal/<token>/view` shows the link in `ambassador.dashboard.social_links`.

---

### Task 4: Render social links + add-form in the portal

**Files:**
- Modify: `static/client-portal.html` — the enrolled Ambassador card (where 2b-1 renders the dashboard).

**Read first:** the enrolled dashboard render added in 2b-1, the page's `esc()`, card/`kv`/`btn` classes, and how the page POSTs to `/api/portal/<seg>/*` (e.g. notify-pref / biofield) to mirror the fetch idiom + the `seg` variable.

- [ ] **Step 1: Add the list + form**

In the enrolled branch (`var dash = amb.dashboard || {}`), after the recent/recruit blocks, add:
- **Social links list:** `(dash.social_links||[]).map(s => ...)` → each `s.url` (link) + a muted line `s.points||0 pts · s.views||0 views · s.likes||0 likes · s.shares||0 shares`; if empty → muted "No social shares yet." `esc()` all values.
- **Add form:** an `<input>`/`<textarea>` (id e.g. `social-url-input`) + a button calling a new JS fn `addSocial()` that reads the input (split on newlines/commas into a urls array), `POST /api/portal/${encodeURIComponent(seg)}/social-links` with `{urls}`, and on `{ok}` re-runs the page's load()/render to show the new link(s). Mirror the page's existing POST helper (same as notify-pref/biofield POSTs).

- [ ] **Step 2: JS sanity + commit**

Sanity-check the inserted `<script>` parses (e.g. `node -e` extracting script blocks, as done in prior portal tasks). Then:
```bash
git add static/client-portal.html
git commit -m "feat(2b2): render social-links list + add-form in the portal Ambassador card"
```

- [ ] **Step 3: Live render-verify (post-deploy — record in report)**

For an enrolled portal user, load `/portal/<token>` → the Ambassador card shows the social-links list + add-form; submitting a URL persists (re-renders with the new link); zero console errors.

---

## Self-Review

**1. Spec coverage:** add_social_links helper → Task 1; standalone refactor → Task 2; identity-authed portal endpoint → Task 3; list + form render → Task 4. ✅
**2. Placeholder scan:** No TBD; Task 1 full code+tests; Task 2/3 give exact anchors + full route code; Task 4 cites the file/idiom to mirror with concrete element ids + live checks. ✅
**3. Type consistency:** `add_social_links(cx, slug, email, urls) -> int` identical in Task 1 def, Task 2 + Task 3 calls; the portal endpoint path `/api/portal/<token>/social-links` matches Task 4's POST target; `dash.social_links` keys (url/points/views/likes/shares) match build_dashboard's output. ✅
