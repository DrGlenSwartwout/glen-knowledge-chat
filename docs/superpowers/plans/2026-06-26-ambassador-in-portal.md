# Ambassador in Personal Portal (Step 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an Ambassador section to the personal portal (authed by the portal's own identity): signup CTA if not enrolled, referral links inline if enrolled — reusing the existing `affiliate_signups` engine.

**Architecture:** A pure `_ambassador_block(cx, email, quiz_url, public_base_url)` in `dashboard/portal_view.py` (added to `get_portal_view`'s output); the `/api/portal/<token>/view` route passes the URL constants; `static/client-portal.html` renders the card.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest.

## Global Constraints

- Reuse the affiliate engine — read `affiliate_signups` (`slug`, `status`); do NOT modify it. Enrolled = `status == "approved"`.
- URL shapes (match `/affiliate/portal-data`): referral = `{quiz_url}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz`; recruit = `{public_base_url}/affiliate?ref={slug}`; signup = `{public_base_url}/affiliate/apply-form`.
- `_ambassador_block` is none-raising: missing table / no row → `{"status":"none", signup_url}`.
- `app.py` constants: `QUIZ_URL = "https://healing.scoreapp.com"`, `PUBLIC_BASE_URL` (env). `app.py`/HTML can't import offline → Tasks 2 & 3 verified live. Task 1 is offline-TDD.
- Offline test cmd: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`.

---

### Task 1: `_ambassador_block` + get_portal_view wiring

**Files:**
- Modify: `dashboard/portal_view.py`
- Test: `tests/test_portal_view_ambassador.py`

**Interfaces:**
- Produces: `_ambassador_block(cx, email, quiz_url, public_base_url) -> dict`; `get_portal_view(..., quiz_url="", public_base_url="")` gains an `"ambassador"` key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_view_ambassador.py
import sqlite3
from dashboard import portal_view as pv

QUIZ = "https://healing.scoreapp.com"
BASE = "https://illtowell.com"

def _cx_with_signups():
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE affiliate_signups (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, name TEXT,
        email TEXT UNIQUE, slug TEXT UNIQUE, token TEXT, status TEXT DEFAULT 'approved')""")
    return cx

def test_enrolled_returns_links():
    cx = _cx_with_signups()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) "
               "VALUES ('t','Amy','amy@example.com','amy7','tok','approved')")
    b = pv._ambassador_block(cx, "amy@example.com", QUIZ, BASE)
    assert b["status"] == "enrolled"
    assert b["slug"] == "amy7"
    assert b["referral_url"] == "https://healing.scoreapp.com?utm_source=amy7&utm_medium=affiliate&utm_campaign=scoreapp-quiz"
    assert b["recruit_url"] == "https://illtowell.com/affiliate?ref=amy7"

def test_pending_status():
    cx = _cx_with_signups()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) "
               "VALUES ('t','Pat','pat@example.com','pat3','tok2','pending')")
    assert pv._ambassador_block(cx, "pat@example.com", QUIZ, BASE) == {"status": "pending"}

def test_not_enrolled_returns_signup_url():
    cx = _cx_with_signups()
    b = pv._ambassador_block(cx, "nobody@example.com", QUIZ, BASE)
    assert b == {"status": "none", "signup_url": "https://illtowell.com/affiliate/apply-form"}

def test_email_lowercased():
    cx = _cx_with_signups()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) "
               "VALUES ('t','Amy','amy@example.com','amy7','tok','approved')")
    assert pv._ambassador_block(cx, "AMY@Example.COM", QUIZ, BASE)["status"] == "enrolled"

def test_missing_table_is_none():
    cx = sqlite3.connect(":memory:")  # no affiliate_signups table
    assert pv._ambassador_block(cx, "x@example.com", QUIZ, BASE)["status"] == "none"
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_view_ambassador.py -v`
Expected: FAIL — `AttributeError: ... '_ambassador_block'`.

- [ ] **Step 3: Implement**

In `dashboard/portal_view.py`, add the helper (near the other `_*_block` functions):

```python
def _ambassador_block(cx, email, quiz_url, public_base_url):
    """Affiliate/ambassador status for the personal portal, by email. None-raising.
    enrolled -> referral links (from slug); pending -> under review; else signup CTA."""
    em = (email or "").strip().lower()
    base = (public_base_url or "").rstrip("/")
    signup = {"status": "none", "signup_url": f"{base}/affiliate/apply-form"}
    if not em:
        return signup
    try:
        row = cx.execute(
            "SELECT slug, status FROM affiliate_signups WHERE lower(email)=? LIMIT 1",
            (em,)).fetchone()
    except Exception:
        return signup
    if not row:
        return signup
    slug, status = row[0], (row[1] or "")
    if status != "approved":
        return {"status": "pending"}
    return {
        "status": "enrolled",
        "slug": slug,
        "referral_url": f"{quiz_url}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz",
        "recruit_url": f"{base}/affiliate?ref={slug}",
    }
```

Then add the params + key to `get_portal_view` — change its signature to:
```python
def get_portal_view(cx, person_id, *, offers_enabled_keys=None, scan_date=None,
                    quiz_url="", public_base_url=""):
```
and add to the returned dict (next to `"upgrade": ...`):
```python
        "ambassador": _ambassador_block(cx, email, quiz_url, public_base_url),
```
(`email` is already derived earlier in `get_portal_view`.)

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_view_ambassador.py -v`
Expected: PASS (5 tests). If `get_portal_view` needs a `people` row for the existing-key assertion, add a minimal seed in a 6th test; the 5 above target `_ambassador_block` directly and don't need it.

- [ ] **Step 5: Commit**

```bash
git add dashboard/portal_view.py tests/test_portal_view_ambassador.py
git commit -m "feat(ambassador): portal_view _ambassador_block + get_portal_view key"
```

---

### Task 2: `/api/portal/<token>/view` passes the URL constants

**Files:**
- Modify: `app.py` — `api_client_portal_view`, the `get_portal_view(...)` call at ~line 11183.

**Why no offline test:** `app.py` can't import offline; verified live (Step 3).

- [ ] **Step 1: Edit the call**

Change the `_pv.get_portal_view(cx, ident.person_id, ...)` call (~line 11183) to also pass:
```python
                                   quiz_url=QUIZ_URL, public_base_url=PUBLIC_BASE_URL)
```
(Add as keyword args to the existing call; keep the existing `offers_enabled_keys=`/`scan_date=` args intact. `QUIZ_URL` and `PUBLIC_BASE_URL` are module-level constants in app.py.)

- [ ] **Step 2: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "feat(ambassador): pass QUIZ_URL/PUBLIC_BASE_URL to portal view"
```

- [ ] **Step 3: Live verification (post-deploy — record in report)**

```bash
# Karin (not an affiliate) -> ambassador.status == "none" + signup_url:
curl -s "https://illtowell.com/api/portal/jsgCzDpudlHyrm7VhvKMwFY8H3dQKcUW88CCu3NnGdQ/view" | python3 -c "import sys,json; a=json.load(sys.stdin).get('ambassador'); print(a)"
```
Expected: `{'status': 'none', 'signup_url': 'https://illtowell.com/affiliate/apply-form'}`.

---

### Task 3: Render the Ambassador card

**Files:**
- Modify: `static/client-portal.html` — the `render(data, view)` function (the `/view` payload is the `view` arg; `view.ambassador` is the block).

**Read first:** the `render(data, view)` function — find where it builds cards from `view` (account/orders/upgrade) to match markup/helpers (`esc`, card classes, button class).

- [ ] **Step 1: Add the card**

In `render(...)`, where `view`-derived cards are appended, add an Ambassador card driven by `view && view.ambassador`:
```javascript
  var amb = view && view.ambassador;
  if (amb) {
    if (amb.status === "enrolled") {
      html += '<div class="card"><h2>Your Ambassador links</h2>'
        + '<div class="kv"><span class="k">Share &amp; earn</span><span class="v"><a href="'+esc(amb.referral_url)+'" target="_blank" rel="noopener">'+esc(amb.referral_url)+'</a></span></div>'
        + '<div class="kv"><span class="k">Invite ambassadors</span><span class="v"><a href="'+esc(amb.recruit_url)+'" target="_blank" rel="noopener">'+esc(amb.recruit_url)+'</a></span></div>'
        + '</div>';
    } else if (amb.status === "pending") {
      html += '<div class="card"><h2>Ambassador</h2><p class="muted">Your ambassador application is under review.</p></div>';
    } else {
      html += '<div class="card"><h2>Become an Ambassador</h2>'
        + '<p class="muted">Earn rewards by sharing.</p>'
        + '<a class="btn" href="'+esc(amb.signup_url)+'" target="_blank" rel="noopener">Become an Ambassador</a></div>';
    }
  }
```
(Match the file's real `esc`, `card`, `btn`, `muted`, `kv/k/v` conventions — adapt the markup to whatever the existing `view` cards use; the above mirrors the account card's `kv` pattern.)

- [ ] **Step 2: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(ambassador): render Ambassador card in the personal portal"
```

- [ ] **Step 3: Live render-verify (post-deploy — record in report)**

Load Karin's portal (`/portal/<token>`); confirm the "Become an Ambassador" card shows with a working signup link and zero console errors. (With a seeded approved `affiliate_signups` row for a test email, the card shows the referral + recruit links.)

---

## Self-Review

**1. Spec coverage:** `_ambassador_block` (enrolled/pending/none, none-raising, URL shapes) → Task 1; get_portal_view key → Task 1; /view passes constants → Task 2; portal card → Task 3; live checks → Task 2/3. ✅
**2. Placeholder scan:** No TBD; full code in Task 1; Task 2/3 cite exact anchors + concrete live checks (the curl uses Karin's real token). ✅
**3. Type consistency:** `_ambassador_block(cx, email, quiz_url, public_base_url)` identical in Task 1 def + get_portal_view call; `get_portal_view(..., quiz_url, public_base_url)` matches the Task 2 call; the block's keys (`status`/`referral_url`/`recruit_url`/`signup_url`) match the Task 3 render. ✅
