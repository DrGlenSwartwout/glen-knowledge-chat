# Affiliate Dashboard in Personal Portal (2b-1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Show the full affiliate dashboard (minus social-gamification) in the personal portal's Ambassador section, reusing the affiliate data via a shared `build_dashboard` helper.

**Architecture:** Extract the inline `/affiliate/portal-data` logic into `dashboard/affiliate_dashboard.py:build_dashboard(cx, slug, *, quiz_url, public_base_url)`; refactor the standalone route to use it; have `_ambassador_block` include `dashboard` when enrolled; render it in `client-portal.html`.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest.

## Global Constraints

- `build_dashboard` returns the EXACT dict `/affiliate/portal-data` returns today (incl. `social_links`) — the standalone page must stay byte-for-byte unchanged.
- All queries are `LOG_DB` (sqlite), keyed by `slug`. None-raising: unknown slug → `{}`; query errors degrade to zero/empty stats.
- `_ambassador_block`: `dashboard` is ADDITIVE to the enrolled branch; none/pending unchanged; a `build_dashboard` failure must not break the block (wrap it).
- Portal renders stats/links/offers/recent/recruit — NOT `social_links` (that's 2b-2). Client portal only (practitioner = follow-up).
- `app.py`/HTML can't import offline → Tasks 2 & 4 verified live. Tasks 1 & 3 are offline-TDD.
- Offline test cmd: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`.

---

### Task 1: `affiliate_dashboard.build_dashboard`

**Files:**
- Create: `dashboard/affiliate_dashboard.py`
- Test: `tests/test_affiliate_dashboard.py`

**Interfaces:**
- Produces: `build_dashboard(cx, slug, *, quiz_url, public_base_url) -> dict`; `_mask_lead_name(first, last) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_affiliate_dashboard.py
import sqlite3
from dashboard import affiliate_dashboard as ad

QUIZ = "https://healing.scoreapp.com"
BASE = "https://illtowell.com"

def _cx():
    cx = sqlite3.connect(":memory:")
    cx.executescript("""
      CREATE TABLE affiliate_signups (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
        name TEXT, email TEXT, organization TEXT DEFAULT '', slug TEXT, token TEXT,
        status TEXT DEFAULT 'approved', short_url TEXT DEFAULT '', referred_by TEXT DEFAULT '');
      CREATE TABLE referral_events (id INTEGER PRIMARY KEY AUTOINCREMENT, utm_source TEXT,
        received_at TEXT, first_name TEXT, last_name TEXT, quiz_score INTEGER);
      CREATE TABLE affiliate_conversions (id INTEGER PRIMARY KEY AUTOINCREMENT, affiliate_slug TEXT);
      CREATE TABLE affiliate_offers (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
        description TEXT, url_template TEXT, instructions TEXT, active INTEGER DEFAULT 1, sort_order INTEGER DEFAULT 0);
      CREATE TABLE affiliate_social_links (id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT,
        url TEXT, points INTEGER, views INTEGER, likes INTEGER, shares INTEGER, ts TEXT);
    """)
    return cx

def _seed(cx):
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,organization,slug,token,status) "
               "VALUES ('2026-01-01','Amy','amy@x.com','AmyCo','amy7','tok','approved')")
    cx.execute("INSERT INTO referral_events (utm_source,received_at,first_name,last_name,quiz_score) "
               "VALUES ('amy7','2026-06-01','Mary','Johnson',80)")
    cx.execute("INSERT INTO referral_events (utm_source,received_at,first_name,last_name,quiz_score) "
               "VALUES ('amy7','2026-06-10','Bob','Lee',55)")
    cx.execute("INSERT INTO affiliate_conversions (affiliate_slug) VALUES ('amy7')")
    cx.execute("INSERT INTO affiliate_offers (name,description,url_template,instructions,active,sort_order) "
               "VALUES ('Quiz','Take the quiz','https://q/{slug}','Share it',1,0)")
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status,referred_by) "
               "VALUES ('2026-02-01','Rec','r@x.com','rec1','tok2','approved','amy7')")
    cx.commit()

def test_build_dashboard_full():
    cx = _cx(); _seed(cx)
    d = ad.build_dashboard(cx, "amy7", quiz_url=QUIZ, public_base_url=BASE)
    assert d["name"] == "Amy" and d["organization"] == "AmyCo" and d["slug"] == "amy7"
    assert d["tracking_url"] == "https://healing.scoreapp.com?utm_source=amy7&utm_medium=affiliate&utm_campaign=scoreapp-quiz"
    assert d["recruit_url"] == "https://illtowell.com/affiliate?ref=amy7"
    assert d["total_leads"] == 2
    assert d["last_lead"] == "2026-06-10"
    assert d["recruited_count"] == 1
    assert d["conversions_count"] == 1
    assert d["member_since"] == "2026-01-01"
    assert d["offers"][0]["url"] == "https://q/amy7"
    names = [r["name"] for r in d["recent"]]
    assert "Mary J." in names and "Bob L." in names

def test_short_url_wins():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status,short_url) "
               "VALUES ('2026-01-01','Amy','a@x.com','amy7','tok','approved','https://sho.rt/x')")
    cx.commit()
    assert ad.build_dashboard(cx, "amy7", quiz_url=QUIZ, public_base_url=BASE)["tracking_url"] == "https://sho.rt/x"

def test_unknown_slug_empty():
    cx = _cx()
    assert ad.build_dashboard(cx, "nope", quiz_url=QUIZ, public_base_url=BASE) == {}

def test_mask_lead_name():
    assert ad._mask_lead_name("Mary", "Johnson") == "Mary J."
    assert ad._mask_lead_name("Mary", "") == "Mary"
    assert ad._mask_lead_name(None, None) == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_affiliate_dashboard.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** — `dashboard/affiliate_dashboard.py`:

```python
"""Affiliate/ambassador dashboard data — shared by the standalone /affiliate/portal-data
route and the personal portal's Ambassador section. Pure, LOG_DB-based, none-raising."""


def _mask_lead_name(first, last):
    fn = (first or "").strip()
    ln = (last or "").strip()
    if ln:
        return f"{fn} {ln[0]}.".strip()
    return fn


def build_dashboard(cx, slug, *, quiz_url, public_base_url):
    """Full affiliate dashboard dict for a slug. {} if the slug isn't an enrolled
    affiliate. Mirrors the legacy /affiliate/portal-data payload exactly."""
    cx.row_factory = None
    row = cx.execute(
        "SELECT name, organization, short_url, created_at FROM affiliate_signups WHERE slug=?",
        (slug,)).fetchone()
    if not row:
        return {}
    name, org, short_url, created_at = row[0], row[1] or "", row[2] or "", row[3] or ""
    base = (public_base_url or "").rstrip("/")
    long_url = f"{quiz_url}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz"
    tracking_url = short_url if short_url else long_url
    recruit_url = f"{base}/affiliate?ref={slug}"
    try:
        stats = cx.execute(
            "SELECT COUNT(*), MAX(received_at) FROM referral_events WHERE utm_source=?",
            (slug,)).fetchone()
        recent = cx.execute(
            "SELECT received_at, first_name, last_name, quiz_score FROM referral_events "
            "WHERE utm_source=? ORDER BY received_at DESC LIMIT 10", (slug,)).fetchall()
        recruited_count = cx.execute(
            "SELECT COUNT(*) FROM affiliate_signups WHERE referred_by=? AND status='approved'",
            (slug,)).fetchone()[0]
        conversions_count = cx.execute(
            "SELECT COUNT(*) FROM affiliate_conversions WHERE affiliate_slug=?",
            (slug,)).fetchone()[0]
        offers = cx.execute(
            "SELECT name, description, url_template, COALESCE(instructions,'') "
            "FROM affiliate_offers WHERE active=1 ORDER BY sort_order ASC").fetchall()
        social = cx.execute(
            "SELECT url, points, views, likes, shares, ts FROM affiliate_social_links "
            "WHERE slug=? ORDER BY id DESC", (slug,)).fetchall()
    except Exception:
        stats, recent, recruited_count, conversions_count, offers, social = None, [], 0, 0, [], []
    return {
        "name": name, "organization": org, "slug": slug,
        "tracking_url": tracking_url, "recruit_url": recruit_url,
        "total_leads": stats[0] if stats else 0,
        "last_lead": stats[1] if stats else None,
        "recruited_count": recruited_count,
        "conversions_count": conversions_count,
        "recent": [{"received_at": r[0], "name": _mask_lead_name(r[1], r[2]), "score": r[3]}
                   for r in recent],
        "offers": [{"name": o[0], "description": o[1],
                    "url": o[2].replace("{slug}", slug), "instructions": o[3]} for o in offers],
        "social_links": [{"url": s[0], "points": s[1], "views": s[2], "likes": s[3],
                          "shares": s[4], "ts": s[5]} for s in social],
        "member_since": created_at,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_affiliate_dashboard.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/affiliate_dashboard.py tests/test_affiliate_dashboard.py
git commit -m "feat(2b1): affiliate_dashboard.build_dashboard (shared dashboard data)"
```

---

### Task 2: Refactor `/affiliate/portal-data` to use `build_dashboard`

**Files:**
- Modify: `app.py` — `affiliate_portal_data` (route `/affiliate/portal-data`, ~line 8665).

**Why no offline test:** app.py can't import offline; verified live (standalone payload unchanged).

- [ ] **Step 1: Refactor**

Keep the token→row lookup + `if status != "approved": return ..., 403`. Replace everything from `long_url = ...` through the big `return jsonify({...})` with:
```python
    from dashboard import affiliate_dashboard as _ad
    with sqlite3.connect(LOG_DB) as cx:
        return jsonify(_ad.build_dashboard(cx, slug, quiz_url=QUIZ_URL, public_base_url=PUBLIC_BASE_URL))
```
(The `slug`/`status` come from the existing token lookup above. Remove the now-dead inline `stats/recent/recruited_count/conversions_count/offers/social` queries and the old return dict — they're in the module.)

- [ ] **Step 2: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "refactor(2b1): /affiliate/portal-data uses build_dashboard (DRY, unchanged payload)"
```

- [ ] **Step 3: Live verification (post-deploy — record in report)**

`GET /affiliate/portal-data?token=<an approved affiliate token>` → returns the same shape as before (name, slug, tracking_url, total_leads, recent, offers, social_links, member_since, …). Compare keys against the spec's list.

---

### Task 3: `_ambassador_block` includes `dashboard` when enrolled

**Files:**
- Modify: `dashboard/portal_view.py` (`_ambassador_block`, the enrolled return ~line 142-147)
- Test: `tests/test_portal_view_ambassador.py` (add a case)

- [ ] **Step 1: Write the failing test** — add to `tests/test_portal_view_ambassador.py`:

```python
def test_enrolled_includes_dashboard():
    cx = _cx_with_signups()  # existing helper in this file
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) "
               "VALUES ('2026-01-01','Amy','amy@example.com','amy7','tok','approved')")
    b = pv._ambassador_block(cx, "amy@example.com", QUIZ, BASE)
    assert b["status"] == "enrolled"
    assert isinstance(b.get("dashboard"), dict)
    assert b["dashboard"].get("slug") == "amy7"

def test_not_enrolled_has_no_dashboard():
    cx = _cx_with_signups()
    b = pv._ambassador_block(cx, "nobody@example.com", QUIZ, BASE)
    assert "dashboard" not in b
```

(If the file's `_cx_with_signups` helper lacks the columns `build_dashboard` needs — `created_at`, `organization`, `short_url`, `referred_by` — extend that CREATE TABLE in the helper to include them; the referral_events/offers/conversions/social tables may be absent, which is fine: `build_dashboard` degrades to zero/empty stats via its try/except.)

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_view_ambassador.py -v`
Expected: FAIL — `dashboard` key missing.

- [ ] **Step 3: Implement** — in `dashboard/portal_view.py`, add the import near the top (`from dashboard import affiliate_dashboard as _ad`) and change the enrolled return (lines ~142-147) to include `dashboard`, wrapped so a failure omits it:

```python
    block = {
        "status": "enrolled",
        "slug": slug,
        "referral_url": f"{quiz_url}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz",
        "recruit_url": f"{base}/affiliate?ref={slug}",
    }
    try:
        block["dashboard"] = _ad.build_dashboard(cx, slug, quiz_url=quiz_url, public_base_url=public_base_url)
    except Exception:
        pass
    return block
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_portal_view_ambassador.py -v`
Expected: PASS (the original 5 + 2 new = 7).

- [ ] **Step 5: Commit**

```bash
git add dashboard/portal_view.py tests/test_portal_view_ambassador.py
git commit -m "feat(2b1): _ambassador_block includes dashboard when enrolled"
```

---

### Task 4: Render the dashboard in the client portal

**Files:**
- Modify: `static/client-portal.html` — the Ambassador card's enrolled branch in `render(...)`.

**Read first:** the existing Ambassador card render (the `amb.status === "enrolled"` branch added in Step 1) + the page's `esc()` and `card`/`kv`/`k`/`v`/`btn`/`muted` helpers.

- [ ] **Step 1: Expand the enrolled branch**

In the `amb.status === "enrolled"` branch, in addition to the existing referral/recruit links, render `amb.dashboard` (guard `var dash = amb.dashboard || {};`):
- a stats row: `dash.total_leads` leads, last lead `dash.last_lead`, `dash.conversions_count` conversions, member since `dash.member_since`;
- offers: iterate `dash.offers` → name + description + a link to `o.url` + `o.instructions`;
- recent referrals: iterate `dash.recent` → `r.name` · score `r.score` · `r.received_at` (empty-state "No referrals yet" if none);
- recruit: `dash.recruit_url` + "`dash.recruited_count` recruited".
Match the page's card/kv idiom; `esc()` every value. **Do not render `dash.social_links`.**

- [ ] **Step 2: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(2b1): render affiliate dashboard in the client portal Ambassador card"
```

- [ ] **Step 3: Live render-verify (post-deploy — record in report)**

For a portal user whose email is an approved `affiliate_signups` row, load `/portal/<token>` → the Ambassador card shows the dashboard (stats/offers/recent/recruit), zero console errors. A non-enrolled user still shows the "Become an Ambassador" CTA.

---

## Self-Review

**1. Spec coverage:** build_dashboard (extract, exact payload) → Task 1; standalone refactor → Task 2; `_ambassador_block` dashboard → Task 3; portal render (no social) → Task 4. ✅
**2. Placeholder scan:** No TBD; Task 1/3 have complete code + tests; Task 2/4 give exact anchors + concrete live checks. ✅
**3. Type consistency:** `build_dashboard(cx, slug, *, quiz_url, public_base_url)` identical in Task 1 def, Task 2 call, Task 3 call; the `dashboard` dict keys (Task 1 output) match Task 4's render reads. ✅
