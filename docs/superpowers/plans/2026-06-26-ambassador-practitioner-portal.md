# Ambassador in Practitioner Portal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Show the same 3-state Ambassador card on the practitioner portal, reusing the merged `dashboard/portal_view.py:_ambassador_block`.

**Architecture:** `/api/practitioner/portal-data` attaches `data["ambassador"]` (block run against `LOG_DB` using the practitioner's Supabase email); `static/practitioner-portal.html` renders the card.

**Tech Stack:** Python 3.11, Flask, sqlite3.

## Global Constraints

- Reuse `_ambassador_block` as-is (already merged + unit-tested in Step 1). No change to it or the affiliate engine or the client portal.
- Cross-DB: practitioner `email` from Supabase (already in `data`); `_ambassador_block` queries `affiliate_signups` in **`LOG_DB`** (sqlite). Join by email.
- The portal-data attach is **best-effort** (try/except → on error, `ambassador` absent, card hides) — never crash portal-data. Mirror the `branding` block already in that route.
- `app.py` + HTML can't import offline → both tasks verified LIVE (needs a practitioner session). Optional: `ast.parse` check on app.py.

---

### Task 1: Attach `ambassador` to practitioner portal-data

**Files:**
- Modify: `app.py` — `api_practitioner_portal_data` (route `/api/practitioner/portal-data`, ~line 9098), just before `return jsonify({"ok": True, **data})`.

- [ ] **Step 1: Add the attach block**

In `api_practitioner_portal_data`, after `data["branding"] = branding` and before the `return`, insert:

```python
    # Ambassador block (reuse the client-portal helper; affiliate_signups lives in LOG_DB,
    # joined to the practitioner by email). Best-effort — never crash portal-data.
    try:
        from dashboard import portal_view as _pv
        _amb_email = (data.get("email") or "").strip()
        with sqlite3.connect(LOG_DB) as _cx:
            data["ambassador"] = _pv._ambassador_block(
                _cx, _amb_email, QUIZ_URL, PUBLIC_BASE_URL)
    except Exception:
        pass
```

(`QUIZ_URL`, `PUBLIC_BASE_URL`, `LOG_DB` are all module-level in app.py and already used elsewhere — incl. the `branding` block in this same route.)

- [ ] **Step 2: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "feat(ambassador): attach ambassador block to practitioner portal-data"
```

- [ ] **Step 3: Live verification (post-deploy — needs a practitioner session; record in report)**

Sign in as a practitioner, then `GET /api/practitioner/portal-data` → response includes an `ambassador` key with `status` none/pending/enrolled (per that practitioner's `affiliate_signups`). A non-enrolled practitioner → `{"status":"none","signup_url":".../affiliate/apply-form"}`.

---

### Task 2: Render the Ambassador card on the practitioner portal

**Files:**
- Modify: `static/practitioner-portal.html` (markup: add a card container; `render(d)`: populate it).

**Read first:** `static/practitioner-portal.html` — the page uses a `$('id')` accessor and sets `.innerHTML`/`.textContent` on elements by id (see `render(d)` setting `$('status').innerHTML`, `$('who')`, etc.). Match that idiom. Find the escape helper if any; if none, the values are app-controlled (slug + fixed bases) — still prefer any existing escape.

- [ ] **Step 1: Add a container in the markup**

Add a card container near the other portal cards (e.g. after the status/wholesale card), matching the page's card markup/classes:
```html
<div id="ambassador-card" class="card" style="display:none"></div>
```
(Hidden by default; `render` reveals + fills it when `d.ambassador` is present.)

- [ ] **Step 2: Populate it in `render(d)`**

In `render(d)`, add (using the file's `$()` idiom + classes; adapt markup to the page's card style):
```javascript
  var amb = d.ambassador;
  var ac = $('ambassador-card');
  if (amb && ac) {
    ac.style.display = '';
    if (amb.status === 'enrolled') {
      ac.innerHTML = '<h3>Your Ambassador links</h3>'
        + '<p>Share &amp; earn: <a href="' + amb.referral_url + '" target="_blank" rel="noopener">' + amb.referral_url + '</a></p>'
        + '<p>Invite ambassadors: <a href="' + amb.recruit_url + '" target="_blank" rel="noopener">' + amb.recruit_url + '</a></p>';
    } else if (amb.status === 'pending') {
      ac.innerHTML = '<h3>Ambassador</h3><p>Your ambassador application is under review.</p>';
    } else {
      ac.innerHTML = '<h3>Become an Ambassador</h3>'
        + '<p>Earn rewards by sharing.</p>'
        + '<a class="btn" href="' + amb.signup_url + '" target="_blank" rel="noopener">Become an Ambassador</a>';
    }
  } else if (ac) {
    ac.style.display = 'none';
  }
```
Match the page's actual heading tag, `.card`, and button class (`.btn` or whatever it uses); if the file has an `esc()`/escape helper, wrap the URL values with it.

- [ ] **Step 3: Commit**

```bash
git add static/practitioner-portal.html
git commit -m "feat(ambassador): render Ambassador card on the practitioner portal"
```

- [ ] **Step 4: Live render-verify (post-deploy, practitioner session — record in report)**

Sign in as a practitioner and load `/practitioner/portal`: the Ambassador card shows the correct state (e.g. "Become an Ambassador" with a working link for a non-enrolled practitioner; referral + recruit links for an approved one), zero console errors.

---

## Self-Review

**1. Spec coverage:** portal-data attaches `ambassador` (best-effort, LOG_DB, email) → Task 1; practitioner-portal.html renders the 3-state card → Task 2; reuse `_ambassador_block` unchanged → both. ✅
**2. Placeholder scan:** No TBD; the app.py block is complete code; Task 2 gives reference JS + says to match the page's real idiom/classes (concrete file to read), not a vague instruction. ✅
**3. Type consistency:** `d.ambassador` keys (`status`/`referral_url`/`recruit_url`/`signup_url`) match what `_ambassador_block` emits (verified in Step 1) and what Task 2's render reads. ✅
