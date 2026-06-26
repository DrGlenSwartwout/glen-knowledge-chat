# Social-Links Gamification in the Personal Portal (Step 2b-2)

**Date:** 2026-06-26
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** affiliate→portal unification. 2b-1 brought the affiliate dashboard view into the portal. 2b-2 (this) adds the social-links display + submit. 2b-3 retires the standalone `/affiliate/portal`.

## Problem

An ambassador's social-share links (with points/views/likes/shares) and the form to add new ones live only in the standalone `/affiliate/portal`. The personal portal shows the dashboard (2b-1) but not the social-links list or the add-form.

## Goal

In the personal portal's enrolled Ambassador card: **display** the member's social links (already in the dashboard payload) and let them **add** new share URLs — authed by the portal identity (no affiliate token). Reuse the affiliate engine; don't build the points-awarding logic (display existing values only).

## Design

### Component 1 — `dashboard/affiliate_dashboard.py: add_social_links(cx, slug, email, urls) -> int` (extract, offline-tested)

The validate-and-insert loop currently inline in `/affiliate/social-links`:
- `CREATE TABLE IF NOT EXISTS affiliate_social_links (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, slug TEXT, email TEXT, url TEXT, points INTEGER DEFAULT 0, views INTEGER DEFAULT 0, likes INTEGER DEFAULT 0, shares INTEGER DEFAULT 0)` — idempotent (no-op on prod where the table exists; self-contained for fresh/test DBs). Matches the columns `build_dashboard` reads.
- For each url in `urls[:10]`: strip + truncate to 500 chars; skip unless it starts with `http://`/`https://`; `INSERT INTO affiliate_social_links (ts, slug, email, url) VALUES (?,?,?,?)` with `ts = now ISO`.
- `cx.commit()`; return the count inserted. None-raising on bad input (non-list `urls` → 0).

### Component 2 — refactor `/affiliate/social-links` (`app.py`)

Keep the token→`(slug,email,status)` lookup + approved gate. Replace the inline insert loop with:
```python
count = affiliate_dashboard.add_social_links(cx, slug, email, urls)
```
Behavior-preserving (same `{ok, count}`).

### Component 3 — new `POST /api/portal/<token>/social-links` (`app.py`)

Mirror the existing `/api/portal/<token>/*` identity pattern:
```python
sess = request.cookies.get("rm_portal_session", "")
urls = (request.get_json(silent=True) or {}).get("urls") or []
with _db_lock, sqlite3.connect(LOG_DB) as cx:
    _cp.init_client_portal_table(cx); _pi._ensure_people_table(cx)
    ident = _pi.resolve_identity(cx, token=token, session_token=sess, client_login_enabled=_client_login_enabled())
    if ident is None: return jsonify({"error": "not found"}), 404
    row = cx.execute("SELECT slug FROM affiliate_signups WHERE lower(email)=? AND status='approved' LIMIT 1",
                     (ident.email,)).fetchone()
    if not row: return jsonify({"error": "not an approved ambassador"}), 403
    count = affiliate_dashboard.add_social_links(cx, row[0], ident.email, urls)
return jsonify({"ok": True, "count": count})
```
(Only the enrolled/approved ambassador whose portal this is can add. `ident.email` is the authenticated identity, never client-supplied.)

### Component 4 — render in `static/client-portal.html`

In the enrolled Ambassador card (where 2b-1 renders the dashboard), add:
- **Social links list:** iterate `dash.social_links` → each `url` (link) + a small "pts/views/likes/shares" line; empty-state "No social shares yet." Escape all values via `esc()`.
- **Add form:** a textarea/input for one-or-more URLs + a button → `POST /api/portal/<seg>/social-links` `{urls:[...]}` → on success, reload the view (so the new links appear). Use the page's existing fetch/idiom (like the other `/api/portal/<seg>/*` POSTs).

## Non-goals

- The points/views/likes/shares **awarding** logic (whatever populates those today is unchanged — we display + let members add URLs).
- Retiring the standalone `/affiliate/portal` (2b-3).
- Practitioner portal (follow-up).
- Any change to enrollment or the affiliate token.

## Error handling

- `add_social_links` none-raising (non-list urls → 0; the CREATE-IF-NOT-EXISTS makes it self-contained).
- The portal endpoint: 404 if identity unresolved, 403 if the user isn't an approved ambassador — never 500 on normal misuse.

## Testing

**Offline (tmp sqlite) — `add_social_links`:**
1. 3 urls (2 http(s), 1 non-http) → count 2; only the http(s) rows inserted with the right slug/email/ts.
2. >10 urls → caps at 10. Non-list `urls` (e.g. None) → 0.
3. Works on a fresh DB (the CREATE IF NOT EXISTS runs) — no pre-created table needed.

**Live post-deploy (`app.py`/HTML can't import offline):**
4. `/affiliate/social-links` (standalone, approved token) still returns `{ok, count}` and inserts — unchanged.
5. `POST /api/portal/<token>/social-links` for an enrolled portal user → `{ok, count}`; a non-enrolled portal user → 403; then `/api/portal/<token>/view` shows the new link in `ambassador.dashboard.social_links`.
6. Render-verify: the enrolled Ambassador card shows the social-links list + add-form; adding a URL persists and re-renders; zero console errors.

## Rollout

Ships on merge → Render deploy. Verify standalone unchanged + an enrolled portal user can view/add social links. 2b-3 (retire standalone) follows.
