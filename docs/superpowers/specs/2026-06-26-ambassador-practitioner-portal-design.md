# Ambassador Section in the Practitioner Portal

**Date:** 2026-06-26
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** affiliate→personal-portal unification. Step 1 put the Ambassador section in the client portal (PR #330). This adds the same section to the **practitioner** portal, reusing the merged `_ambassador_block`.

## Problem

The Ambassador section now lives in the client portal but not the practitioner portal — so practitioners can't see/join the ambassador program from their own portal.

## Goal

Show the same 3-state Ambassador card (signup if not enrolled / pending / referral links if enrolled) on the practitioner portal, **reusing** `dashboard/portal_view.py:_ambassador_block` (already merged + unit-tested). No new core logic.

## Design

**Cross-DB join by email:** practitioners live in **Supabase** (`portal_data` selects their `email`); `affiliate_signups` lives in **sqlite `LOG_DB`**. They join by **email** — exactly the key `_ambassador_block(cx, email, …)` uses. So the block runs against `LOG_DB` with the practitioner's email.

### Component 1 — `/api/practitioner/portal-data` (`app.py`, ~line 9098)

After `data` is built (it already contains the practitioner's `email` from `portal_data`'s Supabase row) and before `return jsonify({"ok": True, **data})`, attach the ambassador block — best-effort, mirroring the `branding` try/except right above it so portal-data never crashes:

```python
    try:
        from dashboard import portal_view as _pv
        _amb_email = (data.get("email") or "").strip()
        with sqlite3.connect(LOG_DB) as _cx:
            data["ambassador"] = _pv._ambassador_block(
                _cx, _amb_email, QUIZ_URL, PUBLIC_BASE_URL)
    except Exception:
        pass
```

(`QUIZ_URL`/`PUBLIC_BASE_URL` are app.py module constants; `LOG_DB` is the sqlite path already used by the branding block in this same route.)

### Component 2 — `static/practitioner-portal.html` `render(d)`

This page uses the `$('id')` DOM idiom (sets `.innerHTML`/`.textContent` on elements by id), not string concatenation. So:
- Add a card **container** element to the page markup (e.g. `<div id="ambassador-card" class="card"></div>`) in a sensible spot among the existing cards.
- In `render(d)`, populate it from `d.ambassador` (guard `if (d.ambassador)`):
  - **enrolled** → "Your Ambassador links" with `referral_url` (share & earn) + `recruit_url` (invite ambassadors) as links.
  - **pending** → "Your ambassador application is under review."
  - **none** → "Become an Ambassador" + a button/link to `signup_url`.
- Match the practitioner portal's existing card markup, classes, and escaping (use the file's helpers; if it has no escape helper, the URLs are app-controlled `affiliate_signups` slugs + fixed bases, but still prefer the file's escaping if present). If `d.ambassador` is absent, hide/leave the container empty.

## Non-goals

- No change to `_ambassador_block` itself (reused as-is).
- Steps 2/3 of the unification (retire standalone `/affiliate/portal`, client/practitioner self-login changes, membership provisioning) — separate.
- No change to the affiliate engine or the client portal.

## Error handling

- The portal-data attach is best-effort (try/except → on any error, `ambassador` is simply absent and the card hides). Never breaks the practitioner portal.
- `_ambassador_block` is itself none-raising.

## Testing

`_ambassador_block` logic is already covered by Step 1's 5 unit tests; this change is route + HTML wiring only.

**Offline:** none new required (no new logic). Optionally a parse check: `python -c "import ast; ast.parse(open('app.py').read())"`.

**Live post-deploy (needs a practitioner session — `app.py`/HTML can't import offline):**
1. Sign in as a practitioner (a test/known practitioner account) → `GET /api/practitioner/portal-data` includes an `ambassador` key (`status` none/pending/enrolled per that practitioner's `affiliate_signups`).
2. Render-verify the practitioner portal page: the Ambassador card shows the correct state (e.g. "Become an Ambassador" for a non-enrolled practitioner) with a working link and zero console errors.
3. Spot-check: for a practitioner whose email IS an approved `affiliate_signups` row, the card shows the referral + recruit links.

## Rollout

Ships on merge → Render deploy. Verify via a practitioner login. (Because it needs a session, the go-live check is a practitioner login rather than a plain curl.)
